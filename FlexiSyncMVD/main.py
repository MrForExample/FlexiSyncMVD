import os
from os.path import join, isdir, isfile, abspath, dirname, basename, splitext
from pathlib import Path
from datetime import datetime
import torch

from transformers import CLIPVisionModelWithProjection
from diffusers import (
	StableDiffusionControlNetPipeline,
	StableDiffusionXLControlNetPipeline,
	ControlNetModel,
	AutoencoderKL,
	UNet2DConditionModel,
)
from diffusers import (
	DDPMScheduler,
)
from diffusers.utils import load_image

from src.mesh_processor import MeshProcessor
from src.pipeline import StableSyncMVDPipeline
from src.pipeline_XL import StableSyncMVDPipelineXL
from src.configs import *
from shutil import copy

def load_config(opt):

	if opt.mesh_config_relative:
		mesh_path = join(dirname(opt.config), opt.mesh)
	else:
		mesh_path = abspath(opt.mesh)

	if opt.output:
		output_root = abspath(opt.output)
	else:
		output_root = dirname(opt.config)

	output_name_components = []
	if opt.prefix and opt.prefix != "":
		output_name_components.append(opt.prefix)
	if opt.use_mesh_name:
		mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
		output_name_components.append(mesh_name)

	if opt.timeformat and opt.timeformat != "":
		output_name_components.append(datetime.now().strftime(opt.timeformat))
	output_name = "_".join(output_name_components)
	output_dir = join(output_root, '_exp', output_name)

	if not isdir(output_dir):
		os.makedirs(output_dir, exist_ok=True)
	else:
		print(f"Results exist in the output directory, use time string to avoid name collision.")
		exit(0)

	print(f"Saving to {output_dir}")

	copy(opt.config, join(output_dir, "config.yaml"))

	logging_config = {
		"output_dir":output_dir, 
		# "output_name":None, 
		# "intermediate":False, 
		"log_interval":opt.log_interval,
		"view_fast_preview": opt.view_fast_preview,
		"tex_fast_preview": opt.tex_fast_preview,
	}
	return logging_config, output_root, mesh_path
	
def load_pipeline(opt):
	if opt.t2i_model == "SD1.5":

		if opt.cond_type == "normal":
			controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", variant="fp16", torch_dtype=torch.float16)
		elif opt.cond_type == "depth":
			controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)
		elif opt.cond_type == "canny":
			controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", variant="fp16", torch_dtype=torch.float16)
		else:
			ValueError(f"Condition {opt.cond_type} is not supported")

		image_encoder = CLIPVisionModelWithProjection.from_pretrained(
			"h94/IP-Adapter",
			subfolder="models/image_encoder",
			torch_dtype=torch.float16
		)

		pipe = StableDiffusionControlNetPipeline.from_pretrained(
			"runwayml/stable-diffusion-v1-5", controlnet=controlnet, image_encoder=image_encoder, torch_dtype=torch.float16
		)

		pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

		syncmvd = StableSyncMVDPipeline(**pipe.components)
	
	elif opt.t2i_model == "SDXL":
		# xinsir/controlnet doesn't work well with this pipeline
		if opt.cond_type == "depth":
			#controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
			controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
		elif opt.cond_type == "canny":
			#controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
			controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
		elif opt.cond_type == "render":
			controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-tile-sdxl-1.0", torch_dtype=torch.float16)
		else:
			ValueError(f"Condition {opt.cond_type} is not supported")

		image_encoder = CLIPVisionModelWithProjection.from_pretrained(
			"h94/IP-Adapter",
			subfolder="models/image_encoder",
			torch_dtype=torch.float16
		)

		if opt.checkpoint_path:
			vae = AutoencoderKL.from_single_file(opt.checkpoint_path, torch_dtype=torch.float16)
			unet = UNet2DConditionModel.from_single_file(opt.checkpoint_path, torch_dtype=torch.float16)

			pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0", image_encoder=image_encoder, vae=vae, unet=unet, controlnet=controlnet, torch_dtype=torch.float16
			)
		else:
			vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
			
			pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
				"stabilityai/stable-diffusion-xl-base-1.0", image_encoder=image_encoder, vae=vae, controlnet=controlnet, torch_dtype=torch.float16
			)

		#pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
		pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

		syncmvd = StableSyncMVDPipelineXL(**pipe.components)
  
	return syncmvd

def initialize_pipeline(syncmvd, opt, logging_config):
	syncmvd.initialize_pipeline(
		camera_azims=opt.camera_azims,
		top_cameras=not opt.no_top_cameras,
		ref_views=[],
		latent_size=opt.latent_view_size,
		max_batch_size=opt.max_batch_size,
		logging_config=logging_config
	) 

def initialize_mesh_processor(opt, output_root, mesh_path, camera_poses, device):
	import pickle
	import hashlib

	mesh_data_root_dir = join(output_root, "_mesh_data")
	os.makedirs(mesh_data_root_dir, exist_ok=True)
	mesh_name = splitext(basename(mesh_path))[0]
	# Convert parameters to a string before hashing
	data_string = f"{mesh_name}{opt.mesh_scale}{opt.latent_view_size}{opt.rgb_view_size}{opt.latent_tex_size}{opt.rgb_tex_size}{camera_poses}"
	data_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
	mesh_data_dir = join(mesh_data_root_dir, data_hash)
	os.makedirs(mesh_data_dir, exist_ok=True)
	pickle_data_path = join(mesh_data_dir, "mesh.pkl")

	recalculated = False
	if isfile(pickle_data_path):
		with open(pickle_data_path, 'rb') as inp:
			mesh_processor = pickle.load(inp)
		print(f"Loaded mesh data from {pickle_data_path}")
	else:
		mesh_processor = MeshProcessor(
			mesh_path=mesh_path,
			mesh_transform={"scale":opt.mesh_scale},
			mesh_autouv=not opt.keep_mesh_uv,
			latent_size=opt.latent_view_size,
			render_rgb_size=opt.rgb_view_size,
			latent_texture_size=opt.latent_tex_size,
			texture_rgb_size=opt.rgb_tex_size,
			camera_poses=camera_poses,
			device=device,
		)
		recalculated = True

		with open(pickle_data_path, 'wb') as outp:
			pickle.dump(mesh_processor, outp, pickle.HIGHEST_PROTOCOL)
			print(f"Saved mesh data to {pickle_data_path}")

	return mesh_processor, recalculated, mesh_data_dir

def load_cond_t2i_pipeline(unet, vae, controlnet, device):
	cond_t2i_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
		"stabilityai/stable-diffusion-xl-base-1.0", 
		unet=unet,
		vae=vae,
		controlnet=controlnet, 
		torch_dtype=torch.float16
	).to(device)

	cond_t2i_pipeline.scheduler = DDPMScheduler.from_config(cond_t2i_pipeline.scheduler.config)

	return cond_t2i_pipeline

def camera_rendering(output_dir, mesh_processor, height, width, batch_size, guess_mode, num_images_per_prompt=1, do_classifier_free_guidance=True, cond_type="depth", render_cam_index=None, device="cuda", dtype=torch.float16):	
	mesh_processor.uvp.to(device)
	conditioning_images, masks = mesh_processor.get_conditioning_images(height, cond_type=cond_type, render_cam_index=render_cam_index)
	conditioning_images = conditioning_images.to(device, dtype=dtype)
	if cond_type == "canny":
		conditioning_images = mesh_processor.get_canny_images(conditioning_images, width, height, batch_size, num_images_per_prompt, do_classifier_free_guidance, guess_mode, device, dtype)
	list_pil = mesh_processor.save_conditioning_images(conditioning_images, output_dir, cond_type=cond_type)
	return list_pil

def load_cond_image(cond_image_dir):
	return load_image(f"{cond_image_dir}/cond.jpg")

def run_cond_t2i_pipeline(opt, cond_t2i_pipeline, cond_image, num_inference_steps=30, seed=0, conditioning_scale=0.5, guidance_scale=5.0, num_images_per_prompt=1, output_dir=None):
	image = cond_t2i_pipeline(
		prompt=opt.prompt, 
		negative_prompt=opt.negative_prompt, 
		image=cond_image,
		height=opt.latent_view_size*8,
		width=opt.latent_view_size*8,
		num_inference_steps=num_inference_steps, 
		generator=torch.manual_seed(seed),
		controlnet_conditioning_scale=conditioning_scale,
		guidance_scale=guidance_scale,
		num_images_per_prompt=num_images_per_prompt,
		#clip_skip=2,
	).images[0]

	# Mesh Front view condition generated reference image
	if output_dir:
		image.save(join(output_dir, f"generated_ip_image_{datetime.now().strftime('%d%b%Y-%H%M%S')}.png"))

	return image

def load_reference_image(opt):
	return load_image(opt.ip_adapter_image) if opt.ip_adapter_image else None

def run_pipeline(opt, syncmvd, logging_config, mesh_processor, ip_adapter_image=None):
	result_tex_rgb, textured_views, v = syncmvd(
		prompt=opt.prompt,
		height=opt.latent_view_size*8,
		width=opt.latent_view_size*8,
		num_inference_steps=opt.steps,
		guidance_scale=opt.guidance_scale,
		negative_prompt=opt.negative_prompt,
		ip_adapter_image=ip_adapter_image,
		ip_adapter_scale=opt.ip_adapter_scale,
		
		generator=torch.manual_seed(opt.seed),
		controlnet_guess_mode=opt.guess_mode,
		controlnet_conditioning_scale = opt.conditioning_scale,
		controlnet_conditioning_end_scale= opt.conditioning_scale_end,
		control_guidance_start= opt.control_guidance_start,
		control_guidance_end = opt.control_guidance_end,
		guidance_rescale = opt.guidance_rescale,

		mesh_processor=mesh_processor,

		multiview_diffusion_end=opt.mvd_end,
		exp_start=opt.mvd_exp_start,
		exp_end=opt.mvd_exp_end,
		ref_attention_end=opt.ref_attention_end,
		shuffle_background_change=opt.shuffle_bg_change,
		shuffle_background_end=opt.shuffle_bg_end,

		logging_config=logging_config,
  
		cond_type=opt.cond_type,
	)

def run_experiment(runtime_cond_t2i=False):
	#region ########## Only need run once when system start ##########
	# Load all the user configurations
	opt = parse_config()
	
	# Load the logging configuration and mesh path
	logging_config, output_root, mesh_path = load_config(opt)

	# Load the pipeline and all the model into vram
	syncmvd = load_pipeline(opt)

	# Setup pipeline (camera, background color, etc.) settings
	initialize_pipeline(syncmvd, opt, logging_config)
	#endregion ########## Only need run once when system start ##########

	#region ########## Only need run once per mesh uploaded by user ##########
	# Load & preprocess mesh, initialize mesh renderer & UV projector
	mesh_processor, recalculated, mesh_data_dir = initialize_mesh_processor(opt, output_root, mesh_path, syncmvd.camera_poses, syncmvd._execution_device)
	#endregion ########## Only need run once per mesh uploaded by user ##########

	#region ########## Run as many times as user want for given preprocessed mesh ##########
	# Inference using pipeline to generate texture
	if runtime_cond_t2i:
		cond_t2i_pipeline = load_cond_t2i_pipeline(syncmvd.unet, syncmvd.vae, syncmvd.controlnet, syncmvd._execution_device)
		if recalculated:
			cond_image = camera_rendering(mesh_data_dir, mesh_processor, opt.latent_view_size*8, opt.latent_view_size*8, opt.max_batch_size, opt.guess_mode, render_cam_index=4, device=syncmvd._execution_device)
		else:
			cond_image = load_cond_image(mesh_data_dir)
		ip_adapter_image = run_cond_t2i_pipeline(opt, cond_t2i_pipeline, cond_image, num_inference_steps=30, seed=0, conditioning_scale=0.5, guidance_scale=5.0, num_images_per_prompt=1, output_dir=logging_config["output_dir"])
	else:
		ip_adapter_image = load_reference_image(opt)
	run_pipeline(opt, syncmvd, logging_config, mesh_processor, ip_adapter_image)
	#endregion ########## Run as many times as user want for given preprocessed mesh ##########

if __name__ == "__main__":
	run_experiment(runtime_cond_t2i=True)