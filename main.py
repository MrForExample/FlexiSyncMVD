import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import (
	StableDiffusionControlNetPipeline,
	StableDiffusionXLControlNetPipeline,
	ControlNetModel,
	AutoencoderKL,
)
from diffusers import (
	DDPMScheduler,
)
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
	return logging_config, mesh_path
	
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

		pipe = StableDiffusionControlNetPipeline.from_pretrained(
			"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
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
		else:
			ValueError(f"Condition {opt.cond_type} is not supported")

		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
		pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
			"stabilityai/stable-diffusion-xl-base-1.0", vae=vae, controlnet=controlnet, torch_dtype=torch.float16
		)

		#pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
		pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

		syncmvd = StableSyncMVDPipelineXL(**pipe.components)
  
	return syncmvd
	
def run_pipeline(opt, syncmvd, logging_config, mesh_path):
	result_tex_rgb, textured_views, v = syncmvd(
		prompt=opt.prompt,
		height=opt.latent_view_size*8,
		width=opt.latent_view_size*8,
		num_inference_steps=opt.steps,
		guidance_scale=opt.guidance_scale,
		negative_prompt=opt.negative_prompt,
		
		generator=torch.manual_seed(opt.seed),
		max_batch_size=48,
		controlnet_guess_mode=opt.guess_mode,
		controlnet_conditioning_scale = opt.conditioning_scale,
		controlnet_conditioning_end_scale= opt.conditioning_scale_end,
		control_guidance_start= opt.control_guidance_start,
		control_guidance_end = opt.control_guidance_end,
		guidance_rescale = opt.guidance_rescale,
		use_directional_prompt=True,

		mesh_path=mesh_path,
		mesh_transform={"scale":opt.mesh_scale},
		mesh_autouv=not opt.keep_mesh_uv,

		camera_azims=opt.camera_azims,
		top_cameras=not opt.no_top_cameras,
		texture_size=opt.latent_tex_size,
		render_rgb_size=opt.rgb_view_size,
		texture_rgb_size=opt.rgb_tex_size,
		multiview_diffusion_end=opt.mvd_end,
		exp_start=opt.mvd_exp_start,
		exp_end=opt.mvd_exp_end,
		ref_attention_end=opt.ref_attention_end,
		shuffle_background_change=opt.shuffle_bg_change,
		shuffle_background_end=opt.shuffle_bg_end,

		logging_config=logging_config,
  
		cond_type=opt.cond_type,
	)

def run_experiment():
	# Load all the user configurations
	opt = parse_config()
	
	# Load the logging configuration and mesh path
	logging_config, mesh_path = load_config(opt)

	# Load the pipeline and all the model into vram
	syncmvd = load_pipeline(opt)

	# Inference using pipeline to generate texture
	run_pipeline(opt, syncmvd, logging_config, mesh_path)

if __name__ == "__main__":
	run_experiment()