import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from IPython.display import display
import numpy as np
from numpy import ndarray
from PIL.Image import Image as PILImage
import random
import torch
from torch import Tensor
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.utils import numpy_to_pil

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from .renderer.project import UVProjection as UVP


from .syncmvd.attention import SamplewiseAttnProcessor2_0, replace_attention_processors
from .syncmvd.prompt import *
from .syncmvd.step import step_tex
from .utils import *

from timeit import default_timer as timer



if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")

# Background colors
color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
			"red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
			"green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
			"aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
			"purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
	"""
	Compose rendered views at time t by blending foregrounds with backgrounds given the alpha mask.

	Args:
		scheduler (Scheduler): Scheduler to sample from.
		backgrounds (List[Tensor]): List of background image tensors.
		foregrounds (List[Tensor]): List of foreground image tensors.
		masks (List[Tensor]): List of alpha mask tensors.
		t (int): Timestep to composite at.

	Returns:
		Tensor: Composited images tensor of shape (B, H, W, C)
	"""
	composited_images = []
	for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
		if t > 0:
			alphas_cumprod = scheduler.alphas_cumprod[t]
			noise = torch.normal(0, 1, background.shape, device=background.device)
			background = (1-alphas_cumprod) * noise + alphas_cumprod * background
		composited = foreground * mask + background * (1-mask)
		composited_images.append(composited)
	composited_tensor = torch.stack(composited_images)
	return composited_tensor


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no possitive effect and use a large "max_batch_size" to skip splitting
def split_groups(attention_mask, max_batch_size, ref_view=[]):
	"""
	Split attention mask into micro-batches to fit into a smaller model.
	
	Args:
		attention_mask (List[List[int]]): Attention mask for each mesh.
		max_batch_size (int): Maximum number of elements in each micro-batch.
		ref_view (List[int], optional): Reference view indices. Defaults to [].
	
	Returns:
		List[List[int]]: List of micro-batch information.
	"""
	group_sets = []
	group = set()
	ref_group = set()
	idx = 0
	while idx < len(attention_mask):
		new_group = group | set([idx])
		new_ref_group = (ref_group | set(attention_mask[idx] + ref_view)) - new_group 
		if len(new_group) + len(new_ref_group) <= max_batch_size:
			group = new_group
			ref_group = new_ref_group
			idx += 1
		else:
			assert len(group) != 0, "Cannot fit into a group"
			group_sets.append((group, ref_group))
			group = set()
			ref_group = set()
	if len(group)>0:
		group_sets.append((group, ref_group))

	group_metas = []
	for group, ref_group in group_sets:
		in_mask = sorted(list(group | ref_group))
		out_mask = []
		group_attention_masks = []
		for idx in in_mask:
			if idx in group:
				out_mask.append(in_mask.index(idx))
			group_attention_masks.append([in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask])
		ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
		group_metas.append([in_mask, out_mask, group_attention_masks, ref_attention_mask])

	return group_metas

'''

	MultiView-Diffusion Stable-Diffusion Pipeline
	Modified from a Diffusers StableDiffusionControlNetPipeline
	Just mimic the pipeline structure but did not follow any API convention

'''

class StableSyncMVDPipelineXL(StableDiffusionXLControlNetPipeline):
	def __init__(
		self, 
		vae: AutoencoderKL,
		text_encoder: CLIPTextModel,
		text_encoder_2: CLIPTextModelWithProjection,
		tokenizer: CLIPTokenizer,
		tokenizer_2: CLIPTokenizer,
		unet: UNet2DConditionModel,
		controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
		scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
		feature_extractor: CLIPImageProcessor = None,
  		image_encoder: CLIPVisionModelWithProjection = None,
	):
		super().__init__(
			vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, 
			controlnet, scheduler, force_zeros_for_empty_prompt, add_watermarker, 
			feature_extractor, image_encoder
		)

		self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
		self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
		self.enable_model_cpu_offload()
		self.enable_vae_slicing()
		self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

	# Setup camera, attention_mask and background color settings
	def initialize_pipeline(
			self,
			camera_azims=None,
			camera_centers=None,
			top_cameras=True,
			ref_views=[],
			latent_size=None,
			max_batch_size=24,
			logging_config=None,
		):
		# Make output dir
		"""
		Initialize the pipeline, including creating the output and intermediate directories, and defining the cameras and attention masks for rendering.
		
		Parameters:
		camera_azims (list of float): The azimuthal angles of the cameras in degrees.
		camera_centers (list of float): The centers of the cameras in x, y, z coordinates.
		top_cameras (bool): Whether to add two additional cameras for painting the top surfaces.
		ref_views (list of int): The indices of the reference views for attention.
		latent_size (int): The size of the latent space.
		max_batch_size (int): The maximum batch size for rendering.
		logging_config (dict): The configuration for logging.
		
		Returns:
		None
		"""
		output_dir = logging_config["output_dir"]

		self.result_dir = f"{output_dir}/results"
		self.intermediate_dir = f"{output_dir}/intermediate"

		dirs = [output_dir, self.result_dir, self.intermediate_dir]
		for dir_ in dirs:
			if not os.path.isdir(dir_):
				os.mkdir(dir_)


		# Define the cameras for rendering
		self.camera_poses = []
		self.attention_mask=[]
		self.centers = camera_centers

		cam_count = len(camera_azims)
		front_view_diff = 360
		back_view_diff = 360
		front_view_idx = 0
		back_view_idx = 0
		for i, azim in enumerate(camera_azims):
			if azim < 0:
				azim += 360
			self.camera_poses.append((0, azim))
			self.attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
			if abs(azim) < front_view_diff:
				front_view_idx = i
				front_view_diff = abs(azim)
			if abs(azim - 180) < back_view_diff:
				back_view_idx = i
				back_view_diff = abs(azim - 180)

		# Add two additional cameras for painting the top surfaces
		if top_cameras:
			self.camera_poses.append((30, 0))
			self.camera_poses.append((30, 180))

			self.attention_mask.append([front_view_idx, cam_count])
			self.attention_mask.append([back_view_idx, cam_count+1])

		# Reference view for attention (all views attend the the views in this list)
		# A forward view will be used if not specified
		if len(ref_views) == 0:
			ref_views = [front_view_idx]

		# Calculate in-group attention mask
		self.group_metas = split_groups(self.attention_mask, max_batch_size, ref_views)

		# Initialize random background color
		color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1,3,1,1).to(dtype=self.text_encoder.dtype, device=self._execution_device)
		color_images = torch.ones(
			(1,1,latent_size*8, latent_size*8), 
			device=self._execution_device, 
			dtype=self.text_encoder.dtype
		) * color_images
		color_images = ((0.5*color_images)+0.5)
		color_latents = encode_latents(self.vae.to(torch.float32), color_images.to(torch.float32)).to(torch.float16)
		assert not torch.isnan(color_latents).any(), "color_latents contains nan values"
		self.color_latents = {color[0]:color[1] for color in zip(color_names, [latent for latent in color_latents])}
		self.vae = self.vae.to(torch.float16)
		#self.vae = self.vae.to(device="cpu", dtype=torch.float16)

		print("Done Initialization")


	'''
		Modified from a StableDiffusion ControlNet pipeline
		Multi ControlNet not supported yet
	'''
	@torch.no_grad()
	def __call__(
		self,
		prompt: str = None,
		prompt_2: str = None,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
		negative_prompt: str = None,
		negative_prompt_2: str = None,
		ip_adapter_image: Optional[Union[PILImage, ndarray, Tensor]] = None,
		ip_adapter_scale: float = 1.0,
		
		
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: int = 1,
		
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		controlnet_guess_mode: bool = False,
		controlnet_conditioning_scale: Union[float, List[float]] = 0.7,
		controlnet_conditioning_end_scale: Union[float, List[float]] = 0.9,
		control_guidance_start: Union[float, List[float]] = 0.0,
		control_guidance_end: Union[float, List[float]] = 0.99,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
		guidance_rescale: float = 0.0,

		mesh_processor= None,

		multiview_diffusion_end=0.8,
		exp_start=0.0,
		exp_end=6.0,
		shuffle_background_change=0.4,
		shuffle_background_end=0.99, #0.4
		
		ref_attention_end=0.2,

		logging_config=None,
		cond_type="depth",
	):		
		num_timesteps = self.scheduler.config.num_train_timesteps
		initial_controlnet_conditioning_scale = controlnet_conditioning_scale
		log_interval = logging_config.get("log_interval", 10)
		view_fast_preview = logging_config.get("view_fast_preview", True)
		tex_fast_preview = logging_config.get("tex_fast_preview", True)

		controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

		# align format for control guidance
		if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
			control_guidance_start = len(control_guidance_end) * [control_guidance_start]
		elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
			control_guidance_end = len(control_guidance_start) * [control_guidance_end]
		elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
			# mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
			mult = 1
			control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
				control_guidance_end
			]


		# 0. Default height and width to unet
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor


		# 1. Check inputs. Raise error if not correct
		self.check_inputs(
			prompt,
			prompt_2,
			torch.zeros((1,3,height,width), device=self._execution_device),
			callback_steps,
			negative_prompt,
			negative_prompt_2,
			prompt_embeds=None,
			negative_prompt_embeds=None,
			ip_adapter_image=ip_adapter_image,
			ip_adapter_image_embeds=None,
			controlnet_conditioning_scale=controlnet_conditioning_scale,
			control_guidance_start=control_guidance_start,
			control_guidance_end=control_guidance_end,
		)


		# 2. Define call parameters
		if prompt is not None and isinstance(prompt, list):
			assert len(prompt) == 1 and len(negative_prompt) == 1, "Only implemented for 1 (negative) prompt"  
		assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
		batch_size = len(mesh_processor.uvp.cameras)


		device = self._execution_device
		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		do_classifier_free_guidance = guidance_scale > 1.0 and self.unet.config.time_cond_proj_dim is None

		# if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
		# 	controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

		global_pool_conditions = (
			controlnet.config.global_pool_conditions
			if isinstance(controlnet, ControlNetModel)
			else controlnet.nets[0].config.global_pool_conditions
		)
		guess_mode = controlnet_guess_mode or global_pool_conditions


		# 3. Encode input prompt
		prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

		text_encoder_lora_scale = (
			cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
		)
		prompt_embeds_tuple = self.encode_prompt(
			prompt,
			prompt_2,
			device,
			num_images_per_prompt,
			do_classifier_free_guidance,
			negative_prompt,
			negative_prompt_2,
			prompt_embeds=None,
			negative_prompt_embeds=None,
			lora_scale=text_encoder_lora_scale,
		)
  
		positive_prompt_embeds, negative_prompt_embeds = prompt_embeds_tuple[0], prompt_embeds_tuple[1]
		prompt_embed_dict = dict(zip(direction_names, [emb for emb in positive_prompt_embeds]))
		negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_prompt_embeds]))
  
		pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_embeds_tuple[2], prompt_embeds_tuple[3]
		pooled_prompt_embed_dict = dict(zip(direction_names, [emb for emb in pooled_prompt_embeds]))
		pooled_negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_pooled_prompt_embeds]))


		# mix prompt embeds according to azim angle
		positive_prompt_embeds = [azim_prompt(prompt_embed_dict, pose) for pose in self.camera_poses]
		positive_prompt_embeds = torch.stack(positive_prompt_embeds, axis=0)

		negative_prompt_embeds = [azim_neg_prompt(negative_prompt_embed_dict, pose) for pose in self.camera_poses]
		negative_prompt_embeds = torch.stack(negative_prompt_embeds, axis=0)

		pooled_positive_prompt_embeds = [azim_prompt(pooled_prompt_embed_dict, pose) for pose in self.camera_poses]
		pooled_positive_prompt_embeds = torch.stack(pooled_positive_prompt_embeds, axis=0)

		pooled_negative_prompt_embeds = [azim_neg_prompt(pooled_negative_prompt_embed_dict, pose) for pose in self.camera_poses]
		pooled_negative_prompt_embeds = torch.stack(pooled_negative_prompt_embeds, axis=0)
  
		if self.text_encoder_2 is None:
			text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
		else:
			text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

		if ip_adapter_image is not None:
			self.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
			self.set_ip_adapter_scale(ip_adapter_scale)

			all_image_embeds = self.prepare_ip_adapter_image_embeds(
				ip_adapter_image,
				None,
				device,
				batch_size * num_images_per_prompt,
				do_classifier_free_guidance,
			)

			negative_image_embeds, positive_image_embeds = torch.chunk(all_image_embeds[0], 2)
		else:
			negative_image_embeds = torch.zeros_like(negative_prompt_embeds)
			positive_image_embeds = torch.zeros_like(positive_prompt_embeds)
   
		#print(f"pooled_prompt_embeds: {pooled_prompt_embeds.shape}; positive_prompt_embeds: {positive_prompt_embeds.shape}")
		#pooled_prompt_embeds: torch.Size([6, 1280]); positive_prompt_embeds: torch.Size([6, 77, 2048])

		# (4. Prepare image) This pipeline use internal conditional images from Pytorch3D
		mesh_processor.uvp.to(self._execution_device)
		conditioning_images, masks = mesh_processor.get_conditioning_images(height, cond_type=cond_type)
		conditioning_images = conditioning_images.to(device, dtype=positive_prompt_embeds.dtype)
		if cond_type == "canny":
			conditioning_images = mesh_processor.get_canny_images(self.prepare_image, conditioning_images, width, height, batch_size, num_images_per_prompt, do_classifier_free_guidance, guess_mode, device, controlnet.dtype)
		#mesh_processor.save_conditioning_images(conditioning_images, self.intermediate_dir, cond_type=cond_type)

		original_size = original_size or conditioning_images.shape[-2:]
		target_size = target_size or (height, width)

		# 5. Prepare timesteps
		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps
  
		positive_add_time_ids = self._get_add_time_ids(
			original_size,
			crops_coords_top_left,
			target_size,
			dtype=positive_prompt_embeds.dtype,
			text_encoder_projection_dim=text_encoder_projection_dim,
		).to(device)
		positive_add_time_ids = positive_add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

		if negative_original_size is not None and negative_target_size is not None:
			negative_add_time_ids = self._get_add_time_ids(
				negative_original_size,
				negative_crops_coords_top_left,
				negative_target_size,
				dtype=positive_prompt_embeds.dtype,
				text_encoder_projection_dim=text_encoder_projection_dim,
			)
		else:
			negative_add_time_ids = positive_add_time_ids

		'''
			Use groups to manage prompt and results
			Make sure negative and positive prompt does not perform attention together
		'''
		prompt_embeds_groups = {"positive": [positive_prompt_embeds, positive_image_embeds, pooled_positive_prompt_embeds, positive_add_time_ids]}
		if do_classifier_free_guidance:
			prompt_embeds_groups["negative"] = [negative_prompt_embeds, negative_image_embeds, pooled_negative_prompt_embeds, negative_add_time_ids]
  
        # 5.5 Optionally get Guidance Scale Embedding
		timestep_cond = None
		if self.unet.config.time_cond_proj_dim is not None:
			guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
			timestep_cond = self.get_guidance_scale_embedding(
			    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
			).to(device=device, dtype=latents.dtype)

		# 6. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents(
			batch_size,
			num_channels_latents,
			height,
			width,
			positive_prompt_embeds.dtype,
			device,
			generator,
			None,
		)

		latent_tex = mesh_processor.uvp.set_noise_texture()
		noise_views = mesh_processor.uvp.render_textured_views()
		foregrounds = [view[:-1] for view in noise_views]
		masks = [view[-1:] for view in noise_views]
		composited_tensor = composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
		latents = composited_tensor.type(latents.dtype)
		mesh_processor.uvp.to("cpu")


		# 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 7.1 Create tensor stating which controlnets to keep
		controlnet_keep = []

		for i in range(len(timesteps)):
			keeps = [
				1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
				for s, e in zip(control_guidance_start, control_guidance_end)
			]
			controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

		# 8. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		intermediate_results = []
		background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
		dbres_sizes_list = []
		mbres_size_list = []
		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):

				# expand the latents if we are doing classifier free guidance
				latent_model_input = self.scheduler.scale_model_input(latents, t)
				
				result_groups = {}
				for prompt_tag, prompt_embeds_list in prompt_embeds_groups.items():
					prompt_embeds, image_embeds, add_text_embeds, add_time_ids = prompt_embeds_list
					if prompt_tag == "positive" or not guess_mode:
						# controlnet(s) inference
						control_model_input = latent_model_input
						controlnet_prompt_embeds = prompt_embeds
						control_add_text_embeds = add_text_embeds
						control_add_time_ids = add_time_ids

						if isinstance(controlnet_keep[i], list):
							cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
						else:
							controlnet_cond_scale = controlnet_conditioning_scale
							if isinstance(controlnet_cond_scale, list):
								controlnet_cond_scale = controlnet_cond_scale[0]
							cond_scale = controlnet_cond_scale * controlnet_keep[i]

						# Split into micro-batches according to group meta info
						# Ignore this feature for now
						down_block_res_samples_list = []
						mid_block_res_sample_list = []

						model_input_batches = [torch.index_select(control_model_input, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						prompt_embeds_batches = [torch.index_select(controlnet_prompt_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						conditioning_images_batches = [torch.index_select(conditioning_images, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						text_embeds_batches = [torch.index_select(control_add_text_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
						time_ids_batches = [torch.index_select(control_add_time_ids, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]

						for model_input_batch ,prompt_embeds_batch, conditioning_images_batch, text_embeds_batch, time_ids_batch \
							in zip (model_input_batches, prompt_embeds_batches, conditioning_images_batches, text_embeds_batches, time_ids_batches):
							controlnet_added_cond_kwargs = {"text_embeds": text_embeds_batch, "time_ids": time_ids_batch}
       
							down_block_res_samples, mid_block_res_sample = self.controlnet(
								model_input_batch,
								t,
								encoder_hidden_states=prompt_embeds_batch,
								controlnet_cond=conditioning_images_batch,
								conditioning_scale=cond_scale,
								guess_mode=guess_mode,
								added_cond_kwargs=controlnet_added_cond_kwargs,
								return_dict=False,
							)
							down_block_res_samples_list.append(down_block_res_samples)
							mid_block_res_sample_list.append(mid_block_res_sample)

						''' For the ith element of down_block_res_samples, concat the ith element of all mini-batch result '''
						model_input_batches = prompt_embeds_batches = conditioning_images_batches = text_embeds_batches = time_ids_batches = None

						if guess_mode:
							for dbres in down_block_res_samples_list:
								dbres_sizes = []
								for res in dbres:
									dbres_sizes.append(res.shape)
								dbres_sizes_list.append(dbres_sizes)

							for mbres in mid_block_res_sample_list:
								mbres_size_list.append(mbres.shape)

					else:
						# Infered ControlNet only for the conditional batch.
						# To apply the output of ControlNet to both the unconditional and conditional batches,
						# add 0 to the unconditional batch to keep it unchanged.
						# We copy the tensor shapes from a conditional batch
						down_block_res_samples_list = []
						mid_block_res_sample_list = []
						for dbres_sizes in dbres_sizes_list:
							down_block_res_samples_list.append([torch.zeros(shape, device=self._execution_device, dtype=latents.dtype) for shape in dbres_sizes])
						for mbres in mbres_size_list:
							mid_block_res_sample_list.append(torch.zeros(mbres, device=self._execution_device, dtype=latents.dtype))
						dbres_sizes_list = []
						mbres_size_list = []


					'''
					
						predict the noise residual, split into mini-batches
						Downblock res samples has n samples, we split each sample into m batches
						and re group them into m lists of n mini batch samples.
					
					'''
					noise_pred_list = []
					model_input_batches = [torch.index_select(latent_model_input, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
					prompt_embeds_batches = [torch.index_select(prompt_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]					
					image_embeds_batches = [torch.index_select(image_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
					text_embeds_batches = [torch.index_select(add_text_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
					time_ids_batches = [torch.index_select(add_time_ids, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas]
      
					for model_input_batch, prompt_embeds_batch, image_embeds_batch, text_embeds_batch, time_ids_batch, down_block_res_samples_batch, mid_block_res_sample_batch, meta \
						in zip(model_input_batches, prompt_embeds_batches, image_embeds_batches, text_embeds_batches, time_ids_batches, down_block_res_samples_list, mid_block_res_sample_list, self.group_metas):
						if t > num_timesteps * (1- ref_attention_end):
							replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=1)
						else:
							replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=0)

						added_cond_kwargs = {"text_embeds": text_embeds_batch, "time_ids": time_ids_batch}
						if ip_adapter_image is not None:
							added_cond_kwargs["image_embeds"] = [image_embeds_batch]
      
						assert not torch.isnan(model_input_batch).any(), "Model input batch contains nan values"
						assert not torch.isnan(prompt_embeds_batch).any(), "Prompt embeds batch contains nan values"
						assert not torch.isnan(conditioning_images_batch).any(), "Conditioning images batch contains nan values"
						assert not torch.isnan(text_embeds_batch).any(), "Text embeds batch contains nan values"
						assert not torch.isnan(time_ids_batch).any(), "Time ids batch contains nan values"

						noise_pred = self.unet(
							model_input_batch,
							t,
							encoder_hidden_states=prompt_embeds_batch,
							timestep_cond=timestep_cond,
							cross_attention_kwargs=cross_attention_kwargs,
							down_block_additional_residuals=down_block_res_samples_batch,
							mid_block_additional_residual=mid_block_res_sample_batch,
							added_cond_kwargs=added_cond_kwargs,
							return_dict=False,
						)[0]
						noise_pred_list.append(noise_pred)

					noise_pred_list = [torch.index_select(noise_pred, dim=0, index=torch.tensor(meta[1], device=self._execution_device)) for noise_pred, meta in zip(noise_pred_list, self.group_metas)]
					noise_pred = torch.cat(noise_pred_list, dim=0)
					down_block_res_samples_list = None
					mid_block_res_sample_list = None
					noise_pred_list = None
					model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = mid_block_res_sample_batches = None

					result_groups[prompt_tag] = noise_pred

				positive_noise_pred = result_groups["positive"]

				# perform guidance
				if do_classifier_free_guidance:
					noise_pred = result_groups["negative"] + guidance_scale * (positive_noise_pred - result_groups["negative"])


				if do_classifier_free_guidance and guidance_rescale > 0.0:
					# Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
					noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
     
				assert not torch.isnan(noise_pred).any(), "Final Noise prediction contains nan values"

				mesh_processor.uvp.to(self._execution_device)
				# compute the previous noisy sample x_t -> x_t-1
				# Multi-View step or individual step
				current_exp = ((exp_end-exp_start) * i / num_inference_steps) + exp_start
				if t > (1-multiview_diffusion_end)*num_timesteps:
					step_results = step_tex(
						scheduler=self.scheduler, 
						uvp=mesh_processor.uvp, 
						model_output=noise_pred, 
						timestep=t, 
						sample=latents, 
						texture=latent_tex,
						return_dict=True, 
						main_views=[], 
						exp= current_exp,
						**extra_step_kwargs
					)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = step_results["prev_tex"]

					# Composit latent foreground with random color background
					background_latents = [self.color_latents[color] for color in background_colors]
					composited_tensor = composite_rendered_view(self.scheduler, background_latents, latents, masks, t)
					assert not torch.isnan(composited_tensor).any(), "composited_tensor contains nan values"
					latents = composited_tensor.type(latents.dtype)

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
				else:
					step_results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = None

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
     
				assert not torch.isnan(pred_original_sample).any(), "pred_original_sample contains nan values"
				assert not torch.isnan(latents).any(), "latents contains nan values"
				#assert not torch.isnan(latent_tex).any(), "latent_tex contains nan values"

				del noise_pred, result_groups
					


				# Update pipeline settings after one step:
				# 1. Annealing ControlNet scale
				if (1-t/num_timesteps) < control_guidance_start[0]:
					controlnet_conditioning_scale = initial_controlnet_conditioning_scale
				elif (1-t/num_timesteps) > control_guidance_end[0]:
					controlnet_conditioning_scale = controlnet_conditioning_end_scale
				else:
					alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (control_guidance_end[0] - control_guidance_start[0])
					controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + (1-alpha) * controlnet_conditioning_end_scale

				# 2. Shuffle background colors; only black and white used after certain timestep
				if (1-t/num_timesteps) < shuffle_background_change:
					background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses))]
				elif (1-t/num_timesteps) < shuffle_background_end:
					background_colors = [random.choice(["black","white"]) for i in range(len(self.camera_poses))]
				else:
					background_colors = background_colors



				# Logging at "log_interval" intervals and last step
				# Choose to uses color approximation or vae decoding
				if i % log_interval == log_interval-1 or t == 1:
					if view_fast_preview:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = latent_preview(latent_images.to(self._execution_device))
							images = np.concatenate([img for img in images], axis=1)
							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")
					else:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = decode_latents(self.vae, latent_images.to(self._execution_device))

							images = np.concatenate([img for img in images], axis=1)

							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")

					if not t < (1-multiview_diffusion_end)*num_timesteps:
						if tex_fast_preview:
							tex = latent_tex.clone()
							texture_color = latent_preview(tex[None, ...])
							numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.jpg")
						else:
							mesh_processor.uvp_rgb.to(self._execution_device)
							result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, mesh_processor.uvp_rgb, pred_original_sample)
							numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
							mesh_processor.uvp_rgb.to("cpu")

				mesh_processor.uvp.to("cpu")

				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						callback(i, t, latents)

				# Signal the program to skip or end
				#import select
				#import sys
				#if select.select([sys.stdin],[],[],0)[0]:
				#	userInput = sys.stdin.readline().strip()
				#	if userInput == "skip":
				#		return None
				#	elif userInput == "end":
				#		exit(0)

		
		mesh_processor.uvp.to(self._execution_device)
		mesh_processor.uvp_rgb.to(self._execution_device)
		result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, mesh_processor.uvp_rgb, latents)
		mesh_processor.uvp.save_mesh(f"{self.result_dir}/textured.glb", result_tex_rgb.permute(1,2,0))


		mesh_processor.uvp_rgb.set_texture_map(result_tex_rgb)
		textured_views = mesh_processor.uvp_rgb.render_textured_views()
		textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
		textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
		v = numpy_to_pil(textured_views_rgb)[0]
		v.save(f"{self.result_dir}/textured_views_rgb.jpg")
		# display(v)

		# Offload last model to CPU
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.final_offload_hook.offload()

		mesh_processor.uvp.to("cpu")
		mesh_processor.uvp_rgb.to("cpu")

		if ip_adapter_image is not None:
			self.unload_ip_adapter()

		return result_tex_rgb, textured_views, v