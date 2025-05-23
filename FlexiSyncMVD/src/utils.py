from PIL import Image
import numpy as np
import math
import random
import torch
from torchvision.transforms import Resize, InterpolationMode


'''
	Encoding and decoding functions similar to diffusers library implementation
'''
@torch.no_grad()
def encode_latents(vae, imgs):
	"""
	Encode a batch of images into their respective latent codes
	Arguments:
		vae: The VAE model to use for encoding
		imgs: The batch of images to be encoded
	Returns:
		latents: The latent codes of the input images
	"""
	assert not torch.isnan(imgs).any(), "Input images contain nan values"
	imgs = (imgs-0.5)*2
	latents = vae.encode(imgs).latent_dist.sample()
	assert not torch.isnan(latents).any(), "Encoded latents contain nan values"
	latents = vae.config.scaling_factor * latents
	# Ensure latents do not contain nan values after scaling
	assert not torch.isnan(latents).any(), "Scaled latents contain nan values"
	return latents


@torch.no_grad()
def decode_latents(vae, latents):

	"""
	Decode a batch of latent codes into their respective images
	Arguments:
		vae: The VAE model to use for decoding
		latents: The batch of latent codes to be decoded
	Returns:
		image: The decoded image batch
	"""
	latents = 1 / vae.config.scaling_factor * latents

	image = vae.decode(latents, return_dict=False)[0]
	torch.cuda.current_stream().synchronize()

	image = (image / 2 + 0.5).clamp(0, 1)
	# we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
	image = image.permute(0, 2, 3, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	
	return image


# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
	# adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
	"""
	Perform a fast, low-quality preview of a latent code by projecting it to RGB using a fixed linear transformation.
	
	Arguments:
		x: The latent code to be previewed, of shape (N, H, W, C)
	
	Returns:
		image: The preview image, of shape (N, H, W, 3)
	
	Notes:
		This function is much faster than the standard decoding process, but produces a lower-quality image.
		The linear transformation used is based on the v1.4 model, and may not be suitable for all models.
	"""
	v1_4_latent_rgb_factors = torch.tensor([
		#   R        G        B
		[0.298, 0.207, 0.208],  # L1
		[0.187, 0.286, 0.173],  # L2
		[-0.158, 0.189, 0.264],  # L3
		[-0.184, -0.271, -0.473],  # L4
	], dtype=x.dtype, device=x.device)
	image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents):
	"""
	Decode a batch of latent codes into their respective images, and bake them into a single RGB texture using the UVProjector.
	
	Arguments:
		vae: The VAE model to use for decoding
		uvp_rgb: The UVProjector object to use for baking the texture
		latents: The batch of latent codes to be decoded
	
	Returns:
		result_tex_rgb: The final RGB texture, of shape (H, W, 3)
		result_tex_rgb_output: The final RGB texture, of shape (1, H, W, 3)
	
	Notes:
		This function is used to generate a single RGB texture from a batch of latent codes.
		The resulting texture is a linear interpolation of the decoded views.
	"""
	result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
	resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
	result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
	textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
	result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
	return result_tex_rgb, result_tex_rgb_output


