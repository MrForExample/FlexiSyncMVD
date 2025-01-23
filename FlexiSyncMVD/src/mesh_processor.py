import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from diffusers.utils import numpy_to_pil
from .renderer.project import UVProjection as UVP
from timeit import default_timer as timer

class MeshProcessor:
	def __init__(
			self,
			mesh_path,
			mesh_transform,
			mesh_autouv,
			latent_size,
			render_rgb_size,
			latent_texture_size,
			texture_rgb_size,
			camera_poses,
			camera_centers=None,
			camera_distance=4.0,
			device=None,
		):
		# Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
		# when mesh_autouv is set to True, will use Xatlas to unwrap UV, but takes much longer time
		start = timer()

		self.latent_size = latent_size
		self.uvp = UVP(texture_size=latent_texture_size, render_size=latent_size, sampling_mode="nearest", channels=4, device=device)
		self.uvp.load_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		self.uvp.set_cameras_and_render_settings(camera_poses, centers=camera_centers, camera_distance=camera_distance)
		
		end = timer()
		print(f"Mesh loaded in {end - start} seconds")


		start = timer()

		self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest", channels=3, device=device)
		self.uvp_rgb.mesh = self.uvp.mesh.clone()
		self.uvp_rgb.target_size = self.uvp.target_size
		self.uvp_rgb.w2h_ratio = self.uvp.w2h_ratio
		self.uvp_rgb.set_cameras_and_render_settings(camera_poses, centers=camera_centers, camera_distance=camera_distance)
		_,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
		self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

		# Save some VRAM
		del _, cos_maps
		#self.uvp.to("cpu")
		#self.uvp_rgb.to("cpu")
		end = timer()
		print(f"UV Projection setup done in {end - start} seconds")

	def HWC3(self, x):
		assert x.dtype == np.uint8
		if x.ndim == 2:
			x = x[:, :, None]
		assert x.ndim == 3
		H, W, C = x.shape
		assert C == 1 or C == 3 or C == 4
		if C == 3:
			return x
		if C == 1:
			return np.concatenate([x, x, x], axis=2)
		if C == 4:
			color = x[:, :, 0:3].astype(np.float32)
			alpha = x[:, :, 3:4].astype(np.float32) / 255.0
			y = color * alpha + 255.0 * (1.0 - alpha)
			y = y.clip(0, 255).astype(np.uint8)
			return y

	# Used to generate depth or normal conditioning images
	@torch.no_grad()
	def get_conditioning_images(self, output_size, render_size=512, blur_filter=5, cond_type="depth", render_cam_index=None):
		start = timer()

		cond_transforms = Compose([
			Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True), 
			GaussianBlur(blur_filter, blur_filter//3+1)]
		)

		if cond_type == "render":
			view_renders = self.uvp_rgb.render_textured_views(return_tensor=True)
			masks = view_renders[:,3,...][:,None,...]
			latent_masks = Resize((self.latent_size,)*2, antialias=True)(masks)

			view_renders = view_renders[:,:3,...]
			render_masks = masks > 0.5
			render_masks = render_masks.expand_as(view_renders)

			view_renders[~render_masks] = 1 # Set pure color for background
			conditioning_images = cond_transforms(view_renders)
		else:
			verts, normals, depths, cos_maps, texels, fragments = self.uvp.render_geometry(image_size=render_size, render_cam_index=render_cam_index)
			masks = normals[...,3][:,None,...]
			latent_masks = Resize((self.latent_size,)*2, antialias=True)(masks)

			if cond_type == "normal" or cond_type == "canny":
				view_normals = self.uvp.decode_view_normal(normals) *2 - 1
				conditioning_images = cond_transforms(view_normals)
			# Some problem here, depth controlnet don't work when depth is normalized
			# But it do generate using the unnormalized form as below
			elif cond_type == "depth":
				view_depths = self.uvp.decode_normalized_depth(depths)
				view_depths * 0.75 + 0.25 # Normalize depth to [0.25, 1]

				depths_mask = masks > 0.5
				depths_mask = depths_mask.expand_as(view_depths)
				
				view_depths[~depths_mask] = 0 # Set depth to 0 for background

				conditioning_images = cond_transforms(view_depths)
			
		end = timer()
		print(f"Rendering conditioning images done in {end - start} seconds")
		
		return conditioning_images, latent_masks
	
	def get_canny_images(self, prepare_image, conditioning_images, width, height, batch_size, num_images_per_prompt, do_classifier_free_guidance, guess_mode, device, dtype):
		cond = (conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy()

		controlnet_img = (cond*255).astype(np.uint8)
		canny_img_list = []
		for img in controlnet_img:
			canny_img = cv2.Canny(img, 100, 200)
			canny_img = self.HWC3(canny_img)
			canny_img = Image.fromarray(canny_img)
			canny_img_list.append(canny_img)

		conditioning_images = prepare_image(
			image=canny_img_list,
			width=width,
			height=height,
			batch_size=batch_size * num_images_per_prompt,
			num_images_per_prompt=num_images_per_prompt,
			device=device,
			dtype=dtype,
			do_classifier_free_guidance=do_classifier_free_guidance,
			guess_mode=guess_mode,
		)
		
		return conditioning_images

	def save_conditioning_images(self, conditioning_images, output_dir, save_each_cond_img=False, cond_type="depth"):
		if cond_type == "normal":
			conditioning_images = conditioning_images/2+0.5
		cond = conditioning_images.permute(0,2,3,1).cpu().numpy()

		list_pil = None
		if save_each_cond_img:
			list_pil = numpy_to_pil(cond)
			for i, img in enumerate(list_pil):
				img.save(f"{output_dir}/cond_{i}.jpg")

		cond = np.concatenate([img for img in cond], axis=1)
		cond = numpy_to_pil(cond)[0]
		cond.save(f"{output_dir}/cond.jpg")

		return cond