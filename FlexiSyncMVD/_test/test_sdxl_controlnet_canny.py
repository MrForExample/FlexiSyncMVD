from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import torch
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

def HWC3(x):
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

timeformat = '%d%b%Y-%H%M%S'

#prompt = "A pink metal flying car with black window on each side of the car door, futuristic cyberpunk style, pure grey background, detailed, 8k"
#prompt = "A terrifying lizard-like monster with lava flowing on its back, back view, detailed, 8k"
prompt = "Cammy from Street Fighter, she wears a tight green V shape bodysuit, bare skin legs from feet to thighs with no pants to cover it. A red beret with a black star tops her head, and she has long blonde braids. Her red combat gauntlets and piercing blue eyes emphasize her readiness for battle, detailed, 8k"
negative_prompt = "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."
generator = torch.manual_seed(1234)
latent_view_size = 128

# Create conditional image
controlnet_img = cv2.imread(r"_test/cammy_front_canny.png")
#controlnet_img = cv2.imread(r"_test/flying_car.jpg")
height, width, _  = controlnet_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
#
#controlnet_img = cv2.Canny(controlnet_img, 100, 200)
#controlnet_img = HWC3(controlnet_img)
controlnet_img = Image.fromarray(controlnet_img)
#controlnet_img = load_image(r"_test/cammy_front_canny.png")


#controlnet = ControlNetModel.from_pretrained("xinsir/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    vae=vae, # negligible difference
    controlnet=controlnet, 
    torch_dtype=torch.float16
).to("cuda")

image = pipeline(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    image=controlnet_img,
	height=latent_view_size*8,
	width=latent_view_size*8,
    num_inference_steps=50, 
    generator=generator,
    controlnet_conditioning_scale=0.5,
    guidance_scale=5.0
).images[0]

#final_image = make_image_grid([image, controlnet_img], 1, 2)
final_image = image
final_image.save(f"_test/_outputs/sdxl_test_{datetime.now().strftime(timeformat)}_canny.png")



