from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
from datetime import datetime

timeformat = '%d%b%Y-%H%M%S'

prompt = "A realistic pink metal cyberpunk flying car, detailed, 8k"
#prompt = "A terrifying lizard-like monster with lava flowing on its back, back view, detailed, 8k"
negative_prompt = "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."
generator = torch.manual_seed(0)
latent_view_size = 96
cond_image = load_image(r"C:\Users\reall\Softwares\Miniconda\envs\SyncTweedies_Test\_Project\SyncMVD\data\flying_car\MVD_23Oct2024-182042\intermediate\cond_5.jpg")
#cond_image = load_image(r"_test\monster_back_depth.png")
#cond_image = load_image(r"_test\car_depth_with_black_bg.png")

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
).to("cuda")

image = pipeline(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    image=cond_image,
	height=latent_view_size*8,
	width=latent_view_size*8,
    num_inference_steps=30, 
    generator=generator, 
    guidance_scale=7.5,
).images[0]

image.save(f"_test\_outputs\sd15_test_{datetime.now().strftime(timeformat)}.png")