from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime

timeformat = '%d%b%Y-%H%M%S'

prompt = "A realistic pink metal cyberpunk flying car."
negative_prompt = "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."
generator = torch.manual_seed(0)
latent_view_size = 96

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

image = pipeline(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
	height=latent_view_size*8,
	width=latent_view_size*8,
    num_inference_steps=30, 
    generator=generator, 
    guidance_scale=15.5
).images[0]

image.save(f"sdxl_test_{datetime.now().strftime(timeformat)}.png")



