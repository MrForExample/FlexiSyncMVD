from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers import (
	DDPMScheduler,
	EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image
import torch
from datetime import datetime

timeformat = '%d%b%Y-%H%M%S'

#prompt = "A pink metal flying car with black window on each side of the car door, futuristic cyberpunk style, pure grey background, detailed, 8k"
#prompt = "A terrifying lizard-like monster with lava flowing on its back, back view, detailed, 8k"
#prompt = "Cammy white from Street Fighter, smooth legs, smooth thighs, no pants, no socks, short red ankle boots, green V shape upper bodysuit, red beret with a black star, blonde braids, black combat gauntlets, blue eyes, grey background, detailed, 8k"
prompt = "1girl, Cammy white, Street Fighter, smooth legs, smooth thighs, no pants, no socks, short red ankle boots, green V shape upper bodysuit, red beret with a black star, blonde braids, black combat gauntlets, blue eyes, grey background, masterpiece, best quality, very aesthetic, absurdres"
#negative_prompt = "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."
#negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
generator = torch.manual_seed(8179493792923949)

#cond_image = load_image(r"C:\Users\reall\Softwares\Miniconda\envs\SyncTweedies_Test\_Project\SyncMVD\data\flying_car\MVD_23Oct2024-182042\intermediate\cond_5.jpg")
#cond_image = load_image(r"_test\monster_back_depth.png")
#cond_image = load_image(r"_test\car_depth_with_black_bg.png")
cond_image = load_image(r"./_test/Cammy_gen_1_renders/cond_4.png")

#vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
	#vae=vae,
	torch_dtype=torch.float16, 
	variant="fp16", 
	safety_checker=None,
).to("cuda")

pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

image = pipeline(
    prompt=prompt, 
    negative_prompt=negative_prompt, 
    image=cond_image,
    num_inference_steps=50, 
    generator=generator,
	strength=0.2,
    guidance_scale=3.5,
).images[0]

image.save(f"./_test/_outputs/sdxl_tile_test_{datetime.now().strftime(timeformat)}.png")



