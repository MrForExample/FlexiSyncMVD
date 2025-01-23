from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from PIL import Image
import torch
import random
import numpy as np
import cv2
#from controlnet_aux import MidasDetector, ZoeDetector
from datetime import datetime

time_str = datetime.now().strftime('%d%b%Y-%H%M%S')


#processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
#processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators")


controlnet_conditioning_scale = 1.0  
prompt = "A realistic pink metal cyberpunk flying car, detailed, 8k"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
generator = torch.manual_seed(0)

#eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")


controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16
)

# when test with other base model, you need to change the vae also.
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    #scheduler=eulera_scheduler,
)

# need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance

#img = cv2.imread(r"_test\flying_car.jpg")
#
#if random.random() > 0.5:
#    controlnet_img = processor_zoe(img, output_type='cv2')
#else:
#    controlnet_img = processor_midas(img, output_type='cv2')
#
#
#height, width, _  = controlnet_img.shape
#ratio = np.sqrt(1024. * 1024. / (width * height))
#new_width, new_height = int(width * ratio), int(height * ratio)
#controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
#controlnet_img = Image.fromarray(controlnet_img)
#
#controlnet_img.save(f"_test\_outputs\sdxl_test_{time_str}_depth.png")

controlnet_img = load_image(r"_test\_outputs\sdxl_test_23Oct2024-171532_depth.png")

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=controlnet_img,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=1024,
    height=1024,
    num_inference_steps=30,
    guidance_scale=5.0,
    generator=generator, 
    ).images[0]

image.save(f"_test\_outputs\sdxl_test_{time_str}.png")
