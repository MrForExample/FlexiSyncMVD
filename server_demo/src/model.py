from datetime import datetime
from genericpath import isdir
import os
from posixpath import abspath, basename, join, splitext
import tempfile
import torch
from typing import Any, Optional
from transformers import CLIPVisionModelWithProjection
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import load_image

from FlexiSyncMVD.src.pipeline import StableSyncMVDPipeline
from FlexiSyncMVD.src.pipeline_XL import StableSyncMVDPipelineXL
from server_demo.src.custom_types import (
    InputConfig,
    LoggingConfig,
    PipelineConfig,
    merge_configs,
)
from FlexiSyncMVD.src.mesh_processor import MeshProcessor
from app_config import AppConfig


def load_pipeline(opt=None, vae=None, controlnet=None, unet=None):
    config = AppConfig.load_config(pipeline_overrides=vars(opt) if opt else None)
    pipeline_config = config.pipeline
    syncmvd_instance = None

    if pipeline_config.t2i_model == "SD1.5":
        if pipeline_config.cond_type == "normal":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_normalbae",
                variant="fp16",
                torch_dtype=torch.float16,
            )
        elif pipeline_config.cond_type == "depth":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                variant="fp16",
                torch_dtype=torch.float16,
            )
        elif pipeline_config.cond_type == "canny":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                variant="fp16",
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Condition {pipeline_config.cond_type} is not supported")

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )

        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        syncmvd_instance = StableSyncMVDPipeline(**pipe.components)

    elif pipeline_config.t2i_model == "SDXL":
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )

        if not unet:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                image_encoder=image_encoder,
                vae=vae,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
        else:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                image_encoder=image_encoder,
                vae=vae,
                unet=unet,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )

        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        syncmvd_instance = StableSyncMVDPipelineXL(**pipe.components)

    else:
        raise ValueError(f"Model {pipeline_config.t2i_model} is not supported")

    return syncmvd_instance


def initialize_pipeline(syncmvd_instance, opt=None, logging_config={}):
    config = AppConfig.load_config(pipeline_overrides=vars(opt) if opt else None)
    pipeline_config = config.pipeline
    syncmvd_instance.initialize_pipeline(
        camera_azims=pipeline_config.camera_azims,
        top_cameras=not pipeline_config.no_top_cameras,
        ref_views=[],
        latent_size=pipeline_config.latent_view_size,
        max_batch_size=pipeline_config.max_batch_size,
        logging_config=logging_config,
    )
    return syncmvd_instance


def initialize_mesh_processor(opt=None, mesh_path="", camera_poses=[], device=None):
    config = AppConfig.load_config(pipeline_overrides=vars(opt) if opt else None)
    pipeline_config = config.pipeline
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return MeshProcessor(
        mesh_path=mesh_path,
        mesh_transform={"scale": pipeline_config.mesh_scale},
        mesh_autouv=not pipeline_config.keep_mesh_uv,
        latent_size=pipeline_config.latent_view_size,
        render_rgb_size=pipeline_config.rgb_view_size,
        latent_texture_size=pipeline_config.latent_tex_size,
        texture_rgb_size=pipeline_config.rgb_tex_size,
        camera_poses=camera_poses,
        device=device,
    )


def load_reference_image(opt=None):
    config = AppConfig.load_config(pipeline_overrides=vars(opt) if opt else None)
    pipeline_config = config.pipeline
    return (
        load_image(pipeline_config.ip_adapter_image)
        if pipeline_config.ip_adapter_image
        else None
    )


def run_pipeline(
    syncmvd_instance, mesh_processor, opt=None, logging_config={}, ip_adapter_image=None
):
    config = AppConfig.load_config(pipeline_overrides=vars(opt) if opt else None)
    pipeline_config = config.pipeline
    result_tex_rgb, textured_views, v = syncmvd_instance(
        prompt=pipeline_config.prompt,
        height=pipeline_config.latent_view_size * 8,
        width=pipeline_config.latent_view_size * 8,
        num_inference_steps=pipeline_config.steps,
        guidance_scale=pipeline_config.guidance_scale,
        negative_prompt=pipeline_config.negative_prompt,
        ip_adapter_image=ip_adapter_image,
        ip_adapter_scale=pipeline_config.ip_adapter_scale,
        generator=torch.manual_seed(pipeline_config.seed),
        controlnet_guess_mode=pipeline_config.guess_mode,
        controlnet_conditioning_scale=pipeline_config.conditioning_scale,
        controlnet_conditioning_end_scale=pipeline_config.conditioning_scale_end,
        control_guidance_start=pipeline_config.control_guidance_start,
        control_guidance_end=pipeline_config.control_guidance_end,
        guidance_rescale=pipeline_config.guidance_rescale,
        mesh_processor=mesh_processor,
        multiview_diffusion_end=pipeline_config.mvd_end,
        exp_start=pipeline_config.mvd_exp_start,
        exp_end=pipeline_config.mvd_exp_end,
        ref_attention_end=pipeline_config.ref_attention_end,
        shuffle_background_change=pipeline_config.shuffle_bg_change,
        shuffle_background_end=pipeline_config.shuffle_bg_end,
        logging_config=logging_config,
        cond_type=pipeline_config.cond_type,
    )


def load_config(opt=None):
    config = AppConfig.load_config(pipeline_overrides=vars(opt) if opt else None)
    pipeline_config = config.pipeline
    mesh_path = abspath(pipeline_config.mesh)
    output_root = (
        abspath(pipeline_config.output)
        if pipeline_config.output
        else tempfile.NamedTemporaryFile().name
    )
    output_name_components = []
    if pipeline_config.prefix:
        output_name_components.append(pipeline_config.prefix)
    if pipeline_config.use_mesh_name:
        mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
        output_name_components.append(mesh_name)
    if pipeline_config.timeformat:
        output_name_components.append(
            datetime.now().strftime(pipeline_config.timeformat)
        )
    output_name = "_".join(output_name_components)
    output_dir = join(output_root, "_exp", output_name)

    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError(f"Output directory {output_dir} already exists")

    logging_config = {
        "output_dir": output_dir,
        "log_interval": pipeline_config.log_interval,
        "view_fast_preview": pipeline_config.view_fast_preview,
        "tex_fast_preview": pipeline_config.tex_fast_preview,
    }

    return logging_config, mesh_path


def preload_pipeline(
    t2i_model: str = "SDXL",
    vae: Optional[AutoencoderKL] = None,
    controlnet: Optional[ControlNetModel] = None,
    unet: Optional[UNet2DConditionModel] = None,
) -> StableSyncMVDPipeline | StableSyncMVDPipelineXL:
    opt = PipelineConfig()
    opt.t2i_model = t2i_model
    syncmvd_instance = load_pipeline(opt, vae, controlnet, unet)
    return syncmvd_instance


def preload_generic_vae() -> AutoencoderKL:
    # @TODO: Move the model name to a constant inside the settings
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    return vae


def preload_custom_checkpoint(
    checkpoint_path: str,
) -> tuple[AutoencoderKL, UNet2DConditionModel]:
    vae = AutoencoderKL.from_single_file(checkpoint_path, torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_single_file(
        checkpoint_path, torch_dtype=torch.float16
    )
    return vae, unet


def preload_controlnet(model_name: str) -> ControlNetModel:
    controlnet = ControlNetModel.from_pretrained(model_name, torch_dtype=torch.float16)
    return controlnet


def run_experiment(
    syncmvd_instance: StableSyncMVDPipeline | StableSyncMVDPipelineXL,
    input: InputConfig,
):
    opt = PipelineConfig()

    # Merge input config with default config
    opt = merge_configs(opt, input)

    logging_config, mesh_path = load_config(opt)

    initialize_pipeline(syncmvd_instance, opt, logging_config.model_dump())

    mesh_processor = initialize_mesh_processor(
        opt,
        mesh_path,
        syncmvd_instance.camera_poses,
        syncmvd_instance._execution_device,
    )

    ip_adapter_image = load_reference_image(opt)

    run_pipeline(
        syncmvd_instance=syncmvd_instance,
        mesh_processor=mesh_processor,
        opt=opt,
        logging_config=logging_config.model_dump(),
        ip_adapter_image=ip_adapter_image,
    )

    return logging_config.output_dir
