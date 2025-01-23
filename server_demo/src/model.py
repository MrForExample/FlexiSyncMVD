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


def load_pipeline(
    opt: Optional[PipelineConfig] = PipelineConfig(),
    vae: Optional[AutoencoderKL] = None,
    controlnet: Optional[ControlNetModel] = None,
    unet: Optional[UNet2DConditionModel] = None,
) -> StableSyncMVDPipeline | StableSyncMVDPipelineXL:
    syncmvd_instance: StableSyncMVDPipeline = None

    if opt.t2i_model == "SD1.5":
        if opt.cond_type == "normal":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_normalbae",
                variant="fp16",
                torch_dtype=torch.float16,
            )
        elif opt.cond_type == "depth":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                variant="fp16",
                torch_dtype=torch.float16,
            )
        elif opt.cond_type == "canny":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                variant="fp16",
                torch_dtype=torch.float16,
            )
        else:
            ValueError(f"Condition {opt.cond_type} is not supported")

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

    elif opt.t2i_model == "SDXL":
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
        ValueError(f"Model {opt.t2i_model} is not supported")

    return syncmvd_instance


def initialize_pipeline(
    syncmvd_instance: StableSyncMVDPipeline,
    opt: Optional[PipelineConfig] = PipelineConfig(),
    logging_config: dict = {},
) -> StableSyncMVDPipeline:
    syncmvd_instance.initialize_pipeline(
        camera_azims=opt.camera_azims,
        top_cameras=not opt.no_top_cameras,
        ref_views=[],
        latent_size=opt.latent_view_size,
        max_batch_size=opt.max_batch_size,
        logging_config=logging_config,
    )

    return syncmvd_instance


def initialize_mesh_processor(
    opt: Optional[PipelineConfig] = PipelineConfig(),
    mesh_path: str = "",
    camera_poses: list = [],
    device: Any = torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu"),
) -> MeshProcessor:
    return MeshProcessor(
        mesh_path=mesh_path,  # This is the input file
        mesh_transform={"scale": opt.mesh_scale},
        mesh_autouv=not opt.keep_mesh_uv,
        latent_size=opt.latent_view_size,
        render_rgb_size=opt.rgb_view_size,
        latent_texture_size=opt.latent_tex_size,
        texture_rgb_size=opt.rgb_tex_size,
        camera_poses=camera_poses,
        device=device,
    )


def load_reference_image(opt: Optional[PipelineConfig] = PipelineConfig()):
    return load_image(opt.ip_adapter_image) if opt.ip_adapter_image else None


def run_pipeline(
    syncmvd_instance: StableSyncMVDPipeline,
    mesh_processor: MeshProcessor,
    opt: Optional[PipelineConfig] = PipelineConfig(),
    logging_config: dict = {},
    ip_adapter_image: Optional[str] = None,
):
    result_tex_rgb, textured_views, v = syncmvd_instance(
        prompt=opt.prompt,
        height=opt.latent_view_size * 8,
        width=opt.latent_view_size * 8,
        num_inference_steps=opt.steps,
        guidance_scale=opt.guidance_scale,
        negative_prompt=opt.negative_prompt,
        ip_adapter_image=ip_adapter_image,
        ip_adapter_scale=opt.ip_adapter_scale,
        generator=torch.manual_seed(opt.seed),
        controlnet_guess_mode=opt.guess_mode,
        controlnet_conditioning_scale=opt.conditioning_scale,
        controlnet_conditioning_end_scale=opt.conditioning_scale_end,
        control_guidance_start=opt.control_guidance_start,
        control_guidance_end=opt.control_guidance_end,
        guidance_rescale=opt.guidance_rescale,
        mesh_processor=mesh_processor,
        multiview_diffusion_end=opt.mvd_end,
        exp_start=opt.mvd_exp_start,
        exp_end=opt.mvd_exp_end,
        ref_attention_end=opt.ref_attention_end,
        shuffle_background_change=opt.shuffle_bg_change,
        shuffle_background_end=opt.shuffle_bg_end,
        logging_config=logging_config,
        cond_type=opt.cond_type,
    )


def load_config(
    opt: Optional[PipelineConfig] = PipelineConfig(),
) -> tuple[LoggingConfig, str]:
    mesh_path = abspath(opt.mesh)

    if opt.output:
        output_root = abspath(opt.output)
    else:
        # Save to tmp directory
        temp_dir = tempfile.NamedTemporaryFile().name
        output_root = abspath(temp_dir)

    output_name_components = []
    if opt.prefix and opt.prefix != "":
        output_name_components.append(opt.prefix)
    if opt.use_mesh_name:
        mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
        output_name_components.append(mesh_name)

    if opt.timeformat and opt.timeformat != "":
        output_name_components.append(datetime.now().strftime(opt.timeformat))

    output_name = "_".join(output_name_components)
    output_dir = join(output_root, "_exp", output_name)

    if not isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError(f"Output directory {output_dir} already exists")

    logging_config: LoggingConfig = LoggingConfig(
        output_dir=output_dir,
        log_interval=opt.log_interval,
        view_fast_preview=opt.view_fast_preview,
        tex_fast_preview=opt.tex_fast_preview,
    )

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
