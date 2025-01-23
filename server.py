from contextlib import asynccontextmanager
import os
import sys
from tempfile import NamedTemporaryFile
import time
from types import TracebackType
from typing import Annotated, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import HttpUrl
import torch
import uvicorn
import gradio as gr

from FlexiSyncMVD.src.pipeline import StableSyncMVDPipeline
from FlexiSyncMVD.src.pipeline_XL import StableSyncMVDPipelineXL
from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
)
from demo import build_gradio_ui
from sortium.src.logger import configure_logger, logger
from sortium.src.custom_types import (
    CPUInfo,
    GPUInfo,
    InputConfig,
    LibraryInfo,
    OSInfo,
    OutputStatus,
    PythonInfo,
    ReplayConfig,
    SystemDetails,
    TextureOutput,
)
from sortium.src.model import (
    preload_controlnet,
    preload_custom_checkpoint,
    preload_generic_vae,
    preload_pipeline,
    run_experiment,
)
from sortium.src.settings import Settings
from sortium.src.utils import (
    create_s3_client,
    download_file,
    download_image_to_tmp,
    format_memory,
    generate_random_name,
    is_valid_output_dir,
    upload_file,
)
import sentry_sdk
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

settings = Settings()

configure_logger()

# Global instances
syncmvd_instance: Optional[StableSyncMVDPipeline] = None
syncmvd_instance_xl: Optional[StableSyncMVDPipelineXL] = None
# Custom checkpoints Preload
vae_instance: AutoencoderKL = None
unet_instance: UNet2DConditionModel = None
controlnet_instance: ControlNetModel = None
# Utils
s3_client = None
# GPU Lock
gpu_lock: Lock = None
# Thread Pool
thread_pool = None
# ReplayConfig
replay_config: ReplayConfig = None


def get_system_details() -> SystemDetails:
    """
    Get system details such as OS, CPU, GPU, and Python information

    Returns:

        SystemDetails: A Pydantic model containing system details
    """
    gpu_memory = format_memory(torch.cuda.get_device_properties(0).total_memory)

    system_memory = format_memory(
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    )

    python_libs = LibraryInfo(
        pytorch_version=torch.__version__,
        fastapi_version=uvicorn.__version__,
    )

    return SystemDetails(
        os=OSInfo(name=os.uname().sysname),
        cpu=CPUInfo(
            name=os.uname().machine,
            cores=os.cpu_count(),
            ram=system_memory,
        ),
        gpu=GPUInfo(
            name=torch.cuda.get_device_name(0),
            memory=gpu_memory,
            cuda_status="available" if torch.cuda.is_available() else "unavailable",
            cuda_version=torch.version.cuda,
        ),
        python=PythonInfo(
            version=sys.version,
            libraries=python_libs,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global syncmvd_instance
    global syncmvd_instance_xl
    global vae_instance
    global unet_instance
    global controlnet_instance
    global s3_client
    global gpu_lock
    global task_queue
    global task_status
    global processing_task
    global thread_pool
    global replay_config

    logger.info("Starting server...")
    logger.info("App Settings")
    logger.info(f"\n{settings.model_dump_json(indent=2)}")

    logger.info("System Details")
    logger.info(f"\n{get_system_details().model_dump_json(indent=2)}")

    logger.info("Loading AI pipelines...")

    try:
        if settings.sentry_dsn:
            logger.info("Starting sentry SDK...")
            sentry_sdk.init(
                dsn=settings.sentry_dsn.get_secret_value(),
                traces_sample_rate=1.0,
                _experiments={
                    "continuous_profiling_auto_start": True,
                },
            )

        s3_client = create_s3_client(
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key.get_secret_value(),
            secret_key=settings.s3_secret_key.get_secret_value(),
            validate_ssl=settings.s3_validate_ssl,
            addressing_style=settings.s3_addressing_style,
        )

        gpu_lock = Lock()

        thread_pool = ThreadPoolExecutor()

        replay_config = ReplayConfig()

        yield
    except ValueError as e:
        logger.error(f"Failed to load AI pipeline: {e}")
        raise e
    finally:
        if syncmvd_instance is not None:
            del syncmvd_instance
        if syncmvd_instance_xl is not None:
            del syncmvd_instance_xl
        if vae_instance is not None:
            del vae_instance
        if unet_instance is not None:
            del unet_instance
        if controlnet_instance is not None:
            del controlnet_instance
        if s3_client is not None:
            del s3_client
        if gpu_lock is not None and gpu_lock.locked():
            gpu_lock.release()

        logger.info("Unloaded AI pipelines.")
        logger.info("Unloaded S3 client.")
        logger.info("Exiting...")


app = FastAPI(
    title="Sortium Texture Pipeline",
    description="A pipeline for creating texture maps from 3D models.",
    version=settings.version,
    lifespan=lifespan,
)

gradio_blocks = build_gradio_ui(theme="sortium", api_url="http://127.0.0.1:8000")
app = gr.mount_gradio_app(app=app, blocks=gradio_blocks, path="/demo")


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/gpu/status")
async def get_gpu_status() -> GPUInfo:
    return GPUInfo(
        name=torch.cuda.get_device_name(0),
        memory=format_memory(torch.cuda.get_device_properties(0).total_memory),
        cuda_status="available" if torch.cuda.is_available() else "unavailable",
        cuda_version=torch.version.cuda,
        gpu_lock_status="locked"
        if gpu_lock is not None and gpu_lock.locked()
        else "unlocked",
    )


@app.post("/texture/upload")
async def generate_texture_upload(
    config: Annotated[InputConfig, Form()] = None,
    file: Annotated[UploadFile, File()] = None,
    reference_image: UploadFile | None = None,
) -> TextureOutput:
    """Generate texture endpoint, accepts a file upload"""

    input = config

    # Extract suffix from original file name
    suffix = os.path.splitext(file.filename)[1]

    # Copy file to NamedTemporaryFile
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.file.read())
        input.mesh = temp_file.name

    # Check if reference image is provided and replace path with uploaded file path
    if reference_image and input.ip_adapter_scale:
        # Extract suffix from original file name
        reference_imag_suffix = os.path.splitext(reference_image.filename)[1]
        with NamedTemporaryFile(
            delete=False, suffix=reference_imag_suffix
        ) as temp_reference_image:
            input.ip_adapter_image = temp_reference_image.name

    logger.info(f"Received input: {input}")

    return await run_pipeline(input)


@app.post("/texture")
async def generate_texture_s3(input: InputConfig) -> TextureOutput:
    """Generate texture endpoint, accepts a file from S3"""

    # Download mesh file from S3
    s3_key = input.mesh
    bucket_name = settings.s3_bucket_name
    mesh_path = download_file(s3_key, bucket_name, s3_client=s3_client)

    # Download ip_adapter_image if provided
    if input.ip_adapter_image:
        ip_adapter_image_path = download_image_to_tmp(
            HttpUrl(url=input.ip_adapter_image)
        )
        input.ip_adapter_image = ip_adapter_image_path

    # Replace mesh path with downloaded file path
    input.mesh = mesh_path

    return await run_pipeline(input)


async def run_pipeline(input: InputConfig) -> TextureOutput:
    # Check GPU Lock
    if not gpu_lock.acquire(blocking=False):
        logger.error("GPU is busy, please try again later.")
        raise HTTPException(
            status_code=429, detail="GPU is busy, please try again later."
        )

    logger.info(f"Received input: {input}")

    try:
        # Create timer
        start_time = time.perf_counter()

        # Download mesh file from S3
        bucket_name = settings.s3_bucket_name
        # s3_key = input.mesh
        # mesh_path = download_file(s3_key, bucket_name, s3_client=s3_client)

        # # Download ip_adapter_image if provided
        # if input.ip_adapter_image:
        #     ip_adapter_image_path = download_image_to_tmp(
        #         HttpUrl(url=input.ip_adapter_image)
        #     )
        #     input.ip_adapter_image = ip_adapter_image_path

        # # Replace mesh path with downloaded file path
        # input.mesh = mesh_path

        output: TextureOutput = TextureOutput()
        if input.t2i_model == "SD1.5":
            if settings.load_sd15:
                raise ValueError("SD1.5 model is not loaded, please use SDXL instead")
            output_dir = await run_pipeline_in_thread(
                syncmvd_instance=syncmvd_instance, input=input
            )
        elif input.t2i_model == "SDXL":
            # Build pipeline config
            current_config = ReplayConfig(
                base_model=input.t2i_model,
                controlnet=input.cond_type,
                vae=input.custom_style,
                unet=input.custom_style,
            )
            # Check if the current config is different from the previous config
            if not replay_config.equals_to(current_config):
                if input.custom_style == "realistic":
                    logger.info("Using Juggernaut XL model")

                    logger.info("Loading Juggernaut XL...")
                    (
                        vae_instance,
                        unet_instance,
                    ) = await run_preload_checkpoint_in_thread(
                        settings.juggernaut_xl_checkpoint_path
                    )
                    replay_config.vae = "juggernaut"
                    replay_config.unet = "juggernaut"
                elif input.custom_style == "anime":
                    logger.info("Using Anything XL model")
                    logger.info("Loading Anything XL...")
                    (
                        vae_instance,
                        unet_instance,
                    ) = await run_preload_checkpoint_in_thread(
                        settings.anything_xl_checkpoint_path
                    )
                    replay_config.vae = "anything"
                    replay_config.unet = "anything"
                else:
                    logger.info("Using Generic VAE model")
                    logger.info("Loading Generic VAE...")
                    # vae_instance = preload_generic_vae()
                    vae_instance = await run_preload_generic_vae_in_thread()
                    unet_instance = None
                    replay_config.vae = "generic"
                    replay_config.unet = None
                if input.cond_type == "canny":
                    logger.info("Using Canny ControlNet")
                    controlnet_instance = await run_preload_controlnet_in_thread(
                        "diffusers/controlnet-canny-sdxl-1.0"
                    )
                    replay_config.controlnet = "canny"
                elif input.cond_type == "render":
                    logger.info("Using Tile ControlNet")
                    controlnet_instance = await run_preload_controlnet_in_thread(
                        "xinsir/controlnet-tile-sdxl-1.0"
                    )
                    replay_config.controlnet = "tile"
                else:
                    logger.info("Using Depth ControlNet")
                    controlnet_instance = await run_preload_controlnet_in_thread(
                        "diffusers/controlnet-depth-sdxl-1.0"
                    )
                    replay_config.controlnet = "depth"

                syncmvd_instance_xl = await run_preload_pipeline_in_thread(
                    "SDXL", vae_instance, controlnet_instance, unet_instance
                )
                replay_config.base_model = "SDXL"
            else:
                logger.info("Using cached pipeline")

            with sentry_sdk.start_transaction(op="task", name="Run Experiment"):
                output_dir = await run_pipeline_in_thread(syncmvd_instance_xl, input)
        else:
            raise ValueError(f"Invalid t2i_model: {input.t2i_model}")

        if is_valid_output_dir(output_dir):
            output_glb_path = os.path.join(output_dir, "results", "textured.glb")

            # Push the GLB file to S3
            final_key = generate_random_name() + ".glb"
            uploaded_key = upload_file(
                file_path=output_glb_path,
                bucket_name=bucket_name,
                s3_key=final_key,
                s3_client=s3_client,
            )

            if settings.s3_bucket_public_url:
                uploaded_key = f"{settings.s3_bucket_public_url}/{uploaded_key}"

            output.status = OutputStatus.SUCCESS
            output.generated_mesh = HttpUrl(uploaded_key)
    # @TODO: Add extra exception type handlers
    except ValueError as e:
        type, value, traceback = sys.exc_info()
        print_traceback(traceback)
        logger.error(f"Failed to run pipeline: {e}")
        output.status = OutputStatus.ERROR
        output.error_message = str(e)
    except Exception as e:
        type, value, traceback = sys.exc_info()
        print_traceback(traceback)
        logger.error(f"Failed to run pipeline: {e}")

        output.status = OutputStatus.ERROR
        output.error_message = str(e)
    finally:
        end_time = time.perf_counter()
        process_time = end_time - start_time
        logger.info(f"Process time: {process_time:.2f} seconds")
        output.process_time = process_time

        # Release GPU Lock
        gpu_lock.release()

        if output.status == OutputStatus.SUCCESS:
            logger.info(f"Generated mesh: {output.generated_mesh}")
            return output
        elif output.status == OutputStatus.ERROR:
            logger.error(f"Error message: {output.error_message}")
            return HTTPException(status_code=500, detail=output.error_message)
        else:
            return HTTPException(status_code=500, detail="Unknown error")


# Async Fixes for long running tasks
async def run_preload_checkpoint_in_thread(
    checkpoint_path: str,
) -> tuple[AutoencoderKL, UNet2DConditionModel]:
    """
    Run the preload checkpoint in a separate thread

    Args:
        checkpoint_path (str): Checkpoint path

    Returns:
        tuple[AutoencoderKL, UNet2DConditionModel]: AutoencoderKL and UNet2DConditionModel instances
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool, preload_custom_checkpoint, checkpoint_path
    )
    return result


async def run_preload_generic_vae_in_thread() -> AutoencoderKL:
    """
    Run the preload generic VAE in a separate thread

    Returns:
        AutoencoderKL: AutoencoderKL instance
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(thread_pool, preload_generic_vae)
    return result


async def run_preload_controlnet_in_thread(model_name: str) -> ControlNetModel:
    """
    Run the preload controlnet in a separate thread

    Args:
        model_name (str): Model name

    Returns:
        ControlNetModel: ControlNetModel instance
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(thread_pool, preload_controlnet, model_name)
    return result


async def run_preload_pipeline_in_thread(
    t2i_model: str,
    vae: Optional[AutoencoderKL],
    controlnet: Optional[ControlNetModel],
    unet: Optional[UNet2DConditionModel],
) -> StableSyncMVDPipeline | StableSyncMVDPipelineXL:
    """
    Run the preload pipeline in a separate thread

    Args:
        t2i_model (str): T2I model
        vae (Optional[AutoencoderKL]): AutoencoderKL instance
        controlnet (Optional[ControlNetModel]): ControlNetModel instance
        unet (Optional[UNet2DConditionModel]): UNet2DConditionModel instance

    Returns:
        StableSyncMVDPipeline | StableSyncMVDPipelineXL: StableSyncMVDPipeline or StableSyncMVDPipelineXL instance
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool, preload_pipeline, t2i_model, vae, controlnet, unet
    )
    return result


async def run_pipeline_in_thread(
    syncmvd_instance: StableSyncMVDPipeline | StableSyncMVDPipelineXL,
    input: InputConfig,
) -> str:
    """
    Run the pipeline in a separate thread

    Args:
        syncmvd_instance (StableSyncMVDPipeline): Synchronous MVD pipeline instance
        input (InputConfig): Input configuration

    Returns:
        str: Output directory path
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool, run_experiment, syncmvd_instance, input
    )
    return result


def print_traceback(traceback: TracebackType):
    """
    Print the traceback

    Args:
        traceback (TracebackType): Traceback object
    """
    while traceback:
        print(f"Traceback: {traceback.tb_frame}")
        traceback = traceback.tb_next


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
