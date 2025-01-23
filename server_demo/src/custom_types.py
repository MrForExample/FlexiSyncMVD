from enum import Enum
import json
from pydantic import BaseModel, Field, HttpUrl, model_validator
from typing import Any, Optional, List


class PipelineConfig(BaseModel):
    # File Config
    config: Optional[str] = Field(None, description="Config file path")
    mesh: Optional[str] = Field(None, description="Mesh file path")
    mesh_config_relative: bool = Field(
        False,
        description="Search mesh file relative to the config path instead of current working directory",
    )
    output: Optional[str] = Field(
        None,
        description="If not provided, use the parent directory of config file for output",
    )
    prefix: str = Field("MVD")
    use_mesh_name: bool = Field(False)
    timeformat: Optional[str] = Field(
        "%d%b%Y-%H%M%S",
        description="Setting to None will not use time string in output directory",
    )

    # Diffusion Config
    t2i_model: str = Field("SD1.5", description="Support SD1.5 and SDXL")
    checkpoint_path: Optional[str] = Field(
        None, description="If provided, will use custom checkpoint for the model"
    )
    prompt: Optional[str] = Field(None)
    negative_prompt: str = Field(
        "oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect."
    )
    steps: int = Field(30)
    guidance_scale: float = Field(
        15.5, description="Recommend above 12 to avoid blurriness"
    )
    seed: int = Field(0)

    # ControlNet Config
    cond_type: str = Field(
        "depth",
        description="Support depth and normal, less multi-face in normal mode, but sometimes less details",
    )
    guess_mode: bool = Field(False)
    conditioning_scale: float = Field(0.7)
    conditioning_scale_end: float = Field(
        0.9,
        description="Gradually increasing conditioning scale for better geometry alignment near the end",
    )
    control_guidance_start: float = Field(0.0)
    control_guidance_end: float = Field(0.99)
    guidance_rescale: float = Field(0.0, description="Not tested")

    # IP Adapter Config
    ip_adapter_image: Optional[str] = Field(
        None, description="Use reference image to guide the diffusion process"
    )
    ip_adapter_scale: float = Field(
        1.0, description="The strength of the reference image in the diffusion process"
    )

    # Multi-View Config
    latent_view_size: int = Field(
        96,
        description="Larger resolution, less aliasing in latent images; quality may degrade if much larger than trained resolution of networks",
    )
    latent_tex_size: int = Field(
        768,
        description="Originally 1536 in paper, use lower resolution to save VRAM and runtime",
    )
    rgb_view_size: int = Field(768)
    rgb_tex_size: int = Field(1024)
    max_batch_size: int = Field(48)
    camera_azims: List[int] = Field(
        [-180, -135, -90, -45, 0, 45, 90, 135],
        description="Place the cameras at the listed azim angles",
    )
    no_top_cameras: bool = Field(
        False, description="Two cameras added to paint the top surface"
    )
    mvd_end: float = Field(
        0.8, description="Time step to stop texture space aggregation"
    )
    mvd_exp_start: float = Field(
        0.0,
        description="Initial exponent for weighted texture space aggregation, low value encourages consistency",
    )
    mvd_exp_end: float = Field(
        6.0,
        description="End exponent for weighted texture space aggregation, high value encourages sharper results",
    )
    ref_attention_end: float = Field(
        0.2, description="Lower->better quality; higher->better harmonization"
    )
    shuffle_bg_change: float = Field(
        0.4, description="Use only black and white background after certain timestep"
    )
    shuffle_bg_end: float = Field(
        0.8,
        description="Don't shuffle background after certain timestep. Background color may bleed onto object",
    )
    mesh_scale: float = Field(
        1.0, description="Set above 1 to enlarge object in camera views"
    )
    keep_mesh_uv: bool = Field(
        False, description="Don't use Xatlas to unwrap UV automatically"
    )

    # Logging Config
    log_interval: int = Field(10)
    view_fast_preview: bool = Field(
        False,
        description="Use color transformation matrix instead of decoder to log view images",
    )
    tex_fast_preview: bool = Field(
        False,
        description="Use color transformation matrix instead of decoder to log texture images",
    )


class Style(str, Enum):
    JUGGERNAUT_XL = "realistic"
    ANYTHING_XL = "anime"


class InputConfig(BaseModel):
    mesh: Optional[str] = Field(None, description="Mesh file path")
    t2i_model: str = Field("SD1.5", description="Support SD1.5 and SDXL")
    custom_style: Optional[Style] = Field(None)
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(
        "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    )
    steps: int = Field(30)
    cond_type: str = Field("depth", description="Support depth, canny and normal")
    seed: int = Field(0)
    ip_adapter_scale: Optional[float] = Field(0.5)
    ip_adapter_image: Optional[str] = Field(None)
    log_interval: int = Field(10)
    mesh_scale: float = Field(1.0)
    tex_fast_preview: bool = Field(False)
    view_fast_preview: bool = Field(False)
    keep_mesh_uv: bool = Field(False)
    latent_view_size: int = Field(96)
    latent_tex_size: int = Field(768)
    rgb_view_size: int = Field(768)
    rgb_tex_size: int = Field(1024)
    conditioning_scale: float = Field(0.7)
    conditioning_scale_end: float = Field(0.9)

    @model_validator(mode="before")
    def check_value(cls, data: Any) -> dict:
        print(data)
        print(type(data))
        if isinstance(data, str):
            data_dict: dict = json.loads(data)
        else:
            data_dict: dict = data
        return data_dict


class OutputStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class TextureOutput(BaseModel):
    status: OutputStatus = OutputStatus.PENDING
    generated_mesh: Optional[HttpUrl] = Field(None)
    error_message: Optional[str] = Field(None)
    process_time: Optional[float] = Field(None)


class LoggingConfig(BaseModel):
    output_dir: str = Field(...)
    log_interval: int = Field(10)
    view_fast_preview: bool = Field(False)
    tex_fast_preview: bool = Field(False)


class GPUInfo(BaseModel):
    name: Optional[str] = None
    memory: Optional[str] = None
    cuda_status: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_lock_status: Optional[str] = None


class CPUInfo(BaseModel):
    name: Optional[str] = None
    cores: Optional[int] = None
    ram: Optional[str] = None


class LibraryInfo(BaseModel):
    pytorch_version: Optional[str] = None
    fastapi_version: Optional[str] = None


class PythonInfo(BaseModel):
    version: Optional[str] = None
    libraries: Optional[LibraryInfo] = None


class OSInfo(BaseModel):
    name: Optional[str] = None


class SystemDetails(BaseModel):
    os: Optional[OSInfo] = None
    cpu: Optional[CPUInfo] = None
    gpu: Optional[GPUInfo] = None
    python: Optional[PythonInfo] = None


class ReplayConfig(BaseModel):
    base_model: Optional[str] = Field(None)
    controlnet: Optional[str] = Field(None)
    vae: Optional[str] = Field(None)
    unet: Optional[str] = Field(None)

    def equals_to(self, other: "ReplayConfig") -> bool:
        if not isinstance(other, ReplayConfig):
            return False
        return self.model_dump() == other.model_dump()


# Function to merge InputConfig into PipelineConfig
def merge_configs(
    base_config: PipelineConfig, input_config: InputConfig
) -> PipelineConfig:
    # Use input_config values to update base_config
    updated_data = base_config.model_dump()
    input_data = input_config.model_dump(exclude_unset=True)
    updated_data.update(input_data)

    # Create a new instance of PipelineConfig with merged values
    merged_config = PipelineConfig(**updated_data)

    return merged_config
