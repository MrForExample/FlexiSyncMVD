from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    port: Optional[int] = 8000
    host: Optional[str] = "0.0.0.0"
    version: Optional[str] = "0.1.0"

    # s3 settings
    s3_endpoint_url: Optional[str] = "http://localhost:9000"
    s3_bucket_name: Optional[str] = None
    s3_access_key: Optional[SecretStr]
    s3_secret_key: Optional[SecretStr]
    s3_bucket_public_url: Optional[str] = None
    s3_validate_ssl: Optional[bool] = False
    s3_addressing_style: Optional[str] = "virtual"

    # AI Models settings
    hf_home: Optional[str] = None
    load_sd15: Optional[bool] = False
    juggernaut_xl_checkpoint_path: Optional[str] = (
        "FlexiSyncMVD/_checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
    )
    anything_xl_checkpoint_path: Optional[str] = (
        "./FlexiSyncMVD/_checkpoints/AnythingXL_xl.safetensors"
    )

    # Sentry
    sentry_dsn: Optional[SecretStr] = None

    model_config = SettingsConfigDict(env_file=".env")
