from pydantic import BaseModel
from typing import Optional, List
from FlexiSyncMVD.src.configs import parse_config

class ServerConfig(BaseModel):
    port: int = 8000
    host: str = "0.0.0.0"
    version: str = "0.1.0"
    sentry_dsn: Optional[str] = None

class S3Config(BaseModel):
    endpoint_url: str = "http://localhost:9000"
    bucket_name: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket_public_url: Optional[str] = None
    validate_ssl: bool = True
    addressing_style: str = "path"

class PipelineConfig:
    def __init__(self, **kwargs):
        # Parse config using the original configs.py
        defaults = parse_config().__dict__
        self.__dict__.update(defaults)
        # Override with any provided kwargs
        self.__dict__.update(kwargs)

class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    s3: S3Config = S3Config()
    pipeline: PipelineConfig = None
    load_sd15: bool = False

    @classmethod
    def load_config(cls, pipeline_overrides=None):
        pipeline_config = PipelineConfig(**(pipeline_overrides or {}))
        return cls(pipeline=pipeline_config)