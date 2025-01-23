import os
import tempfile
import uuid
import httpx
from pydantic import HttpUrl
import boto3
from typing import Optional
from botocore.config import Config
from botocore.exceptions import ClientError
from server_demo.src.logger import logger


def create_s3_client(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    validate_ssl: bool,
    addressing_style: str = "virtual",
) -> boto3.client:
    """Create an S3 client with the given endpoint URL, access key, and secret key."""

    session = boto3.Session()
    client = session.client(
        "s3",
        config=Config(
            s3={
                "addressing_style": "virtual"
                if addressing_style == "virtual"
                else "path"
            }
        ),
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=validate_ssl,
    )

    return client


def download_file(
    s3_key: str,
    bucket_name: str,
    file_name: Optional[str] = None,
    s3_client: boto3.client = None,
) -> str:
    """Download a file from an S3 bucket."""

    # Create a temporary file if no file name is provided
    if file_name is None:
        file_name = tempfile.NamedTemporaryFile(
            delete=False, prefix="", suffix=s3_key
        ).name

    s3_client.download_file(bucket_name, s3_key, file_name)

    logger.info(f"Downloaded file from S3 bucket: {bucket_name} with key: {s3_key}")

    return file_name


def upload_file(
    file_path: str,
    bucket_name: str,
    s3_key: Optional[str] = None,
    content_type: str = "model/gltf-binary; charset=binary",
    s3_client: boto3.client = None,
) -> str:
    logger.info(f"Uploading file to S3 bucket: {bucket_name}")
    logger.info(f"File path: {file_path}")
    logger.info(f"S3 key: {s3_key}")

    file_name = file_path.split("/")[-1]

    try:
        s3_client.create_bucket(Bucket=bucket_name)
    except ClientError as s3_client_error:
        error_code = s3_client_error.response["Error"]["Code"]
        ignored_extensions = ["BucketAlreadyExists", "BucketAlreadyOwnedByYou"]
        if error_code not in ignored_extensions:
            raise s3_client_error

    final_key = s3_key if s3_key is not None else file_name

    # Append the .glb suffix if it is not already present
    if not final_key.endswith(".glb"):
        final_key = f"{final_key}.glb"

    with open(file_path, "rb") as file:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=final_key,
            Body=file,
            ACL="public-read",
            Metadata={"Content-Type": content_type},
        )
        logger.info(f"Uploaded file to S3 bucket: {bucket_name} with key: {s3_key}")

        return final_key


def is_valid_output_dir(output_dir) -> bool:
    """
    Check if the output directory contains the required files.

    Args:
        output_dir (str): The path to the output directory.

    Returns:
        bool: True if the output directory is valid, False otherwise.
    """

    # Validate if output directory exists
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist")

    # Validate if output directory is empty
    if not os.listdir(output_dir):
        raise ValueError(f"Output directory {output_dir} is empty")

    # Check if folder results exists and contains the following files: textured.obj, textured.mtl, textured.png and textured_views_rgb.jpg
    if not os.path.exists(os.path.join(output_dir, "results")):
        raise ValueError(
            f"Output directory {output_dir} does not contain results folder"
        )

    if not os.path.exists(os.path.join(output_dir, "results", "textured.glb")):
        raise ValueError(
            f"Output directory {output_dir} does not contain textured.glb file"
        )

    if not os.path.exists(
        os.path.join(output_dir, "results", "textured_views_rgb.jpg")
    ):
        raise ValueError(
            f"Output directory {output_dir} does not contain textured_views_rgb.jpg file"
        )

    return True


def generate_random_name() -> str:
    """
    Generate a random name using UUID.

    Returns:
        str: A random name generated using UUID.
    """
    return str(uuid.uuid4())


def download_image_to_tmp(image_url: HttpUrl) -> str:
    """
    Download an image from a given URL and save it to a temporary file.

    Args:
        image_url (HttpUrl): The URL of the image to download.

    Returns:
        str: The path to the downloaded image file.
    """

    # Download the image from the URL
    file_extension = os.path.splitext(str(image_url))[1]
    image_path = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension).name

    # Download image usi g httpx
    with open(image_path, "wb") as f:
        with httpx.Client() as client:
            response = client.get(str(image_url))
            f.write(response.content)

    return image_path


def format_memory(memory: int) -> str:
    """
    Format the memory size in bytes to a human-readable format.

    Args:

        memory (int): The memory size in bytes.

    Returns:

        str: The memory size in a human-readable format.
    """
    memory = memory / 1024**3
    memory = f"{memory:.2f} GB"
    return memory
