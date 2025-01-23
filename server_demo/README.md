# Server Texture Pipeline

This repository contains the code for the Server Texture pipeline, built on top of FlexiSyncMVD. The pipeline is designed to be modular and flexible, allowing for easy integration of new models and datasets. The following technologies are included within the current implementation:

- **UV**: A tool runner and dependency manager for Python projects.
- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
- **Pydantic**: Data validation and settings management using Python type annotations.
- **Boto3**: The AWS SDK for Python to interact with AWS services. Used for uploading and downloading files from S3.
- **Docker**: A set of platform as a service (PaaS) products that use OS-level virtualization to deliver software in packages called containers.
- **Docker Compose**: A tool for defining and running multi-container Docker applications.
- **Trimesh**: A pure Python (2.7-3.7) library for loading and using triangular meshes with an emphasis on watertight surfaces. Used to convert the output of the pipeline from a `.obj` file into a compatible `.glb`.

## Installation

### UV Installation

We need to install the UV tool before moving forward:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If using Windows you can install the UV tool by running the following command:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


There multiple other ways to install `uv`, like `brew` or `cargo`, you can find detailed instructions at <https://github.com/MrForExample/FlexiSyncMVD.git>.

### Project Setup

Clone the repository at the desired location:

```bash
git clone https://github.com/MrForExample/FlexiSyncMVD.git
```

Navigate to the repository directory:

```bash
cd FlexiSyncMVD
```

Create a new virtual environment and activate it:

```bash
uv venv && source .venv/bin/activate
```

Next step is to install the project dependencies:

```bash
uv sync
```

In this project we a special case where we need to install the `pytorch3d` library from source, to do that you can run the following command:

```bash
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Running the Application

### Environment Variables

You need a valid `.env` file to be able to execute the service. You can create a new `.env` file by copying the `.env.example` file:

```bash
cp .env.example .env
```

These are the available environment variables:

```ini
PORT=8000 # Server port
HOST=0.0.0.0 # Server host
VERSION=0.0.1 # Application version
S3_ENDPOINT_URL=http://localhost:9000 # S3 endpoint URL (API endpoint for the S3 service)
S3_ACCESS_KEY=minioadmin # S3 access key
S3_SECRET_KEY=minioadmin # S3 secret key
S3_VALIDATE_SSL=false # S3 validate SSL (If we should validate the SSL certificate, set to false for local development)
S3_BUCKET_NAME=data # S3 bucket name (Used for storing the assets)
S3_BUCKET_PUBLIC_URL=http://localhost:9000/data # S3 bucket public URL (Used for public access to the bucket assets, in a real world scenario this should be a CDN URL)
S3_ADDRESSING_STYLE=path # S3 addressing style (Path (path) or VirtualHost (virtual))
```

The values presented here are based on the default MinIO configuration. You can change the values to match your S3 service configuration.

Reference the [Local S3 Service](#local-s3-service) section for more information on how to start a local S3 service using MinIO.

### Starting the FastAPI Server

You can now run the application by executing the following command:

```bash
uv run server.py
```

The application will be available at the following URL:

- __GET__ <http://127.0.0.1:8000/>

### Testing the Application

You can execute the following request to validate the application:

```bash
curl -X 'POST' \
  'http://localhost:8000/texture' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "mesh": "free_merc_hovercar_normalized.glb",
  "t2i_model": "SD1.5",
  "prompt": "A pink metal flying car with black window on each side of the car door, futuristic cyberpunk style, pure grey background, detailed, 8k",
  "steps": 30,
  "cond_type": "depth",
  "seed": 1234,
  "log_interval": 10,
  "mesh_scale": 1
}'
```

This request uses an s3 object as input and returns a new s3 object with the generated texture. The request body contains the following parameters:

- `mesh`: The name of the mesh file to be used as input.
- `t2i_model`: The name of the text-to-image model to be used.
- `prompt`: The text prompt to be used as input for the text-to-image model.
- `steps`: The number of steps to be used in the optimization process.
- `cond_type`: The type of condition to be used in the optimization process.
- `seed`: The seed to be used in the optimization process.
- `log_interval`: The interval to log the optimization process.
- `mesh_scale`: The scale to be used in the optimization process.

This request should generate a response similar to this:

```json
{
  "status": "success",
  "output_dir": "/tmp/tmple_ww7ni/_exp/MVD_21Nov2024-132758",
  "generated_mesh": "https://server_demo-ai-uploads-bucket.nyc3.digitaloceanspaces.com/6c23d011-8545-4b25-8ac0-33b4821b6b4a.glb"
}
```

You can access the generated mesh by opening the `generated_mesh` URL in your browser.

> __INFO:__ Future iterations will include the asset thumbnail and the depth map as additional outputs.

## Docker Instructions

The build process needs GPU support to works properly, so you need to have the NVIDIA Container Toolkit installed on your system. You can follow the instructions on the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) to install it.

After installing the NVIDIA Container Toolkit, you need to enable it inside docker by adding the following line to the `/etc/docker/daemon.json` file:

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
	"nvidia": {
	  "path": "nvidia-container-runtime",
	  "runtimeArgs": []
	}
  }
}
```

After adding the configuration, you need to restart the docker service by running the following command:

```bash
sudo systemctl restart docker
```

After enabling the NVIDIA Container Toolkit, you can build the docker image by running the following command:

```bash
DOCKER_BUILDKIT=0 docker compose --profile full build
```

You can validate if the GPU is enabled inside the container by opening the following URL:

- __GET__ <http://127.0.0.1:8000/gpu/status>

This should return the following response:

```json
{
  "cuda": "available"
}
```

### Running the Docker Container

You can run the docker container by running the following command:

```bash
docker compose --profile full up -d
```

This will start the container in the background, and you can access the application by opening the following URL:

- __GET__ <http://127.0.0.1:8000/>

You can stop the container by running the following command:

```bash
docker compose --profile full down
```

### Local S3 Service

You can also run a local S3 service using MinIO. You can start the MinIO service by running the following command:

```bash
docker compose --profile local up -d
```

This will start the MinIO service in the background, and create a new bucket called `server_demo` contained some example objects that you can use to make requests. The access credentials are the same as the ones defined inside the `.env.example` file.

## Example Requests

You can use the following requests to test the application:

### Car - Using SD1.5

```bash
curl -X 'POST' \
  'http://localhost:8000/texture' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "mesh": "free_merc_hovercar_normalized.glb",
  "t2i_model": "SD1.5",
  "prompt": "A pink metal flying car with black window on each side of the car door, futuristic cyberpunk style, pure grey background, detailed, 8k",
  "steps": 30,
  "cond_type": "depth",
  "seed": 1234,
  "log_interval": 10,
  "mesh_scale": 1
}'
```

Expected response:

```json
{
  "status": "success",
  "output_dir": "/tmp/tmplwbt1jll/_exp/MVD_21Nov2024-150212",
  "generated_mesh": "http://localhost:9000/server_demo/c59102ed-6fc2-4cc4-b219-6972e4396a94.glb"
}
```

### Cammy - SD 1.5

```bash
curl -X 'POST' \
  'http://localhost:8000/texture' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "mesh": "Cammy_Normalized_no_color.glb",
  "t2i_model": "SD1.5",
  "prompt": "Cammy from Street Fighter, she wears a tight green V shape bodysuit, bare skin legs from feet to thighs with no pants to cover it. A red beret with a black star tops her head, and she has long blonde braids. Her red combat gauntlets and piercing blue eyes emphasize her readiness for battle, detailed, 8k",
  "steps": 30,
  "cond_type": "depth",
  "seed": 4399171738989,
  "log_interval": 10,
  "mesh_scale": 0.9,
  "conditioning_scale": 0.3,
  "conditioning_scale_end": 0.7
}'
```