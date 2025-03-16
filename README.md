# FlexiSyncMVD - Server Texture Pipeline

This repository combines the power of **FlexiSyncMVD**, a tool for generating and modifying textures for 3D objects from text or image prompts, with a robust **Server Texture Pipeline**. Built on top of FlexiSyncMVD, this pipeline leverages modern technologies like FastAPI, Pydantic, Boto3, Docker, and Trimesh to provide a scalable, modular, and flexible system for texture generation. The system is designed to integrate seamlessly with new models and datasets, making it ideal for both research and production environments.

## Description of How the System Works

The **Server Texture Pipeline** extends the capabilities of FlexiSyncMVD by adding a server-side API to process texture generation requests asynchronously. The system operates as follows:

1. **Input Processing**: Users submit requests via a RESTful API (e.g., `POST /texture`) with parameters such as mesh file name, text prompt, model type (SD1.5 or SDXL), and various configuration options (e.g., steps, conditioning type, seed). The input mesh is typically retrieved from an S3 bucket.

2. **Pipeline Execution**: The system uses a preloaded diffusion pipeline (based on Stable Diffusion with ControlNet) to generate textures. Depending on the selected model (SD1.5 or SDXL), it may load specific checkpoints (e.g., Juggernaut XL, Anything XL) and control mechanisms (e.g., depth, canny, tile). The pipeline employs synchronized multi-view diffusion to ensure consistent texture application across different camera angles of the 3D object.

3. **Texture Generation**: The process involves:
   - **Multi-View Rendering**: The 3D mesh is rendered from multiple camera angles (default: [-180, -135, -90, -45, 0, 45, 90, 135] degrees).
   - **Diffusion Process**: A text-guided diffusion model generates latent representations, guided by ControlNet conditions (e.g., depth maps) and optional IP adapter images.
   - **Texture Mapping**: The resulting textures are mapped onto the mesh using techniques like UV unwrapping (via Trimesh) and aggregation across views.

4. **Output Handling**: The textured 3D model is saved as a `.glb` file, uploaded to an S3 bucket, and a public URL is returned to the user. Logging is performed at specified intervals to track progress.

5. **Resource Management**: The system uses GPU locking and thread pooling to manage resources efficiently, ensuring multiple requests can be handled without conflicts.

The pipeline supports customization via configuration files or command-line arguments, with defaults managed by the `configs.py` module in FlexiSyncMVD. Server-specific settings (e.g., port, S3 credentials) are handled via a `.env` file.

## Installation

The system is optimized for Linux with an Nvidia GPU, though Windows users can use WSL. Below are the detailed installation steps:

### Prerequisites

- **Nvidia GPU Drivers**: Install the latest drivers.

  ```bash
  sudo apt update
  sudo apt upgrade
  sudo apt install ubuntu-drivers-common
  sudo ubuntu-drivers list
  sudo ubuntu-drivers install
  ```

- **UV Tool**: Install the UV dependency manager.

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  For Windows (PowerShell):

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Project Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/MrForExample/FlexiSyncMVD.git
   cd FlexiSyncMVD
   ```

2. Create and activate a virtual environment:

   ```bash
   uv venv && source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   uv sync
   uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
   ```

4. Install additional Python dependencies (required for FlexiSyncMVD):

   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu121
   pip install git+https://github.com/openai/CLIP.git
   pip install -r requirements.txt
   conda install pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2
   ```

### Pretrained Models

- Models are downloaded automatically on demand:
  - [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - [lllyasviel/control_v11f1p_sd15_depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)
  - [lllyasviel/control_v11p_sd15_normalbae](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae)
  - [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  - [diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0)
- Fine-tuned checkpoints must be manually downloaded and placed in `FlexiSyncMVD/_checkpoints`:
  - [juggernautXL_juggXIByRundiffusion](https://civitai.com/models/133005/juggernaut-xl?modelVersionId=782002)
  - [leosamsHelloworldXL_helloworldXL70](https://civitai.com/models/43977/leosams-helloworld-xl?modelVersionId=570138)
  - [AnythingXL_xl](https://civitai.com/models/9409/or-anything-xl?modelVersionId=384264)

### Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your settings (default values are for MinIO):

```ini
PORT=8000
HOST=0.0.0.0
VERSION=0.0.1
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_VALIDATE_SSL=false
S3_BUCKET_NAME=data
S3_BUCKET_PUBLIC_URL=http://localhost:9000/data
S3_ADDRESSING_STYLE=path
```

### Docker Setup (Optional)

For containerized deployment with GPU support:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian).
2. Enable it in `/etc/docker/daemon.json`:

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

3. Restart Docker:

   ```bash
   sudo systemctl restart docker
   ```

4. Build and run:

   ```bash
   DOCKER_BUILDKIT=0 docker compose --profile full build
   docker compose --profile full up -d
   ```

5. Verify GPU support:
   - **GET** <http://127.0.0.1:8000/gpu/status> should return `{"cuda": "available"}`.

### Local S3 Service (Optional)

Run a local MinIO instance:

```bash
docker compose --profile local up -d
```

This creates a `server_demo` bucket with example objects.

## Training & Inference Instructions

### Training

The pipeline itself does not require training, as it relies on pre-trained diffusion models. However, you can fine-tune models using the FlexiSyncMVD framework by following these steps:

1. Prepare a dataset of 3D meshes and corresponding texture prompts.
2. Modify the `main.py` script to include training loops using the Diffusers library.
3. Adjust hyperparameters (e.g., `steps`, `guidance_scale`) in `configs.py` or a custom `.yaml` file.
4. Run training with a custom script (not included by default; refer to the [Diffusers documentation](https://github.com/huggingface/diffusers)).

### Inference

#### Using the Server API

1. Start the server:

   ```bash
   uv run server.py
   ```

2. Send a request (example for a flying car):

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
     "output_dir": "/tmp/tmple_ww7ni/_exp/MVD_21Nov2024-132758",
     "generated_mesh": "http://localhost:9000/data/6c23d011-8545-4b25-8ac0-33b4821b6b4a.glb"
   }
   ```

#### Using the Command Line (FlexiSyncMVD)

Run inference with pre-configured examples:

```bash
python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/flying_car/config.yaml
python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/monster/config.yaml
python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/cammy/config_sdxl_ip.yaml
```

Customize settings via `.yaml` files or command-line arguments (see `configs.py`).

## Showcase of Results and Training Logs

### Showcase of Results

The pipeline generates high-quality textured 3D models based on text prompts. Below are some examples:

<table style="table-layout: fixed; width: 100%;">
  <col style="width: 25%;">
  <col style="width: 25%;">
  <col style="width: 25%;">
  <col style="width: 25%;">
  <tr>
    <td><img src="FlexiSyncMVD/assets/gif/batman.gif" width="170"></td>
    <td><img src="FlexiSyncMVD/assets/gif/david.gif" width="170"></td>
    <td><img src="FlexiSyncMVD/assets/gif/teapot.gif" width="170"></td>
    <td><img src="FlexiSyncMVD/assets/gif/vangogh.gif" width="170"></td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"Photo of Batman, sitting on a rock."</td>
    <td style="font-family:courier">"Publicity photo of a 60s movie, full color."</td>
    <td style="font-family:courier">"A photo of a beautiful chintz glided teapot."</td>
    <td style="font-family:courier">"A beautiful oil paint of a stone building in Van Gogh style."</td>
  </tr>
  <tr>
    <td><img src="FlexiSyncMVD/assets/gif/gloves.gif" width="170"></td>
    <td><img src="FlexiSyncMVD/assets/gif/link.gif" width="170"></td>
    <td><img src="FlexiSyncMVD/assets/gif/house.gif" width="170"></td>
    <td><img src="FlexiSyncMVD/assets/gif/luckycat.gif" width="170"></td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A photo of a robot hand with mechanical joints."</td>
    <td style="font-family:courier">"Photo of Link in the Legend of Zelda, photo-realistic, Unreal 5."</td>
    <td style="font-family:courier">"Photo of a lowpoly fantasy house from Warcraft game, lawn."</td>
    <td style="font-family:courier">"Blue and white pottery style lucky cat with intricate patterns."</td>
  </tr>
</table>

### Training Logs (Example)

When running inference, logs are generated to track progress. Example log output for the flying car example:

```
[2025-03-16 10:00:00] INFO: Starting server...
[2025-03-16 10:00:01] INFO: App Settings
{
  "port": 8000,
  "host": "0.0.0.0",
  "version": "0.0.1",
  "sentry_dsn": null
}
[2025-03-16 10:00:02] INFO: System Details
{
  "cuda": "available",
  "gpu_count": 1,
  "memory_total": "16GB"
}
[2025-03-16 10:00:03] INFO: Loading AI pipelines...
[2025-03-16 10:00:10] INFO: Received input: {"mesh": "free_merc_hovercar_normalized.glb", "t2i_model": "SD1.5", "prompt": "A pink metal flying car...", "steps": 30, "cond_type": "depth", "seed": 1234, "log_interval": 10, "mesh_scale": 1}
[2025-03-16 10:00:15] INFO: Process time: 12.34 seconds
[2025-03-16 10:00:15] INFO: Generated mesh: http://localhost:9000/data/6c23d011-8545-4b25-8ac0-33b4821b6b4a.glb
```

Logs are saved in the `output_dir` specified in the response, with images logged at intervals (e.g., every 10 steps if `log_interval=10`).

## References

- [Pytorch](https://pytorch.org/)
- [Diffusers](https://github.com/huggingface/diffusers)
- [Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/pdf/2311.12891)
- [SyncTweedies: A General Generative Framework Based on Synchronized Diffusions](https://arxiv.org/abs/2403.14370)
- [FlexiTex: Enhancing SyncTweedies Texture Generation via Visual Guidance](https://arxiv.org/abs/2409.12431)
