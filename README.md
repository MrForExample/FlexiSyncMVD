# FlexiSyncMVD

Generate, modify texture for a 3D object from a text or image prompt.


<table style="table-layout: fixed; width: 100%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
  <tr>
  <td>
    <img src=FlexiSyncMVD/assets/gif/batman.gif width="170">
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/david.gif width="170">
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/teapot.gif width="170">
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/vangogh.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"Photo of Batman, sitting on a rock."</td>
    <td style="font-family:courier">"Publicity photo of a 60s movie, full color."</td>
    <td style="font-family:courier">"A photo of a beautiful chintz glided teapot."</td>
    <td style="font-family:courier">"A beautiful oil paint of a stone building in Van Gogh style."</td>
  </tr>
   <tr>
  <td>
    <img src=FlexiSyncMVD/assets/gif/gloves.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/link.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/house.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/luckycat.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A photo of a robot hand with mechanical joints."</td>
    <td style="font-family:courier">"Photo of link in the legend of zelda, photo-realistic, unreal 5."</td>
    <td style="font-family:courier">"Photo of a lowpoly fantasy house from warcraft game, lawn."</td>
    <td style="font-family:courier">"Blue and white pottery style lucky cat with intricate patterns."</td>
  </tr>

  <tr>
  <td>
    <img src=FlexiSyncMVD/assets/gif/chair.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/Moai.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/sneakers.gif width="170">
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/dragon.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A photo of an beautiful embroidered seat with royal patterns"</td>
    <td style="font-family:courier">"Photo of James Harden."</td>
    <td style="font-family:courier">"A photo of a gray and black Nike Airforce high top sneakers."</td>
    <td style="font-family:courier">"A photo of a Chinese dragon sculpture, glazed facing, vivid colors."</td>
  </tr>

  <tr>
  <td>
    <img src=FlexiSyncMVD/assets/gif/guardian.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/horse.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/jackiechan.gif width="170" >
  </td>
  <td>
    <img src=FlexiSyncMVD/assets/gif/knight.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A muscular man wearing grass hula skirt."</td>
    <td style="font-family:courier">"Photo of a horse."</td>
    <td style="font-family:courier">"A Jackie Chan figure."</td>
    <td style="font-family:courier">"A photo of a demon knight, flame in eyes, warcraft style."</td>
  </tr>

</table>

## Installation
The program is developed and tested on Linux system with Nvidia GPU. If you find compatibility issues on Windows platform, you can also consider using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

To install, first clone the repository and install the basic dependencies

```bash
# NVIDIA Driver
sudo apt update
sudo apt upgrade
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers list
sudo ubuntu-drivers install

# Conda Python Environment
git clone https://github.com/Sortium-io/FlexiSyncMVD.git
cd FlexiSyncMVD
conda create -n FlexiSyncMVD python=3.11
conda activate FlexiSyncMVD
conda install nvidia/label/cuda-12.1.1::cuda-toolkit -c nvidia/label/cuda-12.1.1

# Python Dependencies
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
conda install pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2
```

The pretrained models will be downloaded automatically on demand, including:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [lllyasviel/control_v11f1p_sd15_depth](lllyasviel/control_v11f1p_sd15_depth)
- [lllyasviel/control_v11p_sd15_normalbae](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0)

The fine-tuned checkpoints need to download manually and put into `FlexiSyncMVD/_checkpoints`
- [juggernautXL_juggXIByRundiffusion](https://civitai.com/models/133005/juggernaut-xl?modelVersionId=782002)
- [leosamsHelloworldXL_helloworldXL70](https://civitai.com/models/43977/leosams-helloworld-xl?modelVersionId=570138)
- [AnythingXL_xl](https://civitai.com/models/9409/or-anything-xl?modelVersionId=384264)

You can try out the method with the following pre-processed meshes and configs:
- [Flying Car](FlexiSyncMVD/data/flying_car/)
- [Monster](FlexiSyncMVD/data/monster/)
- [Cammy](FlexiSyncMVD/data/cammy/)

## Inference
```bash
python FlexiSyncMVD/main.py --config {your config}.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/flying_car/config.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/monster/config.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/cammy/config_sdxl_ip.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/cammy/config_sdxl_anime.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/cammy/config_sdxl_realistic.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/cammy/config_sdxl_ip_anime.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/cammy/config_sdxl_ip_realistic.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/marcille/config_sdxl_anime.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/marcille/config_sdxl_realistic.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/lamp/config_sdxl_realistic.yaml
# python FlexiSyncMVD/main.py --config FlexiSyncMVD/data/gate/config_sdxl_realistic.yaml
```
Refer to [config.py](src/configs.py) for the list of arguments and settings you can adjust. You can change these settings by including them in a `.yaml` config file or passing the related arguments in command line; values specified in command line will overwrite those in config files.

When no output path is specified, the generated result will be placed in the same folder as the config file by default.

## Reference
[Pytorch](https://pytorch.org/) & [Diffusers](https://github.com/huggingface/diffusers) implemetation base on the paper:

- **[Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/pdf/2311.12891)**

- **[SyncTweedies: A General Generative Framework Based on Synchronized Diffusions](https://arxiv.org/abs/2403.14370)**

- **[FlexiTex: EnhancingSyncTweedies Texture Generation via Visual Guidance](https://arxiv.org/abs/2409.12431)**