# FlexiSyncMVD

Generate, modify texture for a 3D object from a text or image prompt.


<table style="table-layout: fixed; width: 100%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
        <col style="width: 25%;">
  <tr>
  <td>
    <img src=assets/gif/batman.gif width="170">
  </td>
  <td>
    <img src=assets/gif/david.gif width="170">
  </td>
  <td>
    <img src=assets/gif/teapot.gif width="170">
  </td>
  <td>
    <img src=assets/gif/vangogh.gif width="170">
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
    <img src=assets/gif/gloves.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/link.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/house.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/luckycat.gif width="170">
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
    <img src=assets/gif/chair.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/Moai.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/sneakers.gif width="170">
  </td>
  <td>
    <img src=assets/gif/dragon.gif width="170">
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
    <img src=assets/gif/guardian.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/horse.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/jackiechan.gif width="170" >
  </td>
  <td>
    <img src=assets/gif/knight.gif width="170">
  </td>
  </tr>
  <tr style="vertical-align: text-top;">
    <td style="font-family:courier">"A muscular man wearing grass hula skirt."</td>
    <td style="font-family:courier">"Photo of a horse."</td>
    <td style="font-family:courier">"A Jackie Chan figure."</td>
    <td style="font-family:courier">"A photo of a demon knight, flame in eyes, warcraft style."</td>
  </tr>

</table>

## Installation :wrench:
The program is developed and tested on Linux system with Nvidia GPU. If you find compatibility issues on Windows platform, you can also consider using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

To install, first clone the repository and install the basic dependencies
```bash
git clone https://github.com/LIU-Yuxin/SyncMVD.git
cd SyncMVD
conda create -n syncmvd python=3.8
conda activate syncmvd
pip install -r requirements.txt
```
Then install PyTorch3D through the following URL (change the respective Python, CUDA and PyTorch version in the link for the binary compatible with your setup), or install according to official [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html
```
The pretrained models will be downloaded automatically on demand, including:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [lllyasviel/control_v11f1p_sd15_depth](lllyasviel/control_v11f1p_sd15_depth)
- [lllyasviel/control_v11p_sd15_normalbae](https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae) 

## Data
The current program based on [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library requires a input `.obj` mesh with `.mtl` material and related textures to read the original UV mapping of the object, which may require manual cleaning. Alternatively the program also support auto unwrapping based on [XAtlas](https://github.com/jpcy/xatlas) to load mesh that does not met the above requirements. The program also supports loading `.glb` mesh, but it may not be stable as its a PyTorch3D experiment feature.

To avoid unexpected artifact, the object being textured should avoid flipped face normals and overlapping UV, and keep the number of triangle faces within around 40,000. You can try [Blender](https://www.blender.org/) for manual mesh cleaning and processing, or its python scripting for automation.

You can try out the method with the following pre-processed meshes and configs:
- [Flying Car](data/flying_car/config.yaml)
- [Monster](data/monster/config.yaml)
- [Cammy](data/monster/cammy.yaml)

## Inference
```bash
python main.py --config {your config}.yaml
```
Refer to [config.py](src/configs.py) for the list of arguments and settings you can adjust. You can change these settings by including them in a `.yaml` config file or passing the related arguments in command line; values specified in command line will overwrite those in config files.

When no output path is specified, the generated result will be placed in the same folder as the config file by default.

## Reference
[Pytorch](https://pytorch.org/) & [Diffusers](https://github.com/huggingface/diffusers) implemetation base on the paper:

- **[Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/pdf/2311.12891)**

- **[SyncTweedies: A General Generative Framework Based on Synchronized Diffusions](https://arxiv.org/abs/2403.14370)**

- **[FlexiTex: Enhancing Texture Generation via Visual Guidance](https://arxiv.org/abs/2409.12431)**