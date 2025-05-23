import configargparse

def parse_config():
    parser = configargparse.ArgumentParser(
                        prog='Multi-View Diffusion',
                        description='Generate texture given mesh and texture prompt',
                        epilog='Refer to https://arxiv.org/abs/2311.12891 for more details')
    # File Config
    parser.add_argument('--config', type=str, required=True, is_config_file=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--mesh_config_relative', action='store_true', help="Search mesh file relative to the config path instead of current working directory")
    parser.add_argument('--output', type=str, default=None, help="If not provided, use the parent directory of config file for output")
    parser.add_argument('--prefix', type=str, default='MVD')
    parser.add_argument('--use_mesh_name', action='store_true')
    parser.add_argument('--timeformat', type=str, default='%d%b%Y-%H%M%S', help='Setting to None will not use time string in output directory')
    # Diffusion Config
    parser.add_argument('--t2i_model', type=str, default='SD1.5', help='Support SD1.5 and SDXL')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='If provided, will use custom checkpoint for the model')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=15.5, help='Recommend above 12 to avoid blurriness')
    parser.add_argument('--seed', type=int, default=0)
    # ControlNet Config
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--conditioning_scale', type=float, default=0.7)
    parser.add_argument('--conditioning_scale_end', type=float, default=0.9, help='Gradually increasing conditioning scale for better geometry alignment near the end')
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=0.99)
    parser.add_argument('--guidance_rescale', type=float, default=0.0, help='Not tested')
    # IP Adapter Config
    parser.add_argument('--ip_adapter_image', type=str, default=None, help='Use reference image to guide the diffusion process')
    parser.add_argument('--ip_adapter_scale', type=float, default=0.5, help='The strength of the reference image in the diffusion process')
    # Multi-View Config
    parser.add_argument('--latent_view_size', type=int, default=96, help='Larger resolution, less aliasing in latent images; quality may degrade if much larger trained resolution of networks')
    parser.add_argument('--latent_tex_size', type=int, default=768, help='Originally 1536 in paper, use lower resolution save VRAM and runtime')
    parser.add_argument('--rgb_view_size', type=int, default=768)
    parser.add_argument('--rgb_tex_size', type=int, default=1024)
    parser.add_argument('--max_batch_size', type=int, default=48)
    parser.add_argument('--camera_azims', type=int, nargs="*", default=[-180, -135, -90, -45, 0, 45, 90, 135], help='Place the cameras at the listed azim angles')
    parser.add_argument('--no_top_cameras', action='store_true', help='Two cameras added to paint the top surface')
    parser.add_argument('--mvd_end', type=float, default=0.8, help='Time step to stop texture space aggregation')
    parser.add_argument('--mvd_exp_start', type=float, default=0.0, help='Initial exponent for weighted texture space aggregation, low value encourage consistency')
    parser.add_argument('--mvd_exp_end', type=float, default=6.0, help='End exponent for weighted texture space aggregation, high value encourage sharper results')
    parser.add_argument('--ref_attention_end', type=float, default=0.2, help='Lower->better quality; higher->better harmonization')
    parser.add_argument('--shuffle_bg_change', type=float, default=0.4, help='Use only black and white background after certain timestep')
    parser.add_argument('--shuffle_bg_end', type=float, default=0.8, help='Don\'t shuffle background after certain timestep. background color may bleed onto object')
    parser.add_argument('--mesh_scale', type=float, default=1.0, help='Set above 1 to enlarge object in camera views')
    parser.add_argument('--keep_mesh_uv', action='store_true', help='Don\'t use Xatlas to unwrap UV automatically')
    # Logging Config
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--view_fast_preview', action='store_true', help='Use color transformation matrix instead of decoder to log view images')
    parser.add_argument('--tex_fast_preview', action='store_true', help='Use color transformation matrix instead of decoder to log texture images')
    options = parser.parse_args()

    return options
