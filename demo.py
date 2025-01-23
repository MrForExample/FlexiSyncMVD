from typing import Optional
import gradio as gr
from gradio.themes.utils import fonts
import httpx
from tempfile import NamedTemporaryFile

from sortium.src.custom_types import InputConfig, TextureOutput


def configure_sortium_theme() -> gr.themes.Base | str:
    sortium_theme: gr.themes.Base = gr.themes.Soft(
        font=[
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "sans-serif",
        ]
    )
    # .set(body_background_fill="#000000", block_background_fill="#000000")
    return sortium_theme


def run_pipeline(
    mesh: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cond_type: str,
    seed: int,
    mesh_scale: int,
    latent_view_size: int,
    latent_tex_size: int,
    rgb_view_size: int,
    rgb_tex_size: int,
    conditioning_scale: float,
    conditioning_scale_end: float,
    base_url: str,
) -> str:
    if not base_url:
        raise ValueError("API URL cannot be empty")

    api_client = httpx.Client(base_url=base_url, timeout=360)

    if not prompt:
        raise gr.Error(message="Prompt cannot be empty", title="Prompt Error")
    if not mesh:
        raise gr.Error(message="Mesh cannot be empty", title="Mesh Error")

    # Build input config
    input_config = InputConfig(
        t2i_model="SDXL",
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cond_type=cond_type,
        seed=seed,
        mesh_scale=mesh_scale,
        latent_view_size=latent_view_size,
        latent_tex_size=latent_tex_size,
        rgb_view_size=rgb_view_size,
        rgb_tex_size=rgb_tex_size,
        conditioning_scale=conditioning_scale,
        conditioning_scale_end=conditioning_scale_end,
    )

    headers = {
        "accept": "application/json",
    }

    data = {
        "config": input_config.model_dump_json(exclude_none=True, exclude_unset=True),
    }

    files = [
        ("file", open(mesh, "rb")),
    ]

    try:
        response = api_client.post(
            "/texture/upload", headers=headers, data=data, files=files
        )
        response.raise_for_status()

        result: TextureOutput = TextureOutput(**response.json())

        # Convert from HttpURL to string
        generated_mesh = str(result.generated_mesh)
        print(f"Generated mesh URL: {generated_mesh}")

        # Extract suffix from result.generated_mesh url
        suffix = generated_mesh.split(".")[-1]

        # Save generated mesh to a named temporary file
        with NamedTemporaryFile(suffix=f".{suffix}", delete=False) as temp_file:
            temp_file.write(api_client.get(generated_mesh).content)
            print(f"Generated mesh saved to {temp_file.name}")
            return temp_file.name

    except Exception as e:
        raise gr.Error(message=f"Failed to run pipeline: {e}", title="Pipeline Error")


def build_gradio_ui(
    theme: Optional[gr.themes.Base | str] = "soft", api_url: Optional[str] = None
) -> gr.Blocks:
    """
    Build the Gradio UI for the Sortium Texture Pipeline

    Args:

    - theme: Gradio theme to use for the UI
    - api_url: URL of the Sortium API

    Returns:

    - Gradio Blocks object
    """

    if not api_url:
        raise ValueError("API URL cannot be empty")

    if type(theme) is str and theme.lower() == "sortium":
        theme = configure_sortium_theme()

    with gr.Blocks(
        theme=theme, fill_width=False, title="Sortium Texture Pipeline"
    ) as demo:
        gr.Markdown(
            """
            # __SORTIUM__ Texture Pipeline

            Demo application to showcase the State of the Art Sortium Texture Generation System for industry standard 3d mesh.
            """
        )
        base_url = gr.Label(label="API URL", value=api_url, visible=False)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                mesh = gr.Model3D(label="Mesh")
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt")
                steps = gr.Slider(
                    minimum=1, maximum=50, step=1.0, label="Steps", value=30
                )
                seed = gr.Slider(
                    minimum=1,
                    maximum=999999,
                    step=1.0,
                    label="Seed",
                    randomize=True,
                )
                # with gr.Row(equal_height=True):
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
                        )
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            cond_type = gr.Radio(
                                label="Conditioning Type",
                                choices=["depth"],
                                value="depth",
                            )
                            conditioning_scale = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=0.3,
                                label="Conditioning Scale",
                                info="The scale of the conditioning image.",
                            )
                            conditioning_scale_end = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=0.7,
                                label="Conditioning Scale End",
                                info="Gradually increasing conditioning scale for better geometry alignment near the end",
                            )
                            mesh_scale = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=1,
                                label="Mesh Scale",
                                info="Set above 1 to enlarge object in camera views",
                            )
                        with gr.Column():
                            latent_view_size = gr.Radio(
                                label="Latent View Size",
                                choices=["96", "128"],
                                value="128",
                                info="Larger resolution, less aliasing in latent images; quality may degrade if much larger trained resolution of networks",
                            )
                            latent_tex_size = gr.Radio(
                                label="Latent Tex Size",
                                choices=["512", "768", "1024"],
                                value="768",
                                info="Originally 1536 in paper, use lower resolution save VRAM and runtime",
                            )
                            rgb_tex_size = gr.Radio(
                                label="RGB Texture Size",
                                choices=["768", "1024", "1280"],
                                value="768",
                            )
                            rgb_view_size = gr.Radio(
                                label="RGB View Size",
                                choices=["512", "768", "1024"],
                                value="1024",
                            )

        with gr.Row(equal_height=True):
            generate_btn = gr.Button("Generate", variant="primary")
            gr.ClearButton(value="Clear", inputs=[mesh, prompt, negative_prompt])
            generate_btn.click(
                fn=run_pipeline,
                inputs=[
                    mesh,
                    prompt,
                    negative_prompt,
                    steps,
                    cond_type,
                    seed,
                    mesh_scale,
                    latent_view_size,
                    latent_tex_size,
                    rgb_tex_size,
                    rgb_view_size,
                    conditioning_scale,
                    conditioning_scale_end,
                    base_url,
                ],
                outputs=[mesh],
                api_name="texture",
                queue=True,
                concurrency_limit=1,
            )

    return demo


if __name__ == "__main__":
    demo: gr.Blocks = build_gradio_ui(
        theme=configure_sortium_theme(), api_url="http://localhost:8000"
    )
    demo.queue()
    demo.launch(
        show_api=True,
    )
