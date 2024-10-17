import json
import os
from importlib.metadata import version
from typing import Optional, Tuple, TypedDict

import gradio as gr
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

from tts_webui.bark.npz_tools import save_npz_musicgen
from tts_webui.bark.parse_or_set_seed import parse_or_set_seed
from tts_webui.history_tab.save_to_favorites import save_to_favorites
from tts_webui.musicgen.audio_array_to_sha256 import audio_array_to_sha256
from tts_webui.musicgen.setup_seed_ui_musicgen import setup_seed_ui_musicgen
from tts_webui.utils.create_base_filename import create_base_filename
from tts_webui.utils.date import get_date_string
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.save_waveform_plot import middleware_save_waveform_plot
from tts_webui.utils.set_seed import set_seed


def extension__tts_generation_webui():
    generation_tab_musicgen()
    return {
        "package_name": "extension_audiocraft_mac",
        "name": "MusicGen (Mac)",
        "version": "0.0.7",
        "requirements": "git+https://github.com/rsxdalv/extension_audiocraft_mac@main",
        "description": "MusicGen allows generating music from text",
        "extension_type": "interface",
        "extension_class": "audio-music-generation",
        "author": "rsxdalv",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/rsxdalv/extension_audiocraft_mac",
        "extension_website": "https://github.com/rsxdalv/extension_audiocraft_mac",
        "extension_platform_version": "0.0.1",
    }


AUDIOCRAFT_VERSION = version("audiocraft_apple_silicon")


class MusicGenGeneration(TypedDict):
    model: str
    text: str
    melody: Optional[Tuple[int, np.ndarray]]
    duration: float
    topk: int
    topp: float
    temperature: float
    cfg_coef: float
    seed: int
    use_multi_band_diffusion: bool


def melody_to_sha256(melody: Optional[Tuple[int, np.ndarray]]) -> Optional[str]:
    if melody is None:
        return None
    sr, audio_array = melody
    return audio_array_to_sha256(audio_array)


def generate_and_save_metadata(
    prompt: str,
    date: str,
    filename_json: str,
    params: MusicGenGeneration,
    audio_array: np.ndarray,
):
    metadata = {
        "_version": "0.0.1",
        "_hash_version": "0.0.3",
        "_type": "musicgen",
        "_audiocraft_version": AUDIOCRAFT_VERSION,
        "models": {},
        "text": prompt,
        "hash": audio_array_to_sha256(audio_array),
        "date": date,
        **params,
        "seed": str(params["seed"]),
        "melody": melody_to_sha256(params.get("melody", None)),
    }
    with open(filename_json, "w") as outfile:
        json.dump(metadata, outfile, indent=2)

    return metadata


def save_generation(
    audio_array: np.ndarray,
    SAMPLE_RATE: int,
    params: MusicGenGeneration,
    tokens: Optional[torch.Tensor] = None,
):
    prompt = params["text"]
    date = get_date_string()
    title = prompt[:20].replace(" ", "_")
    base_filename = create_base_filename(title, "outputs", model="musicgen", date=date)

    def get_filenames(base_filename):
        filename = f"{base_filename}.wav"
        filename_png = f"{base_filename}.png"
        filename_json = f"{base_filename}.json"
        filename_npz = f"{base_filename}.npz"
        return filename, filename_png, filename_json, filename_npz

    filename, filename_png, filename_json, filename_npz = get_filenames(base_filename)
    stereo = audio_array.shape[0] == 2
    if stereo:
        audio_array = np.transpose(audio_array)
    write_wav(filename, SAMPLE_RATE, audio_array)
    plot = middleware_save_waveform_plot(audio_array, filename_png)

    metadata = generate_and_save_metadata(
        prompt=prompt,
        date=date,
        filename_json=filename_json,
        params=params,
        audio_array=audio_array,
    )
    if tokens is not None:
        save_npz_musicgen(filename_npz, tokens, metadata)

    return filename, plot, metadata


@manage_model_state("musicgen_audiogen_apple_silicon")
def load_model(version):
    from audiocraft_apple_silicon.models.musicgen import MusicGen

    return MusicGen.get_pretrained(version)


def log_generation_musicgen(
    params: MusicGenGeneration,
):
    print("Generating: '''", params["text"], "'''")
    print("Parameters:")
    for key, value in params.items():
        print(key, ":", value)


def generate(params: MusicGenGeneration, melody_in: Optional[Tuple[int, np.ndarray]]):
    model = params["model"]
    text = params["text"]
    # due to JSON serialization limitations
    params["melody"] = None if "melody" not in model else melody_in
    melody = params["melody"]

    MODEL = load_model(model)

    MODEL.set_generation_params(
        use_sampling=True,
        top_k=params["topk"],
        top_p=params["topp"],
        temperature=params["temperature"],
        cfg_coef=params["cfg_coef"],
        duration=params["duration"],
    )

    tokens = None

    import time

    start = time.time()

    params["seed"] = parse_or_set_seed(params["seed"], 0)
    log_generation_musicgen(params)
    if "melody" in model and melody is not None:
        sr, melody = (
            melody[0],
            torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0),
        )
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., : int(sr * MODEL.lm.cfg.dataset.segment_duration)]  # type: ignore
        output = MODEL.generate_with_chroma(
            descriptions=[text],
            melody_wavs=melody,
            melody_sample_rate=sr,
            progress=True,
        )
    else:
        output = MODEL.generate(
            descriptions=[text],
            progress=True,
        )
    set_seed(-1)

    elapsed = time.time() - start
    print("Generated in", "{:.3f}".format(elapsed), "seconds")

    output = output.detach().cpu().numpy().squeeze()

    filename, plot, _metadata = save_generation(
        audio_array=output,
        SAMPLE_RATE=MODEL.sample_rate,
        params=params,
        tokens=tokens,
    )

    return [
        (MODEL.sample_rate, output.transpose()),
        os.path.dirname(filename),
        plot,
        params["seed"],
        _metadata,
    ]


def generation_tab_musicgen():
    gr.Markdown(f"""Audiocraft version: {AUDIOCRAFT_VERSION}""")
    with gr.Row(equal_height=False):
        with gr.Column():
            text = gr.Textbox(label="Prompt", lines=3, placeholder="Enter text here...")
            model = gr.Radio(
                [
                    "small",
                    "medium",
                    "large",
                    "melody",
                ],
                label="Model",
                value="small",
            )
            melody = gr.Audio(
                sources="upload",
                type="numpy",
                label="Melody (optional)",
                elem_classes="tts-audio",
            )
            submit = gr.Button("Generate", variant="primary")
        with gr.Column():
            duration = gr.Slider(
                minimum=1,
                maximum=360,
                value=10,
                label="Duration",
            )
            with gr.Row():
                topk = gr.Number(label="Top-k", value=250, interactive=True)
                topp = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.0,
                    label="Top-p",
                    step=0.05,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=1.0,
                    label="Temperature",
                    step=0.05,
                )
                cfg_coef = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=3.0,
                    label="Classifier Free Guidance",
                    step=0.1,
                )
            use_multi_band_diffusion = gr.Checkbox(
                label="(Not supported in this old version) Use Multi-Band Diffusion (High VRAM Usage)",
                value=False,
                interactive=False,
            )
            seed, set_old_seed_button, _ = setup_seed_ui_musicgen()

            unload_model_button("musicgen_audiogen_apple_silicon")

    with gr.Column():
        output = gr.Audio(
            label="Generated Music",
            type="numpy",
            interactive=False,
            elem_classes="tts-audio",
        )
        image = gr.Image(label="Waveform", elem_classes="tts-image")  # type: ignore
        with gr.Row():
            history_bundle_name_data = gr.Textbox(visible=False)
            save_button = gr.Button("Save to favorites", visible=True)
            melody_button = gr.Button("Use as melody", visible=True)
        save_button.click(
            fn=save_to_favorites,
            inputs=[history_bundle_name_data],
            outputs=[save_button],
        )

        melody_button.click(
            fn=lambda melody_in: melody_in,
            inputs=[output],
            outputs=[melody],
        )

    inputs = [
        text,
        melody,
        model,
        duration,
        topk,
        topp,
        temperature,
        cfg_coef,
        seed,
        use_multi_band_diffusion,
    ]

    def update_json(
        text,
        _melody,
        model,
        duration,
        topk,
        topp,
        temperature,
        cfg_coef,
        seed,
        use_multi_band_diffusion,
    ):
        return {
            "text": text,
            "melody": "exists" if _melody else "None",  # due to JSON limits
            "model": model,
            "duration": float(duration),
            "topk": int(topk),
            "topp": float(topp),
            "temperature": float(temperature),
            "cfg_coef": float(cfg_coef),
            "seed": int(seed),
            "use_multi_band_diffusion": bool(use_multi_band_diffusion),
        }

    seed_cache = gr.State()  # type: ignore
    result_json = gr.JSON(
        visible=False,
    )

    set_old_seed_button.click(
        fn=lambda x: gr.Number(value=x),
        inputs=seed_cache,
        outputs=seed,
    )

    submit.click(
        inputs=inputs,
        fn=lambda text,
        melody,
        model,
        duration,
        topk,
        topp,
        temperature,
        cfg_coef,
        seed,
        use_multi_band_diffusion: generate(
            params=update_json(
                text,
                melody,
                model,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                seed,
                use_multi_band_diffusion,
            ),  # type: ignore
            melody_in=melody,
        ),
        outputs=[output, history_bundle_name_data, image, seed_cache, result_json],
        api_name="musicgen_mac",
    )


if __name__ == "__main__":
    if "demo" in locals():
        demo.close()  # type: ignore
    with gr.Blocks() as demo:
        generation_tab_musicgen()

    demo.launch()
