"""Local Gradio demo for transcript-conditioned VoxMel voice cloning."""
import argparse
import base64
import json
import math
from pathlib import Path
import threading
import time

import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parent


class VoxMelDemo:
    """Serialize generation because inference changes the process RNG state."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def generate(self, text, prompt_audio, prompt_text, seed, steps, up_steps,
                 cfg_text_strength, cfg_spk_strength,
                 min_seconds, max_seconds):
        text = (text or "").strip()
        transcript = (prompt_text or "").strip()
        if not text:
            raise ValueError("Enter the text to synthesize.")
        if not prompt_audio or not transcript:
            raise ValueError("Voice cloning requires both reference audio and its exact transcript.")
        actual_seed = 0
        for value in (steps, up_steps):
            if value is None or not math.isfinite(value) or int(value) != value or value < 1:
                raise ValueError("Sampling steps must be positive integers.")
        for value in (cfg_text_strength, cfg_spk_strength):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Guidance must be a finite, non-negative number, or left blank.")
        if (min_seconds is None or max_seconds is None
                or not math.isfinite(min_seconds) or not math.isfinite(max_seconds)
                or not 0 <= min_seconds < max_seconds):
            raise ValueError("Duration must satisfy 0 <= minimum < maximum.")
        with self.lock:
            start = time.perf_counter()
            wav = self.model.generate(
                text, prompt_wav_path=prompt_audio or None,
                prompt_text=transcript if prompt_audio else None,
                seed=actual_seed, inference_timesteps=int(steps),
                up_num_steps=int(up_steps),
                cfg_text_strength=cfg_text_strength, cfg_spk_strength=cfg_spk_strength,
                min_seconds=min_seconds, max_seconds=max_seconds)
            elapsed = time.perf_counter() - start
        duration = len(wav) / self.model.sample_rate
        output_dir = PROJECT_DIR / "examples" / "voxmel" / "gen_tmp"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"voxmel_{time.time_ns()}_{actual_seed}.wav"
        sf.write(output_path, wav, self.model.sample_rate)
        output_path = str(output_path)
        return (audio_player(output_path), actual_seed,
                f"Generated {duration:.2f}s of audio in {elapsed:.2f}s · "
                f"{self.model.sample_rate} Hz · Seed {actual_seed}")


def audio_player(path):
    if not path:
        return ""
    encoded_audio = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (f'<audio controls preload="auto" style="width: 100%" '
            f'src="data:audio/wav;base64,{encoded_audio}"></audio>')


def create_demo_interface(demo):
    import gradio as gr

    example_dir = Path(__file__).resolve().parent / "assets" / "voice_cloning"
    with (example_dir / "examples.json").open(encoding="utf-8") as f:
        examples = [[item["target_text"], str(example_dir / item["prompt_audio"]),
                     item["prompt_text"]] for item in json.load(f)]

    def generate(*args):
        try:
            return demo.generate(*args)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="VoxMel Voice Cloning", analytics_enabled=False) as interface:
        gr.Markdown("# VoxMel Voice Cloning\n"
                    "Zero-shot voice cloning with a reference audio and its exact transcript.\n\n"
                    "**零样本声音克隆**：请选择本地示例，或提供参考音频及其准确转录。")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Text to synthesize / 合成文本", lines=5,
                                  placeholder="输入需要使用参考声音合成的文本。")
                reference = gr.Audio(label="Reference audio / 参考音频", type="filepath",
                                     sources=["upload", "microphone"], visible=False)
                reference_preview = gr.HTML(label="Reference audio / 参考音频")
                reference.change(audio_player, inputs=[reference], outputs=[reference_preview])
                transcript = gr.Textbox(label="Exact reference transcript / 参考音频原文", lines=2)
                gr.Examples(
                    examples=examples,
                    inputs=[text, reference, transcript],
                    label="Voice-cloning examples / 声音克隆示例（点击选择）",
                )
            with gr.Column(scale=2):
                gr.Markdown("### Generated speech / 生成语音")
                audio = gr.HTML()
                status = gr.Textbox(label="Generation status / 生成状态", interactive=False)
                seed_used = gr.Number(label="Seed used / 本次随机种子", precision=0,
                                      interactive=False)
                with gr.Accordion("Generation settings / 生成设置", open=False):
                    seed = gr.Number(label="Seed (fixed) / 固定随机种子", value=0,
                                     precision=0, interactive=False)
                    steps = gr.Slider(1, 64, value=32, step=1, label="Coarse-Mel solver steps")
                    up_steps = gr.Slider(1, 64, value=32, step=1, label="Residual solver steps")
                    with gr.Row():
                        minimum = gr.Number(label="Minimum seconds", value=.3)
                        maximum = gr.Number(label="Maximum seconds", value=30)
                    gr.Markdown("Text and speaker guidance are set independently.\n\n"
                                "文本与说话人 CFG 分别设置。")
                    text_cfg = gr.Number(label="Text CFG / 文本 CFG", value=3.5)
                    spk_cfg = gr.Number(label="Speaker CFG / 说话人 CFG", value=3.0)
                run = gr.Button("Clone voice / 克隆生成", variant="primary")
        run.click(generate,
                  inputs=[text, reference, transcript, seed, steps, up_steps,
                          text_cfg, spk_cfg, minimum, maximum],
                  outputs=[audio, seed_used, status], concurrency_limit=1,
                  api_name="generate")
    return interface


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local experiment directory")
    parser.add_argument("--checkpoint", default="model_L_final.pt")
    parser.add_argument("--tokenizer-path", default="examples/voxmel/qwen_tokenizer")
    parser.add_argument("--vocos-path", help="Local Vocos directory; otherwise use the model config")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    try:
        import gradio  # noqa: F401
    except ImportError:
        parser.error('Install the demo dependencies first: pip install -e ".[web]"')
    from voxcore import VoxMel
    model = VoxMel.from_pretrained(
        args.model, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer_path,
        vocos_path=args.vocos_path, device=args.device, cfg_batch=True)
    interface = create_demo_interface(VoxMelDemo(model))
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, show_error=True,
        allowed_paths=[str(PROJECT_DIR)])


if __name__ == "__main__":
    main()
