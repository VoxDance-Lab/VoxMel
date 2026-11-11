"""Command-line synthesis with a local VoxMel model."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Directory containing config.yaml and checkpoint")
    parser.add_argument("--checkpoint", default="model_L_final.pt")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--vocos-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio")
    parser.add_argument("--prompt-text")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--up-steps", type=int)
    parser.add_argument("--cfg-text-strength", type=float)
    parser.add_argument("--cfg-spk-strength", type=float)
    parser.add_argument("--min-seconds", type=float, default=0.3)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--output", default="output.wav")
    args = parser.parse_args()
    if (args.prompt_audio is None) != (args.prompt_text is None):
        parser.error("--prompt-audio and --prompt-text must be supplied together")
    import soundfile as sf
    from voxcore.inference import VoxMel
    model = VoxMel.from_pretrained(
        args.model, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer_path,
        vocos_path=args.vocos_path, device=args.device)
    wav = model.generate(
        args.text, prompt_wav_path=args.prompt_audio, prompt_text=args.prompt_text,
        seed=args.seed, inference_timesteps=args.timesteps, up_num_steps=args.up_steps,
        cfg_text_strength=args.cfg_text_strength,
        cfg_spk_strength=args.cfg_spk_strength, min_seconds=args.min_seconds,
        max_seconds=args.max_seconds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), wav, model.sample_rate)
    print(f"Saved: {output} ({model.sample_rate} Hz)")


if __name__ == "__main__":
    main()
