import logging
import os
from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import DictConfig

from voxcore.train_utils.device_funcs import resolve_device


def read_lst(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                items.append((parts[0], parts[1], parts[2], parts[3]))
    return items


@hydra.main(config_name=None, version_base=None)
def main(kwargs: DictConfig):
    from voxcore import VoxMel

    device = resolve_device(kwargs.get("device", "auto"))
    kwargs["device"] = str(device)
    testset = kwargs["input"]
    langs = str(kwargs.get("langs", "zh,en")).split(",")
    lst_name = str(kwargs.get("lst_name", "meta.lst"))
    output_dir = kwargs["output_dir"]
    max_items = int(kwargs.get("max_items", 0))

    checkpoint = Path(kwargs["init_param"])
    model = VoxMel.from_pretrained(
        checkpoint.parent,
        checkpoint=checkpoint.name,
        tokenizer_path=kwargs["qwen_pretrain_path"],
        vocos_path=kwargs.get("vocos_path"),
        device=device,
        cfg_batch=True,
    )

    def _opt(name, cast):
        v = kwargs.get(name, None)
        return None if v is None else cast(v)

    gen_kwargs = dict(
        up_num_steps=_opt("up_num_steps", int),
        inference_timesteps=_opt("coarse_steps", int),
        min_seconds=float(kwargs.get("min_seconds", 0.3)),
        max_seconds=float(kwargs.get("max_seconds", 30.0)),
        cfg_text_strength=_opt("cfg_text_strength", float),
        cfg_spk_strength=_opt("cfg_spk_strength", float),
    )

    num_shards = int(kwargs.get("num_shards", 1))
    shard_id = int(kwargs.get("shard_id", 0))

    for lang in langs:
        lang_dir = os.path.join(testset, lang)
        items = read_lst(os.path.join(lang_dir, lst_name))
        if num_shards > 1:
            items = items[shard_id::num_shards]
        if max_items > 0:
            items = items[:max_items]
        os.makedirs(os.path.join(output_dir, lang), exist_ok=True)
        logging.info(f"{lang}/{lst_name}: {len(items)} items")

        for key, prompt_text, prompt_wav, target_text in items:
            wav_path = os.path.join(output_dir, lang, f"{key}.wav")
            if os.path.exists(wav_path):
                continue
            prompt_path = os.path.join(lang_dir, prompt_wav)
            if not os.path.exists(prompt_path):
                logging.warning(f"skip {key}: missing {prompt_path}")
                continue

            wav = model.generate(
                target_text,
                prompt_wav_path=prompt_path,
                prompt_text=prompt_text,
                **gen_kwargs,
            )
            torchaudio.save(wav_path, torch.from_numpy(wav).unsqueeze(0), model.sample_rate)
            logging.info(f"done {lang}/{key}")


if __name__ == "__main__":
    main()
