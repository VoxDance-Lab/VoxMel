"""User-facing synthesis API using an existing VoxMel checkpoint."""
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torchaudio

from voxcore.train_utils.device_funcs import resolve_device
from voxcore.train_utils.set_all_random_seed import set_all_random_seed
from voxcore.utils.file_utils import load_wav


class VoxMel:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.sample_rate = model.sample_rate
        self._mel_transform = None
        self._prompt_cache = OrderedDict()
        self._prompt_cache_size = 4

    @classmethod
    def from_pretrained(cls, model_dir, *, checkpoint="model_L_final.pt",
                        tokenizer_path=None, vocos_path=None, device="auto", **overrides):
        """Load a local config.yaml and checkpoint; no automatic weight download."""
        from omegaconf import OmegaConf
        from voxcore.auto.auto_model import AutoModel
        from voxcore.tokenizer.qwen_tokenizer import get_qwen_tokenizer

        root = Path(model_dir).expanduser().resolve()
        for path in (root / "config.yaml", root / checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = OmegaConf.load(root / "config.yaml")
        OmegaConf.update(config, "model_conf.load_checkpoint", False, force_add=True)
        kwargs = OmegaConf.to_container(config, resolve=True)
        kwargs.update(overrides)
        kwargs.update(init_param=str(root / checkpoint), device=str(resolve_device(device)))
        if tokenizer_path is not None:
            kwargs["qwen_pretrain_path"] = str(tokenizer_path)
        if vocos_path is not None:
            kwargs["vocos_path"] = str(vocos_path)
            kwargs["vocoder_path"] = str(vocos_path)
        token_path = kwargs.get("qwen_pretrain_path")
        if not token_path:
            raise ValueError("tokenizer_path or qwen_pretrain_path in config is required")
        instance = cls(AutoModel(**kwargs).model, get_qwen_tokenizer(token_path, True), kwargs["device"])
        if instance.model._load_vocoder(instance.device) is None:
            raise RuntimeError("Vocos could not be loaded; check vocos_path and its weights")
        return instance

    @torch.inference_mode()
    def generate(self, text, *, prompt_wav_path=None, prompt_text=None, seed=None,
                 inference_timesteps=None, up_num_steps=None,
                 cfg_text_strength=None, cfg_spk_strength=None,
                 min_seconds=0.3, max_seconds=30.0):
        """Return a mono float32 NumPy waveform at self.sample_rate."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must be supplied together")
        if prompt_text is not None and not prompt_text.strip():
            raise ValueError("prompt_text must be non-empty")
        if not (math.isfinite(min_seconds) and math.isfinite(max_seconds)
                and 0 <= min_seconds < max_seconds):
            raise ValueError("require finite 0 <= min_seconds < max_seconds")
        for value in (inference_timesteps, up_num_steps):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError("sampling steps must be positive integers")
        if seed is not None:
            set_all_random_seed(seed)
        fps = self.sample_rate / self.model.hop_length / 2 ** self.model.num_time_splits
        options = dict(min_frames=max(1, math.ceil(min_seconds * fps)),
                       max_frames=max(1, math.ceil(max_seconds * fps)),
                       num_steps=inference_timesteps, up_num_steps=up_num_steps,
                       cfg_text_strength=cfg_text_strength,
                       cfg_spk_strength=cfg_spk_strength)
        def tokens(value):
            return torch.tensor([self.tokenizer.encode(value)], dtype=torch.int64, device=self.device)
        vocoder = self.model._load_vocoder(self.device)
        if vocoder is None:
            raise RuntimeError("Vocos could not be loaded; check vocos_path and its weights")
        if prompt_wav_path is None:
            target = tokens(text)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(torch.device(self.device).type == "cuda"
                                         and self.model.inference_dtype == "bf16")):
                mel = self.model.inference(target, torch.tensor([target.size(1)], device=self.device), **options)
        else:
            prompt_path = Path(prompt_wav_path).expanduser().resolve()
            stat = prompt_path.stat()
            key = (str(prompt_path), stat.st_size, stat.st_mtime_ns, prompt_text,
                   self.sample_rate, self.model.hop_length, self.model.n_mels,
                   self.model.num_time_splits)
            cached = self._prompt_cache.get(key)
            if cached is None:
                wav = load_wav(str(prompt_path), self.sample_rate)
                if wav.numel() == 0 or not torch.isfinite(wav).all():
                    raise ValueError("prompt audio must be non-empty and finite")
                if self._mel_transform is None:
                    self._mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate, n_fft=1024, win_length=1024,
                        hop_length=self.model.hop_length, n_mels=self.model.n_mels,
                        power=1, center=True, normalized=False, norm=None)
                if wav.size(-1) <= 512:
                    raise ValueError("prompt audio is too short for mel analysis")
                prompt = self._mel_transform(wav).transpose(1, 2).clamp_min(1e-5).log()
                grid = 2 ** self.model.num_time_splits
                if prompt.size(1) < grid:
                    raise ValueError("prompt audio is shorter than one coarse frame")
                prompt = prompt[:, :prompt.size(1) // grid * grid]
                cached = (prompt, self.tokenizer.encode(prompt_text))
                self._prompt_cache[key] = cached
                self._prompt_cache.move_to_end(key)
                if len(self._prompt_cache) > self._prompt_cache_size:
                    self._prompt_cache.popitem(last=False)
            else:
                self._prompt_cache.move_to_end(key)
            prompt, prompt_tokens = cached
            prompt = prompt.to(self.device)
            prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.int64, device=self.device)
            target_text = text.lstrip()
            if target_text and not ("一" <= target_text[0] <= "鿿"):
                text = " " + text
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(torch.device(self.device).type == "cuda"
                                         and self.model.inference_dtype == "bf16")):
                mel = self.model.inference_with_prompt(prompt, prompt_tokens, tokens(text), **options)
        return vocoder(mel.float()).reshape(-1).float().cpu().numpy()
