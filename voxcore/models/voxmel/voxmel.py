
import math

import torch
import torch.nn as nn

import logging
import os

from voxcore.register import tables
from voxcore.models.llm_tts.llm import Qwen2Encoder

def _timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

def _apply_cfg(v_cond, v_uncond, cfg_strength):
    return v_cond + cfg_strength * (v_cond - v_uncond)


def _apply_cfg_two_term(v_full, v_dropT, v_dropTS, s_text, s_spk):
    return v_dropTS + s_text * (v_full - v_dropT) + s_spk * (v_dropT - v_dropTS)


class FlowMatchingHead(nn.Module):

    def __init__(self, data_dim, cond_dim, hidden=512, depth=3, time_embed_dim=128):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self._time_hidden_cache = {}

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.in_proj = nn.Linear(data_dim, hidden)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                              nn.Linear(hidden, hidden))
            )
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, data_dim))

    def velocity(self, phi, t, cond, cond_hidden=None, time_hidden=None):
        if cond_hidden is None:
            cond_hidden = self.cond_proj(cond)
        if time_hidden is None:
            time_hidden = self.time_mlp(_timestep_embedding(t, self.time_embed_dim))
        h = self.in_proj(phi) + cond_hidden + time_hidden
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)

    @torch.no_grad()
    def sample(self, cond, num_steps=16, cond_uncond=None, cfg_strength=0.0,
               cond_dropT=None, cond_dropTS=None, s_text=0.0, s_spk=0.0):
        N = cond.size(0)
        x = torch.randn(N, self.data_dim, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / num_steps

        legacy = cond_uncond is not None and abs(cfg_strength) > 1e-6
        three = (cond_dropT is not None and cond_dropTS is not None
                 and (abs(s_text) > 1e-6 or abs(s_spk) > 1e-6))
        batch_size = N * (3 if three else 2 if legacy else 1)
        if three:
            cond = torch.cat([cond, cond_dropT, cond_dropTS], dim=0)
        elif legacy:
            cond = torch.cat([cond, cond_uncond], dim=0)
        cond_hidden = self.cond_proj(cond)
        versions = tuple(parameter._version for parameter in self.time_mlp.parameters())
        cache_key = (num_steps, batch_size, cond.device, cond.dtype, versions)
        time_hiddens = self._time_hidden_cache.get(cache_key)
        if time_hiddens is None:
            time_hiddens = [
                self.time_mlp(_timestep_embedding(
                    torch.full((batch_size,), i * dt, device=cond.device, dtype=cond.dtype),
                    self.time_embed_dim,
                ))
                for i in range(num_steps)
            ]
            if len(self._time_hidden_cache) >= 8:
                self._time_hidden_cache.clear()
            self._time_hidden_cache[cache_key] = time_hiddens

        for i in range(num_steps):
            ti = i * dt
            t = torch.full((N,), ti, device=cond.device, dtype=cond.dtype)
            time_hidden = time_hiddens[i]

            if three:
                x3 = torch.cat([x, x, x], dim=0)
                v3 = self.velocity(x3, t, cond, cond_hidden, time_hidden)
                v_full, v_dropT, v_dropTS = torch.chunk(v3, 3, dim=0)
                v = _apply_cfg_two_term(v_full, v_dropT, v_dropTS, s_text, s_spk)
            elif legacy:
                x2 = torch.cat([x, x], dim=0)
                v2 = self.velocity(x2, t, cond, cond_hidden, time_hidden)
                v_cond, v_uncond = torch.chunk(v2, 2, dim=0)
                v = _apply_cfg(v_cond, v_uncond, cfg_strength)
            else:
                v = self.velocity(x, t, cond, cond_hidden, time_hidden)
            x = x + dt * v
        return x

def sinusoidal_pos_emb(T, dim, device, dtype):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / half)
    args = pos * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(T, 1, device=device)], dim=-1)
    return emb.to(dtype)[None]

class _BidirAttention(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class UpsampleTier(nn.Module):

    def __init__(self, n_mels, dim, num_heads, n_layers, flow_hidden=512, flow_depth=3,
                 detail_norm=False, attn_window=-1):
        super().__init__()
        self.n_mels = n_mels
        self.dim = dim

        self.attn_window = int(attn_window)

        self.in_proj = nn.Linear(n_mels, dim)
        self.blocks = nn.ModuleList([_BidirAttention(dim, num_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.flow = FlowMatchingHead(n_mels, dim, hidden=flow_hidden, depth=flow_depth)

        self.detail_norm = detail_norm
        self.register_buffer("detail_mean", torch.zeros(n_mels))
        self.register_buffer("detail_var", torch.ones(n_mels))

    def _denorm_detail(self, d):
        if not self.detail_norm:
            return d
        return d * self.detail_var.clamp_min(1e-8).sqrt() + self.detail_mean

    def _window_attn_mask(self, T, device):
        if self.attn_window < 0:
            return None
        idx = torch.arange(T, device=device)
        return (idx[None, :] - idx[:, None]).abs() > self.attn_window

    def _context(self, x):
        B, T, M = x.shape
        h = self.in_proj(x) + sinusoidal_pos_emb(T, self.dim, x.device, x.dtype)
        mask = self._window_attn_mask(T, x.device)
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        return self.norm(h)

    @torch.no_grad()
    def sample(self, x_known, num_steps=16):
        B, T, M = x_known.shape
        cond = self._context(x_known).reshape(B * T, self.dim)
        y = self.flow.sample(cond, num_steps=num_steps)

        y = self._denorm_detail(y)
        return y.reshape(B, T, M)

def _even_odd(x, axis):
    if axis == "t":
        e, o = x[:, 0::2], x[:, 1::2]
        L = min(e.size(1), o.size(1))
        return e[:, :L], o[:, :L]
    elif axis == "f":
        e, o = x[..., 0::2], x[..., 1::2]
        L = min(e.size(-1), o.size(-1))
        return e[..., :L], o[..., :L]
    raise ValueError(f"axis must be 't' or 'f', got {axis!r}")

def _split(x, axis):
    e, o = _even_odd(x, axis)
    return (e + o) / 2, (e - o) / 2

def _interleave(e, o, axis):
    first, second = e + o, e - o
    if axis == "t":
        B, L, M = e.shape
        out = e.new_empty(B, 2 * L, M)
        out[:, 0::2] = first
        out[:, 1::2] = second
        return out
    elif axis == "f":
        B, T, L = e.shape
        out = e.new_empty(B, T, 2 * L)
        out[..., 0::2] = first
        out[..., 1::2] = second
        return out
    raise ValueError(f"axis must be 't' or 'f', got {axis!r}")

def _pyramid(x, split_axes):
    details_rev = []
    cur = x
    for axis in reversed(split_axes):
        e, o = _split(cur, axis)
        details_rev.append((axis, e, o))
        cur = e
    return cur, list(reversed(details_rev))

def _tier_dims(n_mels, split_axes):
    cur = n_mels
    dims_rev = []
    for axis in reversed(split_axes):
        if axis == "f":
            cur = cur // 2
        dims_rev.append(cur)
    return cur, list(reversed(dims_rev))

@tables.register("model_classes", "VoxMel")
class VoxMel(nn.Module):
    def __init__(
        self,
        n_mels: int,
        llm_input_size: int,
        llm_output_size: int,
        llm_config: str,
        dim: int = 512,
        num_heads: int = 8,
        n_layers=(6, 4, 4),
        split_axes=None,
        flow_hidden: int = 512,
        flow_depth: int = 3,
        num_sampling_steps: int = 16,
        vocos_path: str = None,
        vocoder_path: str = None,
        vocoder_weights_ckpt: str = None,
        sample_rate: int = 24000,
        hop_length: int = 256,
        cfg_text_strength: float = 0.0,
        cfg_spk_strength: float = 0.0,
        detail_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        if kwargs.get("patch_size", 1) != 1:
            raise ValueError("VoxMel supports only frame-wise generation.")
        self.n_mels = n_mels
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.dim = dim
        self.n_tiers = len(n_layers)
        self.num_sampling_steps = num_sampling_steps
        self.inference_dtype = kwargs.get("inference_dtype", "fp32")
        self.cfg_batch = kwargs.get("cfg_batch", True)
        self.cfg_text_strength = float(cfg_text_strength)
        self.cfg_spk_strength = float(cfg_spk_strength)



        self.sos_eos = 0
        self.task_id = 1
        self.special_embedding = nn.Embedding(50, llm_input_size)

        self.null_text = nn.Parameter(torch.zeros(1, llm_input_size))

        if split_axes is None:
            split_axes = ("t",) * (self.n_tiers - 1)
        split_axes = tuple(split_axes)
        assert len(split_axes) == self.n_tiers - 1, (
            f"split_axes needs {self.n_tiers - 1} entries, got {len(split_axes)}"
        )
        assert all(a in ("t", "f") for a in split_axes)
        self.split_axes = split_axes
        self.num_time_splits = sum(a == "t" for a in split_axes)
        self.num_freq_splits = sum(a == "f" for a in split_axes)

        coarse_M, up_dims = _tier_dims(n_mels, split_axes)
        assert n_mels % (2 ** self.num_freq_splits) == 0, (
            f"n_mels {n_mels} not divisible by 2**{self.num_freq_splits}"
        )
        self.coarse_M = coarse_M

        self.coarse_frame_proj = nn.Linear(coarse_M, llm_input_size)
        self.llm = Qwen2Encoder(llm_config, load_checkpoint=kwargs.get("load_checkpoint", True))

        self.coarse_flow = FlowMatchingHead(
            coarse_M, cond_dim=llm_output_size, hidden=flow_hidden, depth=flow_depth,
        )
        self.stop_token_head = nn.Linear(llm_output_size, 1, bias=False)


        self.detail_norm = bool(detail_norm)

        _taw = kwargs.get("tier_attn_window", None)
        if _taw is None:
            _tier_windows = [-1] * len(self.split_axes)
        elif isinstance(_taw, (int, float)):
            _tier_windows = [int(_taw)] * len(self.split_axes)
        else:
            _tier_windows = [int(_taw.get(a, -1)) for a in self.split_axes]
        self.tier_attn_window = _tier_windows
        self.up_tiers = nn.ModuleList(
            [UpsampleTier(d, dim, num_heads, nl, flow_hidden, flow_depth,
                          detail_norm=self.detail_norm,
                          attn_window=aw)
             for nl, d, aw in zip(
                 n_layers[1:], up_dims,
                 _tier_windows)]
        )
        if kwargs.get("compile_flow", False):
            self.coarse_flow.velocity = torch.compile(self.coarse_flow.velocity)
            for tier in self.up_tiers:
                tier.flow.velocity = torch.compile(tier.flow.velocity)

        self.vocoder_path = vocoder_path or vocos_path
        self.vocoder_weights_ckpt = vocoder_weights_ckpt
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self._vocoder = None

        self.register_buffer("mel_mean", torch.zeros(n_mels))
        self.register_buffer("mel_std", torch.ones(n_mels))

    def _normalize_mel(self, mel):
        return (mel - self.mel_mean) / self.mel_std

    def _denormalize_mel(self, mel):
        return mel * self.mel_std + self.mel_mean


    def _prompt_tier_grids(self, prompt_mel):
        p_coarse, details = _pyramid(prompt_mel, self.split_axes)
        p_known_list = [known for (_axis, known, _target) in details]
        return p_coarse, p_known_list


    @torch.no_grad()
    def inference(self, text_token, text_token_len, max_frames=1000,
                  min_frames=1, num_steps=None, return_stages=False,
                  cfg_text_strength=None, cfg_spk_strength=None,
                  up_num_steps=None):
        device = text_token.device
        steps = num_steps or self.num_sampling_steps
        up_steps = up_num_steps or steps


        s_text = (self.cfg_text_strength if cfg_text_strength is None
                  else cfg_text_strength)
        use_text_cfg = abs(s_text) > 1e-6

        text_emb = self.llm.model.model.embed_tokens(text_token)
        L = text_emb.size(1)
        sos = self.special_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task = self.special_embedding.weight[self.task_id].reshape(1, 1, -1)
        prime_cond = torch.cat([sos, text_emb, task], dim=1)
        null_txt = self.null_text.reshape(1, 1, -1).expand(1, L, -1)
        prime_uncond = torch.cat([sos, null_txt, task], dim=1)

        cache_c = cache_u = None
        frames = []
        cur_c, cur_u = prime_cond, prime_uncond
        seen = 0

        def _step(cur_c, cur_u, cache_c, cache_u, seen):
            total = seen + cur_c.size(1)
            masks = torch.ones((1, total), dtype=torch.bool, device=device)
            h_c, cache_c = self.llm.forward_one_step(cur_c, masks=masks, cache=cache_c)
            if use_text_cfg:
                h_u, cache_u = self.llm.forward_one_step(cur_u, masks=masks, cache=cache_u)
            else:
                h_u = h_c
            return h_c[:, -1], h_u[:, -1], cache_c, cache_u, total

        def _coarse_frame(last_c, last_u):
            return self.coarse_flow.sample(
                last_c, num_steps=steps,
                cond_dropT=(last_u if use_text_cfg else None),
                cond_dropTS=(last_u if use_text_cfg else None),
                s_text=s_text, s_spk=0.0)

        for t in range(max_frames):
            last_c, last_u, cache_c, cache_u, seen = _step(
                cur_c, cur_u, cache_c, cache_u, seen)
            stop = torch.sigmoid(self.stop_token_head(last_c)) > 0.5
            if bool(stop) and t > min_frames:
                break
            frame = _coarse_frame(last_c, last_u)
            frames.append(frame)
            nxt = self.coarse_frame_proj(frame).unsqueeze(1)
            cur_c = cur_u = nxt

        if len(frames) == 0:
            frame = _coarse_frame(last_c, last_u)
            frames.append(frame)

        x = torch.stack(frames, dim=1)

        stages = [("tier1_coarse", x)] if return_stages else None
        for k, axis in enumerate(self.split_axes):
            o = self.up_tiers[k].sample(x, num_steps=up_steps)
            x = _interleave(x, o, axis)
            if return_stages:
                stages.append((f"tier{k + 2}_{axis}", x))

        x = self._denormalize_mel(x)
        if return_stages:
            stages[-1] = (stages[-1][0], x)
            return x, stages
        return x


    @torch.no_grad()
    def inference_with_prompt(self, prompt_mel, prompt_text_token,
                              target_text_token, max_frames=1000, min_frames=1,
                              num_steps=None,
                              cfg_text_strength=None, cfg_spk_strength=None,
                              return_tiers=False,
                              up_num_steps=None):

        device = prompt_mel.device
        steps = num_steps or self.num_sampling_steps
        up_steps = up_num_steps or steps



        s_text = (self.cfg_text_strength if cfg_text_strength is None
                  else cfg_text_strength)
        s_spk = self.cfg_spk_strength if cfg_spk_strength is None else cfg_spk_strength
        use_text_cfg = abs(s_text) > 1e-6
        use_spk_cfg = abs(s_spk) > 1e-6

        factor = 2 ** self.num_time_splits
        rem = prompt_mel.size(1) % factor
        if factor > 1 and rem:
            prompt_mel = prompt_mel[:, :-rem, :]

        prompt_mel = self._normalize_mel(prompt_mel)
        p_coarse, p_known_list = self._prompt_tier_grids(prompt_mel)
        T0_p = p_coarse.size(1)
        assert T0_p >= 1, "prompt too short: < 1 coarse frame"

        text_token = torch.cat([prompt_text_token, target_text_token], dim=1)
        text_emb = self.llm.model.model.embed_tokens(text_token)
        L = text_emb.size(1)
        sos = self.special_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task = self.special_embedding.weight[self.task_id].reshape(1, 1, -1)
        p_frame_emb = self.coarse_frame_proj(p_coarse)
        prime_full = torch.cat([sos, text_emb, task, p_frame_emb], dim=1)

        null_txt = self.null_text.reshape(1, 1, -1).expand(1, L, -1)
        zero_prompt = torch.zeros_like(p_frame_emb)
        # text_first factorization: drop text first, then text+speaker.
        prime_p = torch.cat([sos, null_txt, task, p_frame_emb], dim=1)
        prime_b = torch.cat([sos, null_txt, task, zero_prompt], dim=1)
        use_p, use_b = (use_text_cfg or use_spk_cfg), use_spk_cfg

        cache_f = cache_p = cache_b = None
        cur_f, cur_p, cur_b = prime_full, prime_p, prime_b
        seen = 0
        frames = []

        def _coarse_frame(last_f, last_p, last_b):
            cond_dropT = last_p if use_p else None
            cond_dropTS = last_b if use_b else (last_p if use_p else None)
            return self.coarse_flow.sample(
                last_f, num_steps=steps, cond_dropT=cond_dropT,
                cond_dropTS=cond_dropTS, s_text=s_text, s_spk=s_spk)

        for tstep in range(max_frames):
            total = seen + cur_f.size(1)
            masks = torch.ones((1, total), dtype=torch.bool, device=device)
            if self.cfg_batch and use_p:
                active = [cur_f, cur_p] + ([cur_b] if use_b else [])
                caches = [cache_f, cache_p] + ([cache_b] if use_b else [])
                batch_size = len(active)
                cache = None
                if caches[0] is not None:
                    cache = tuple(
                        tuple(torch.cat(tensors, dim=0) for tensors in zip(*layers))
                        for layers in zip(*caches)
                    )
                hidden, cache = self.llm.forward_one_step(
                    torch.cat(active, dim=0), masks=masks.expand(batch_size, -1), cache=cache)
                hidden = hidden.chunk(batch_size, dim=0)
                caches = [
                    tuple(tuple(tensor[i:i + 1] for tensor in layer) for layer in cache)
                    for i in range(batch_size)
                ]
                h_f, cache_f = hidden[0], caches[0]
                h_p, cache_p = hidden[1], caches[1]
                if use_b:
                    h_b, cache_b = hidden[2], caches[2]
            else:
                h_f, cache_f = self.llm.forward_one_step(cur_f, masks=masks, cache=cache_f)
                if use_p:
                    h_p, cache_p = self.llm.forward_one_step(cur_p, masks=masks, cache=cache_p)
                if use_b:
                    h_b, cache_b = self.llm.forward_one_step(cur_b, masks=masks, cache=cache_b)
            seen = total
            last_f = h_f[:, -1]
            last_p = h_p[:, -1] if use_p else None
            last_b = h_b[:, -1] if use_b else None

            stop = torch.sigmoid(self.stop_token_head(last_f)) > 0.5
            if bool(stop) and tstep > min_frames:
                break

            frame = _coarse_frame(last_f, last_p, last_b)
            frames.append(frame)
            nxt = self.coarse_frame_proj(frame).unsqueeze(1)
            cur_f = cur_p = cur_b = nxt

        if len(frames) == 0:
            frames.append(_coarse_frame(last_f, last_p, last_b))

        x = torch.stack(frames, dim=1)

        x_coarse = x.clone() if return_tiers else None

        for k, axis in enumerate(self.split_axes):
            p_known = p_known_list[k]
            x_cat = torch.cat([p_known, x], dim=1)
            o_cat = self.up_tiers[k].sample(x_cat, num_steps=up_steps)
            o = o_cat[:, p_known.size(1):]
            x = _interleave(x, o, axis)

        final = self._denormalize_mel(x)
        if not return_tiers:
            return final

        return final, x_coarse

    def _load_vocoder(self, device):
        if self._vocoder is not None:
            return self._vocoder
        if not self.vocoder_path:
            return None
        try:
            self._vocoder = self._load_vocos(device)
        except Exception as e:
            logging.warning(
                f"[VoxMel] failed to load vocos from {self.vocoder_path}: {e}"
            )
            self._vocoder = None
        return self._vocoder

    def _load_vocos(self, device):
        from vocos import Vocos

        voc = Vocos.from_hparams(os.path.join(self.vocoder_path, "config.yaml"))
        state = torch.load(
            os.path.join(self.vocoder_path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        voc.load_state_dict(state, strict=False)

        if self.vocoder_weights_ckpt:
            ck = torch.load(
                self.vocoder_weights_ckpt, map_location="cpu", weights_only=True
            )
            sd = ck.get("state_dict", ck)
            gen = {k: v for k, v in sd.items()
                   if k.split(".", 1)[0] in ("feature_extractor", "backbone", "head")}
            missing, unexpected = voc.load_state_dict(gen, strict=False)
            logging.info(f"[VoxMel] vocos fine-tuned weights from {self.vocoder_weights_ckpt}: "
                         f"loaded {len(gen)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
        voc = voc.eval().to(device)

        def _decode(mel):
            return voc.decode(mel.transpose(1, 2))

        return _decode


