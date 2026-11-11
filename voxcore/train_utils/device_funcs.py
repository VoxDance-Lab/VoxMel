import dataclasses
import warnings

import numpy as np
import torch

def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    if isinstance(data, dict):
        return {k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()}
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[to_device(v, device, dtype, non_blocking, copy) for v in dataclasses.astuple(data)]
        )

    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[to_device(o, device, dtype, non_blocking, copy) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data

def force_gatherable(data, device):
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}

    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:

            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        warnings.warn(f"{type(data)} may not be gatherable by DataParallel")
        return data


def resolve_device(device="auto"):
    """Resolve automatic placement; reject unavailable explicit accelerators."""
    value = str(device)
    mps = getattr(torch.backends, "mps", None)
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else (
            "mps" if mps is not None and mps.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='auto' or 'cpu'")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {resolved.index}")
    elif resolved.type == "mps":
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS is unavailable; use device='auto' or 'cpu'")
    elif resolved.type != "cpu":
        raise ValueError("supported devices: auto, cpu, mps, cuda, cuda:N")
    return resolved
