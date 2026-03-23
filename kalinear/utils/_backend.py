import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at runtime
    torch = None


def is_torch_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def infer_backend(*values):
    for value in values:
        if is_torch_tensor(value):
            return "torch"
    return "numpy"


def to_numpy(value):
    if value is None:
        return None
    if is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_backend(value, backend, reference=None):
    if value is None:
        return None
    if backend == "torch":
        if torch is None:
            raise ImportError("Torch backend requested but torch is not installed.")
        device = reference.device if is_torch_tensor(reference) else None
        return torch.as_tensor(value, device=device)
    return np.asarray(value)
