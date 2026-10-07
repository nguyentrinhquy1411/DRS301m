"""Load the pretrained FER2013 ResNet-18 checkpoint (no training, no weight changes).

Checkpoint format (inspected from TyomPapiyan/fer2013-emotion-scanner): a plain dict with
``model_name`` (a timm model id), ``num_classes``, ``classes``, ``img_size``, ``src_size``,
``mean``, ``std``, ``grayscale``, ``preprocess``, ``metrics`` and ``state_dict``.
The architecture is therefore rebuilt with ``timm.create_model`` exactly like the original repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "fer2013_emotion_resnet18_v2.pt"
DEFAULT_DETECTOR = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"

REQUIRED_KEYS = ("model_name", "num_classes", "classes", "img_size", "src_size", "mean", "std", "state_dict")


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint is missing, unreadable or incompatible."""


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return CUDA when available, otherwise CPU, unless a device is requested explicitly."""
    if device is not None:
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("CUDA was requested but is not available. Use device='cpu' or install a CUDA build of PyTorch.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_emotion_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Load the pretrained model.

    Returns:
        model: the network in ``eval()`` mode on ``device`` (parameters frozen, weights untouched).
        class_names: class labels in the order of the output logits.
        metadata: preprocessing parameters read from the checkpoint (never hard-coded).
    """
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise ModelLoadError(
            f"Checkpoint not found: {path}\n"
            "Copy 'fer2013_emotion_resnet18_v2.pt' from https://github.com/TyomPapiyan/fer2013-emotion-scanner "
            "into the 'models/' folder."
        )
    try:
        import timm
    except ImportError as exc:
        raise ModelLoadError("Package 'timm' is missing. Run: pip install -r requirements.txt") from exc

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # corrupted file, wrong pickle format, ...
        raise ModelLoadError(f"Could not read checkpoint {path}: {exc}") from exc

    if not isinstance(ckpt, dict):
        raise ModelLoadError(
            f"Unsupported checkpoint type {type(ckpt).__name__}; expected a dict with metadata and 'state_dict'."
        )
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ModelLoadError(f"Checkpoint {path.name} is missing keys {missing}. Found keys: {sorted(ckpt)}")
    if len(ckpt["classes"]) != ckpt["num_classes"]:
        raise ModelLoadError("Checkpoint is inconsistent: len(classes) != num_classes.")

    torch_device = get_device(device)
    try:
        # pretrained=False: weights come only from this checkpoint, never from ImageNet.
        model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"])
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except Exception as exc:
        raise ModelLoadError(
            f"Checkpoint weights do not match architecture '{ckpt['model_name']}' "
            f"(timm {timm.__version__}): {exc}"
        ) from exc

    model.eval().to(torch_device)
    for p in model.parameters():  # Grad-CAM needs activation gradients only, not weight gradients
        p.requires_grad_(False)

    metadata: dict[str, Any] = {
        "model_name": ckpt["model_name"],
        "img_size": int(ckpt["img_size"]),
        "src_size": int(ckpt["src_size"]),
        "grayscale": bool(ckpt.get("grayscale", True)),
        "mean": [float(v) for v in ckpt["mean"]],
        "std": [float(v) for v in ckpt["std"]],
        "preprocess": ckpt.get("preprocess", ""),
        "metrics": ckpt.get("metrics", {}),
        "device": str(torch_device),
        "checkpoint": str(path),
    }
    return model, list(ckpt["classes"]), metadata
