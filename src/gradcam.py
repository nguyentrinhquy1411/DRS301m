"""Standard Grad-CAM (Selvaraju et al., 2017) with safe hook management.

    alpha_k = mean over (i, j) of dY_c / dA_k(i, j)
    CAM     = ReLU( sum_k alpha_k * A_k )

Grad-CAM highlights image regions whose activations have a strong positive influence on the
selected class score. It is not an attention map and not a measurement of a person's emotion.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAMError(RuntimeError):
    """Raised when a heatmap cannot be computed (e.g. NaN / Inf values)."""


def find_last_conv_layer(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """Return ``(name, module)`` of the last ``nn.Conv2d`` in registration order."""
    last: tuple[str, nn.Conv2d] | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last = (name, module)
    if last is None:
        raise GradCAMError("The model contains no nn.Conv2d layer; Grad-CAM needs a convolutional target.")
    return last


def find_target_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Recommended Grad-CAM layer: the last residual block (``layer4[-1]``) of a ResNet.

    Its output is *after* the residual addition and ReLU, i.e. the feature map that actually
    feeds global pooling + the classifier. ``layer4.1.conv2`` (the last Conv2d) is pre-BatchNorm and
    pre-skip-connection, so it is only used as a fallback for architectures without ``layer4``.
    """
    layer4 = getattr(model, "layer4", None)
    if isinstance(layer4, nn.Sequential) and len(layer4) > 0:
        return f"layer4.{len(layer4) - 1}", layer4[-1]
    return find_last_conv_layer(model)


class GradCAM:
    """Reusable Grad-CAM. Hooks are registered once and removed with ``remove_hooks()``."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, target_layer_name: str = "") -> None:
        self.model = model
        self.target_layer = target_layer
        self.target_layer_name = target_layer_name or target_layer.__class__.__name__
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[Any] = []
        self.last: dict[str, Any] = {}
        self.register_hooks()

    # ---- hook management -------------------------------------------------------------------
    def register_hooks(self) -> None:
        if self._handles:  # never stack hooks
            return
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _save_activation(self, _module, _inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove_hooks()

    # ---- core ------------------------------------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute the Grad-CAM heatmap for one image.

        Args:
            input_tensor: preprocessed tensor ``[1, C, H, W]``.
            class_idx: class to explain; ``None`` explains the model's predicted class.
            output_size: ``(H, W)`` of the heatmap; defaults to the input tensor size.

        Returns:
            heatmap: float32 ``[H, W]`` in ``[0, 1]``.
            predicted_class: the model's argmax class (not necessarily the explained one).
            confidence: softmax probability of ``predicted_class``.

        ``self.last`` additionally holds ``target_class`` / ``target_prob`` (the explained class),
        ``cam_min`` / ``cam_max`` (before normalisation) and ``degenerate`` (CAM was all zeros).
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, C, H, W] (one CAM per call), got {tuple(input_tensor.shape)}")
        if not self._handles:
            raise GradCAMError("Hooks were removed; call register_hooks() before generate().")
        self.model.eval()  # eval (BatchNorm frozen) but gradients stay ON: no torch.no_grad() here
        size = output_size or tuple(input_tensor.shape[-2:])

        try:
            with torch.enable_grad():
                x = input_tensor.detach().clone().requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(x)
                probs = logits.detach().softmax(dim=1)[0]
                predicted_class = int(probs.argmax())
                target = predicted_class if class_idx is None else int(class_idx)
                if not 0 <= target < logits.shape[1]:
                    raise ValueError(f"class_idx {target} out of range [0, {logits.shape[1] - 1}]")
                logits[0, target].backward()

            if self._activations is None or self._gradients is None:
                raise GradCAMError("Hooks did not fire; the target layer is not on the path to the output.")

            acts, grads = self._activations, self._gradients            # [1, K, h, w]
            weights = grads.mean(dim=(2, 3), keepdim=True)              # alpha_k: global-average-pooled gradients
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))     # weighted sum over channels + ReLU
            cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
            cam_np = cam.cpu().numpy().astype(np.float32)
        finally:
            self._activations = self._gradients = None  # release the references to the graph tensors

        if not np.isfinite(cam_np).all():
            raise GradCAMError("Grad-CAM produced NaN/Inf values (check the input image and the model).")

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        degenerate = cam_max <= 1e-12
        if degenerate:
            warnings.warn("Grad-CAM is all zeros (no positive evidence for this class at the target layer).")
            heatmap = np.zeros_like(cam_np)
        else:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)

        self.last = {
            "target_class": target,
            "target_prob": float(probs[target]),
            "probs": probs.cpu().numpy(),
            "cam_min": cam_min,
            "cam_max": cam_max,
            "degenerate": degenerate,
        }
        return heatmap, predicted_class, float(probs[predicted_class])


def check_heatmap(heatmap: np.ndarray, expected_shape: tuple[int, int] | None = None) -> dict[str, Any]:
    """Sanity checks: shape, NaN/Inf and [0, 1] range. Raises ``GradCAMError`` on failure."""
    if heatmap.ndim != 2:
        raise GradCAMError(f"Heatmap must be 2-D, got shape {heatmap.shape}")
    if expected_shape is not None and heatmap.shape != tuple(expected_shape):
        raise GradCAMError(f"Heatmap shape {heatmap.shape} != expected {tuple(expected_shape)}")
    if np.isnan(heatmap).any() or np.isinf(heatmap).any():
        raise GradCAMError("Heatmap contains NaN or Inf values.")
    lo, hi = float(heatmap.min()), float(heatmap.max())
    if lo < -1e-6 or hi > 1 + 1e-6:
        raise GradCAMError(f"Heatmap outside [0, 1]: min={lo:.4f}, max={hi:.4f}")
    return {"shape": tuple(heatmap.shape), "min": lo, "max": hi}
