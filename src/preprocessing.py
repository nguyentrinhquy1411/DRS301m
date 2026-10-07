"""Face detection, cropping and tensor preprocessing identical to the original scanner.

Colour convention used everywhere in this project: **images are uint8 BGR (OpenCV)**.
RGB inputs must be converted with ``ensure_bgr(image, "rgb")`` before entering the pipeline.
The visualisation images are never normalised; only the model tensor is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model_loader import DEFAULT_DETECTOR

FACE_MARGIN = 1.15  # same value as the original repo: enlarge the box so the crop looks like FER2013
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

Geometry = tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates, may extend outside the image


class ImageLoadError(ValueError):
    """Raised when an image cannot be read or decoded."""


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class PreparedFace:
    tensor: torch.Tensor   # [1, 3, img_size, img_size], normalised, on the model device
    face48: np.ndarray     # uint8 grayscale [src_size, src_size] actually seen by the model
    crop_bgr: np.ndarray   # colour crop of the same region (for visualisation only)
    geometry: Geometry     # crop rectangle in original-image coordinates


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. ``np.fromfile`` keeps non-ASCII paths working on Windows."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ImageLoadError(f"Could not decode '{path}': the file is corrupted or not a valid image.")
    return image


def ensure_bgr(image: np.ndarray, color_order: str = "bgr") -> np.ndarray:
    """Return a 3-channel uint8 BGR image from grayscale / BGR / RGB / BGRA / RGBA input."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageLoadError("Empty or invalid image array.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = color_order.lower() == "rgb"
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR if rgb else cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image.copy()
    raise ImageLoadError(f"Unsupported image shape {image.shape}.")


class FaceDetector:
    """Thin wrapper around OpenCV YuNet (the detector used by the original repo)."""

    def __init__(self, model_path: str | Path | None = None, score_threshold: float = 0.7,
                 nms_threshold: float = 0.3, max_side: int = 1280) -> None:
        path = Path(model_path) if model_path else DEFAULT_DETECTOR
        if not path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {path}\nDownload 'face_detection_yunet_2023mar.onnx' from "
                "https://github.com/TyomPapiyan/fer2013-emotion-scanner and put it into 'models/'."
            )
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(f"OpenCV {cv2.__version__} has no FaceDetectorYN. Install opencv-python >= 4.5.4.")
        self._detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, nms_threshold, 5000)
        self.max_side = max_side

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        """Return faces sorted by area, largest first. Empty list when nothing is found."""
        h, w = image_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        work = cv2.resize(image_bgr, (round(w * scale), round(h * scale))) if scale < 1.0 else image_bgr
        self._detector.setInputSize((work.shape[1], work.shape[0]))
        _, faces = self._detector.detect(work)
        if faces is None:
            return []
        boxes = []
        for f in faces:
            x, y, bw, bh = (f[:4] / scale).round().astype(int)
            if bw > 0 and bh > 0:
                boxes.append(FaceBox(int(x), int(y), int(bw), int(bh), float(f[-1])))
        return sorted(boxes, key=lambda b: b.area, reverse=True)


def crop_geometry(box: FaceBox, margin: float = FACE_MARGIN) -> Geometry:
    """Square crop rectangle centred on the face box (same maths as the original ``crop_face``)."""
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    side = max(box.w, box.h) * margin
    return (int(round(cx - side / 2)), int(round(cy - side / 2)),
            int(round(cx + side / 2)), int(round(cy + side / 2)))


def crop_region(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Crop ``geometry``; parts outside the image are filled by replicating border pixels."""
    x1, y1, x2, y2 = geometry
    pad = max(0, -x1, -y1, x2 - image.shape[1], y2 - image.shape[0])
    if pad > 0:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return image[y1:y2, x1:x2]


def face_to_tensor(face48: np.ndarray, metadata: dict[str, Any], device: torch.device) -> torch.Tensor:
    """uint8 [S,S] -> float [1,3,img_size,img_size]. Order matches training:
    /255 -> bilinear resize to img_size -> repeat to 3 channels -> (x - mean) / std (applied once)."""
    src, size = metadata["src_size"], metadata["img_size"]
    x = torch.from_numpy(face48).to(device).float().div(255.0).view(1, 1, src, src)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(metadata["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(metadata["std"], device=device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - mean) / std


def prepare_face(image_bgr: np.ndarray, metadata: dict[str, Any], device: torch.device,
                 box: FaceBox | None = None, margin: float = FACE_MARGIN) -> PreparedFace:
    """Crop (or use the whole image when ``box`` is None, i.e. it is already a face crop) and preprocess."""
    h, w = image_bgr.shape[:2]
    geometry: Geometry = crop_geometry(box, margin) if box is not None else (0, 0, w, h)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    src = metadata["src_size"]
    face48 = cv2.resize(crop_region(gray, geometry), (src, src), interpolation=cv2.INTER_AREA)
    return PreparedFace(
        tensor=face_to_tensor(face48, metadata, device),
        face48=face48,
        crop_bgr=crop_region(image_bgr, geometry).copy(),
        geometry=geometry,
    )
