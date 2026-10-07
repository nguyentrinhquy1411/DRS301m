"""End-to-end pipeline: detect -> crop -> predict -> Grad-CAM -> overlay (image, folder, video)."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .gradcam import GradCAM, check_heatmap, find_target_layer
from .model_loader import load_emotion_model
from .preprocessing import (FACE_MARGIN, FaceBox, FaceDetector, Geometry, PreparedFace, prepare_face)
from .visualization import (check_overlay, create_heatmap, draw_face_label, make_comparison, overlay_heatmap,
                            overlay_heatmap_in_image, paste_heatmap_in_image)

NO_FACE = {"status": "no_face", "message": "No face detected."}
SMALL_FACE_WARNING = "Face too small for reliable emotion classification."
LOW_CONFIDENCE_LABEL = "Low confidence"


@dataclass
class FaceResult:
    box: FaceBox | None
    geometry: Geometry
    crop_bgr: np.ndarray                 # colour crop used for visualisation
    probs: np.ndarray                    # class probabilities (flip-TTA average when enabled)
    class_index: int                     # predicted class
    prediction: str
    confidence: float
    target_index: int                    # class explained by Grad-CAM (== class_index unless chosen manually)
    target_name: str
    target_prob: float
    low_confidence: bool
    warnings: list[str] = field(default_factory=list)
    heatmap: np.ndarray | None = None    # [0, 1] float32, model-input resolution
    cam_raw_max: float = 0.0

    def title_lines(self) -> list[str]:
        lines = [f"Prediction: {self.prediction.capitalize()}   Confidence: {self.confidence * 100:.1f}%"
                 + (f"   ({LOW_CONFIDENCE_LABEL})" if self.low_confidence else "")]
        if self.target_index != self.class_index:
            lines.append(f"Grad-CAM explains: {self.target_name.capitalize()} (p = {self.target_prob * 100:.1f}%)")
        return lines

    def label(self) -> str:
        text = f"{self.prediction} {self.confidence * 100:.0f}%"
        if self.low_confidence:
            text += f" - {LOW_CONFIDENCE_LABEL}"
        if self.target_index != self.class_index:
            text += f" | CAM: {self.target_name}"
        return text

    def to_dict(self, target_layer: str, alpha: float) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prediction": self.prediction,
            "class_index": self.class_index,
            "confidence": round(self.confidence, 4),
            "explained_class": self.target_name,
            "explained_class_index": self.target_index,
            "explained_class_probability": round(self.target_prob, 4),
            "target_layer": target_layer,
            "alpha": alpha,
            "cam_min": None if self.heatmap is None else round(float(self.heatmap.min()), 4),
            "cam_max": None if self.heatmap is None else round(float(self.heatmap.max()), 4),
            "cam_raw_max": round(self.cam_raw_max, 6),
            "low_confidence": self.low_confidence,
            "probabilities": {},
            "face_box_xywh": None if self.box is None else list(self.box.as_tuple()),
            "crop_xyxy": list(self.geometry),
            "warnings": self.warnings,
        }
        return d


def parse_class(value: str | int | None, class_names: list[str]) -> int | None:
    """'predicted' / None -> None; a class name or index -> index."""
    if value is None or str(value).lower() == "predicted":
        return None
    text = str(value).strip().lower()
    if text.isdigit() and int(text) < len(class_names):
        return int(text)
    if text in class_names:
        return class_names.index(text)
    raise ValueError(f"Unknown class '{value}'. Use 'predicted', an index 0-{len(class_names) - 1} or one of {class_names}.")


class EmotionExplainer:
    """Holds the model, the Grad-CAM hooks (registered once) and the face detector."""

    def __init__(self, checkpoint_path: str | Path | None = None, detector_path: str | Path | None = None,
                 device: str | None = None, confidence_threshold: float = 0.50, min_face_size: int = 40,
                 use_tta: bool = True, target_layer: str | None = None, enable_gradcam: bool = True,
                 verbose: bool = True) -> None:
        self.model, self.class_names, self.metadata = load_emotion_model(checkpoint_path, device)
        self.device = next(self.model.parameters()).device
        self.detector_path = detector_path
        self._detector: FaceDetector | None = None
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.use_tta = use_tta
        self.enable_gradcam = enable_gradcam
        self.verbose = verbose
        if target_layer:
            try:
                layer, self.target_layer_name = self.model.get_submodule(target_layer), target_layer
            except AttributeError as exc:
                raise ValueError(f"Layer '{target_layer}' not found in the model.") from exc
        else:
            self.target_layer_name, layer = find_target_layer(self.model)
        self.cam = GradCAM(self.model, layer, self.target_layer_name)
        if verbose:
            print(f"Device: {self.device} | Model: {self.metadata['model_name']} | "
                  f"input {self.metadata['img_size']}x{self.metadata['img_size']} from {self.metadata['src_size']}x"
                  f"{self.metadata['src_size']} gray faces")
            print(f"Target Grad-CAM layer: {self.target_layer_name}")

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector(self.detector_path)
        return self._detector

    def close(self) -> None:
        self.cam.remove_hooks()

    # ---- prediction (no gradients) ---------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, tensor: torch.Tensor) -> np.ndarray:
        """Class probabilities; optionally averaged with the horizontally flipped input (as in the repo)."""
        batch = torch.cat([tensor, torch.flip(tensor, dims=[3])]) if self.use_tta else tensor
        return self.model(batch).softmax(dim=1).mean(dim=0).cpu().numpy()

    # ---- Grad-CAM (needs gradients) --------------------------------------------------------
    def compute_cam(self, tensor: torch.Tensor, class_idx: int) -> tuple[np.ndarray, float]:
        """Grad-CAM for ``class_idx``. With TTA the CAM of the flipped view is un-flipped and averaged."""
        heat, _, _ = self.cam.generate(tensor, class_idx)
        info = self.cam.last
        raw = heat * (info["cam_max"] - info["cam_min"]) + info["cam_min"]
        if self.use_tta:
            heat_f, _, _ = self.cam.generate(torch.flip(tensor, dims=[3]), class_idx)
            info_f = self.cam.last
            raw_f = heat_f * (info_f["cam_max"] - info_f["cam_min"]) + info_f["cam_min"]
            raw = (raw + raw_f[:, ::-1]) / 2.0
        top = float(raw.max())
        heatmap = (raw / top).astype(np.float32) if top > 1e-12 else np.zeros_like(raw, dtype=np.float32)
        return np.ascontiguousarray(heatmap), top

    def explain_face(self, image_bgr: np.ndarray, box: FaceBox | None, class_idx: int | None = None,
                     with_cam: bool | None = None, prepared: PreparedFace | None = None,
                     sanity: bool = True) -> FaceResult:
        """Predict one face and (optionally) compute its Grad-CAM. One forward+backward per face."""
        prepared = prepared or prepare_face(image_bgr, self.metadata, self.device, box)
        probs = self.predict_probs(prepared.tensor)
        pred = int(probs.argmax())
        target = pred if class_idx is None else class_idx
        if not 0 <= target < len(self.class_names):
            raise ValueError(f"class_idx {target} out of range.")
        confidence = float(probs[pred])

        x1, y1, x2, y2 = prepared.geometry
        warns: list[str] = []
        if min(x2 - x1, y2 - y1) < self.min_face_size or (box is not None and min(box.w, box.h) < self.min_face_size):
            warns.append(SMALL_FACE_WARNING)
        low = confidence < self.confidence_threshold
        if low:
            warns.append(f"{LOW_CONFIDENCE_LABEL} (< {self.confidence_threshold:.2f}); this does not by itself mean the "
                         "prediction is wrong.")

        result = FaceResult(box=box, geometry=prepared.geometry, crop_bgr=prepared.crop_bgr, probs=probs,
                            class_index=pred, prediction=self.class_names[pred], confidence=confidence,
                            target_index=target, target_name=self.class_names[target], target_prob=float(probs[target]),
                            low_confidence=low, warnings=warns)
        if with_cam is None:
            with_cam = self.enable_gradcam
        if with_cam:
            result.heatmap, result.cam_raw_max = self.compute_cam(prepared.tensor, target)
            if sanity:
                self._sanity_check(result, prepared, probs)
        return result

    def _sanity_check(self, r: FaceResult, prepared: PreparedFace, probs_before: np.ndarray) -> None:
        stats = check_heatmap(r.heatmap, tuple(prepared.tensor.shape[-2:]))
        after = self.predict_probs(prepared.tensor)
        if not np.allclose(probs_before, after, atol=1e-5):
            raise RuntimeError("Prediction changed after Grad-CAM generation; the model state was modified.")
        if self.verbose:
            print(f"Prediction: {r.prediction}\nConfidence: {r.confidence:.3f}\nTarget layer: {self.target_layer_name}\n"
                  f"Explained class: {r.target_name} (p={r.target_prob:.3f})\nCAM shape: {stats['shape']}\n"
                  f"CAM min: {stats['min']:.3f}\nCAM max: {stats['max']:.3f}\nPrediction unchanged after CAM: OK")
        for w in r.warnings:
            warnings.warn(w)

    # ---- single image ----------------------------------------------------------------------
    def analyze_image(self, image_bgr: np.ndarray, class_idx: int | None = None, use_face_detection: bool = True,
                      face_selection: str = "largest") -> dict[str, Any]:
        """Returns ``{"status": "ok", "faces": [FaceResult, ...]}`` or the ``no_face`` dict."""
        if face_selection not in ("largest", "all"):
            raise ValueError("face_selection must be 'largest' or 'all'")
        if not use_face_detection:
            return {"status": "ok", "faces": [self.explain_face(image_bgr, None, class_idx)]}
        boxes = self.detector.detect(image_bgr)
        if not boxes:
            return dict(NO_FACE)
        if face_selection == "largest":
            boxes = boxes[:1]
        elif len(boxes) > 1 and self.verbose:
            print(f"{len(boxes)} faces detected; processing each one separately.")
        return {"status": "ok", "faces": [self.explain_face(image_bgr, b, class_idx) for b in boxes]}

    # ---- rendering -------------------------------------------------------------------------
    @staticmethod
    def _display_crop(crop: np.ndarray) -> np.ndarray:
        side = max(224, min(max(crop.shape[:2]), 512))
        if max(crop.shape[:2]) == side:
            return crop
        return cv2.resize(crop, (round(crop.shape[1] * side / max(crop.shape[:2])),
                                 round(crop.shape[0] * side / max(crop.shape[:2]))), interpolation=cv2.INTER_CUBIC)

    def render_face(self, face: FaceResult, alpha: float) -> dict[str, np.ndarray]:
        """MODE 1: original crop / Jet heatmap / overlay for one face, plus the comparison figure."""
        original = self._display_crop(face.crop_bgr)
        h, w = original.shape[:2]
        heat_bgr = create_heatmap(face.heatmap, size=(w, h))
        overlay = overlay_heatmap(original, face.heatmap, alpha)
        check_overlay(original, overlay)
        return {"original": original, "heatmap": heat_bgr, "overlay": overlay,
                "comparison": make_comparison(original, heat_bgr, overlay, face.title_lines())}

    def render_full(self, image_bgr: np.ndarray, faces: list[FaceResult], alpha: float) -> dict[str, np.ndarray]:
        """MODE 2: every face's Grad-CAM placed back at its position in the original image."""
        overlay, heat_canvas = image_bgr.copy(), np.zeros_like(image_bgr)
        for f in faces:
            overlay = overlay_heatmap_in_image(overlay, f.heatmap, f.geometry, alpha)
            paste_heatmap_in_image(image_bgr.shape[:2], f.heatmap, f.geometry, heat_canvas)
            if f.box is not None:
                draw_face_label(overlay, f.box.as_tuple(), f.label(), (0, 255, 255) if f.low_confidence else (0, 255, 0))
        check_overlay(image_bgr, overlay)
        title = faces[0].title_lines() if len(faces) == 1 else [f"{len(faces)} faces analysed (see labels)"]
        return {"original": image_bgr, "heatmap": heat_canvas, "overlay": overlay,
                "comparison": make_comparison(image_bgr, heat_canvas, overlay, title)}

    # ---- video -----------------------------------------------------------------------------
    def process_video(
        self,
        source: str | int,
        output_path: str | Path | None = None,
        gradcam_every_n: int = 10,
        alpha: float = 0.4,
        smoothing: float = 0.7,
        frame_width: int | None = 640,
        mirror: bool = False,
        show: bool = False,
        hold_heatmap: bool = True,
        max_frames: int | None = None,
    ) -> dict[str, int]:
        """Run emotion recognition + Grad-CAM on all detected faces in a video."""
        src = int(source) if str(source).isdigit() else str(source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source '{source}'.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer: cv2.VideoWriter | None = None
        stats = {"frames": 0, "face_frames": 0, "gradcam_frames": 0, "faces_processed": 0}

        # Keep prediction/Grad-CAM state separately for each detected face index.
        smooth_probs: dict[int, np.ndarray] = {}
        held_heatmaps: dict[int, np.ndarray] = {}
        since_cam: dict[int, int] = {}

        try:
            while max_frames is None or stats["frames"] < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                if frame_width:
                    frame = cv2.resize(
                        frame,
                        (frame_width, round(frame.shape[0] * frame_width / frame.shape[1])),
                    )

                boxes = self.detector.detect(frame)

                if boxes:
                    stats["face_frames"] += 1
                    stats["faces_processed"] += len(boxes)

                    # Remove state for face indices that disappeared.
                    valid_indices = set(range(len(boxes)))
                    smooth_probs = {k: v for k, v in smooth_probs.items() if k in valid_indices}
                    held_heatmaps = {k: v for k, v in held_heatmaps.items() if k in valid_indices}
                    since_cam = {k: v for k, v in since_cam.items() if k in valid_indices}

                    for face_idx, box in enumerate(boxes):
                        prepared = prepare_face(frame, self.metadata, self.device, box)
                        probs = self.predict_probs(prepared.tensor)

                        if face_idx not in smooth_probs:
                            smooth_probs[face_idx] = probs
                        else:
                            smooth_probs[face_idx] = (
                                smoothing * smooth_probs[face_idx] + (1 - smoothing) * probs
                            )

                        smooth = smooth_probs[face_idx]
                        top = int(smooth.argmax())

                        # Run Grad-CAM independently for each face.
                        if face_idx not in since_cam:
                            since_cam[face_idx] = gradcam_every_n

                        if (
                            self.enable_gradcam
                            and gradcam_every_n > 0
                            and since_cam[face_idx] >= gradcam_every_n
                        ):
                            held_heatmaps[face_idx], _ = self.compute_cam(prepared.tensor, top)
                            since_cam[face_idx] = 0
                            stats["gradcam_frames"] += 1

                        since_cam[face_idx] += 1

                        if (
                            face_idx in held_heatmaps
                            and (hold_heatmap or since_cam[face_idx] == 1)
                        ):
                            frame = overlay_heatmap_in_image(
                                frame, held_heatmaps[face_idx], prepared.geometry, alpha
                            )

                        conf = float(smooth[top])
                        text = f"Face {face_idx + 1}: {self.class_names[top]} {conf * 100:.0f}%"
                        if conf < self.confidence_threshold:
                            text += f" - {LOW_CONFIDENCE_LABEL}"

                        draw_face_label(frame, box.as_tuple(), text)

                else:
                    smooth_probs.clear()
                    held_heatmaps.clear()
                    since_cam.clear()
                    cv2.putText(
                        frame,
                        "No face",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                stats["frames"] += 1

                if output_path is not None:
                    if writer is None:
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (frame.shape[1], frame.shape[0]),
                        )
                    writer.write(frame)

                if show:
                    cv2.imshow("Emotion + Grad-CAM (q to quit)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        return stats


# ---- saving ----------------------------------------------------------------------------------
def write_image(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))  # tofile keeps non-ASCII paths working on Windows


def save_outputs(explainer: EmotionExplainer, image_bgr: np.ndarray, analysis: dict[str, Any], out_dir: str | Path,
                 stem: str, alpha: float = 0.4, mode: str = "crop") -> dict[str, Any]:
    """Write ``{stem}_original/heatmap/overlay/comparison.jpg`` and ``{stem}_result.json``.

    mode "crop": Grad-CAM drawn on the face crop (one file set per face when several faces are processed).
    mode "full": Grad-CAM pasted back into the full image at each face's position.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if analysis["status"] != "ok":
        write_image(out / f"{stem}_original.jpg", image_bgr)
        (out / f"{stem}_result.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        return analysis

    faces: list[FaceResult] = analysis["faces"]
    if any(f.heatmap is None for f in faces):
        raise RuntimeError("Grad-CAM is disabled (enable_gradcam=False); nothing to visualise.")
    if mode == "full":
        renders = [("", explainer.render_full(image_bgr, faces, alpha))]
    else:
        renders = [("" if len(faces) == 1 else f"_face{i + 1}", explainer.render_face(f, alpha))
                   for i, f in enumerate(faces)]
    for suffix, images in renders:
        for name, img in images.items():
            write_image(out / f"{stem}{suffix}_{name}.jpg", img)

    face_dicts = []
    for f in faces:
        d = f.to_dict(explainer.target_layer_name, alpha)
        d["probabilities"] = {n: round(float(p), 4) for n, p in zip(explainer.class_names, f.probs)}
        face_dicts.append(d)
    result = {"status": "ok", "visualization_mode": mode, "tta": explainer.use_tta, **face_dicts[0], "faces": face_dicts}
    (out / f"{stem}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
