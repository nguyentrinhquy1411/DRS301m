"""CLI: python main.py --image examples/input.png [--alpha 0.4] [--class predicted|happy|5] ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.gradcam import GradCAMError
from src.inference import EmotionExplainer, parse_class, save_outputs
from src.model_loader import ModelLoadError
from src.preprocessing import SUPPORTED_EXTENSIONS, ImageLoadError, load_image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grad-CAM explanations for the pretrained FER2013 ResNet-18.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="path to one image")
    src.add_argument("--folder", type=Path, help="process every image in a folder (one CAM per image)")
    src.add_argument("--video", help="video file path or webcam index (e.g. 0)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--alpha", type=float, default=0.4, help="overlay opacity in [0, 1] (e.g. 0.2-0.6; default 0.4)")
    p.add_argument("--class", dest="target_class", default="predicted",
                   help="'predicted' (default), a class name (e.g. sad) or an index 0-6 to explain")
    p.add_argument("--no-face-detection", action="store_true",
                   help="treat the image as an already tightly cropped face (e.g. 48x48 grayscale)")
    p.add_argument("--face-selection", choices=("largest", "all"), default="largest")
    p.add_argument("--mode", choices=("crop", "full"), default="crop",
                   help="crop: overlay on the face crop; full: paste Grad-CAM back into the full image")
    p.add_argument("--confidence-threshold", type=float, default=0.50)
    p.add_argument("--min-face-size", type=int, default=40, help="warn when the face is smaller than this (pixels)")
    p.add_argument("--target-layer", default=None, help="override the Grad-CAM layer (default: layer4.1 block)")
    p.add_argument("--no-tta", action="store_true", help="disable horizontal-flip test-time augmentation")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--detector", type=Path, default=None)
    v = p.add_argument_group("video")
    v.add_argument("--gradcam-every", type=int, default=10, help="Grad-CAM every N frames (0 off, 1 every frame)")
    v.add_argument("--out-video", type=Path, default=None, help="output video (default: outputs/<name>_gradcam.mp4)")
    v.add_argument("--show", action="store_true", help="display a window while processing")
    v.add_argument("--mirror", action="store_true", help="mirror frames (selfie webcam)")
    v.add_argument("--max-frames", type=int, default=None)
    return p


def run_image(explainer: EmotionExplainer, path: Path, args, class_idx: int | None) -> None:
    image = load_image(path)
    analysis = explainer.analyze_image(image, class_idx, not args.no_face_detection, args.face_selection)
    result = save_outputs(explainer, image, analysis, args.output_dir, path.stem, args.alpha, args.mode)
    if result["status"] != "ok":
        print(f"[{path.name}] {result['message']}")
    else:
        print(f"[{path.name}] saved to {args.output_dir}/ ({result['prediction']}, {result['confidence'] * 100:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        print("Error: --alpha must be between 0 and 1.", file=sys.stderr)
        return 2
    try:
        explainer = EmotionExplainer(args.checkpoint, args.detector, args.device, args.confidence_threshold,
                                     args.min_face_size, not args.no_tta, args.target_layer,
                                     enable_gradcam=args.gradcam_every != 0 or not args.video)
        class_idx = parse_class(args.target_class, explainer.class_names)
        if args.video:
            out = args.out_video or args.output_dir / f"{Path(str(args.video)).stem}_gradcam.mp4"
            stats = explainer.process_video(args.video, out, args.gradcam_every, args.alpha, mirror=args.mirror,
                                            show=args.show, max_frames=args.max_frames)
            print(f"Video done: {stats} -> {out}")
        elif args.folder:
            files = sorted(f for f in args.folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not files:
                print(f"Error: no supported images in {args.folder}", file=sys.stderr)
                return 1
            for f in files:
                try:
                    run_image(explainer, f, args, class_idx)
                except ImageLoadError as exc:
                    print(f"[{f.name}] skipped: {exc}")
        else:
            run_image(explainer, args.image, args, class_idx)
    except (ModelLoadError, ImageLoadError, GradCAMError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
