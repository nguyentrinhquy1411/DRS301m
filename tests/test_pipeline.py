"""Run with:  python -m unittest discover -s tests -v   (from the project root).

Uses examples/input.jpg (public-domain NASA portrait) to synthesise the test conditions.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as cli  # noqa: E402
from src.inference import EmotionExplainer, parse_class, save_outputs  # noqa: E402
from src.model_loader import ModelLoadError, load_emotion_model  # noqa: E402
from src.preprocessing import ImageLoadError, ensure_bgr, load_image, prepare_face  # noqa: E402
from src.visualization import create_heatmap, overlay_heatmap  # noqa: E402

FACE = load_image(ROOT / "examples" / "input.jpg")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ex = EmotionExplainer(verbose=False, device="cpu")
        warnings.simplefilter("ignore")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ex.close()

    def test01_clear_frontal_face(self):
        res = self.ex.analyze_image(FACE)
        self.assertEqual(res["status"], "ok")
        f = res["faces"][0]
        self.assertEqual(f.prediction, "happy")
        self.assertEqual(f.heatmap.shape, (112, 112))
        self.assertAlmostEqual(float(f.heatmap.max()), 1.0, places=5)
        with tempfile.TemporaryDirectory() as d:
            save_outputs(self.ex, FACE, res, d, "input", 0.4)
            names = {p.name for p in Path(d).iterdir()}
            self.assertEqual(names, {f"input_{s}" for s in
                                     ("original.jpg", "heatmap.jpg", "overlay.jpg", "comparison.jpg", "result.json")})
            data = json.loads((Path(d) / "input_result.json").read_text())
            self.assertEqual(data["prediction"], "happy")
            self.assertEqual(data["class_index"], 3)

    def test02_no_face(self):
        blank = np.full((300, 300, 3), 127, np.uint8)
        res = self.ex.analyze_image(blank)
        self.assertEqual(res, {"status": "no_face", "message": "No face detected."})
        with tempfile.TemporaryDirectory() as d:
            save_outputs(self.ex, blank, res, d, "blank")
            self.assertTrue((Path(d) / "blank_original.jpg").exists())
            self.assertFalse((Path(d) / "blank_overlay.jpg").exists())

    def test03_two_faces(self):
        small = cv2.resize(FACE, (300, 300))
        canvas = np.full((512, 900, 3), 200, np.uint8)
        canvas[:, :512], canvas[100:400, 560:860] = FACE, small
        largest = self.ex.analyze_image(canvas, face_selection="largest")
        self.assertEqual(len(largest["faces"]), 1)
        self.assertLess(largest["faces"][0].box.x, 450)  # the big (left) face
        both = self.ex.analyze_image(canvas, face_selection="all")
        self.assertEqual(len(both["faces"]), 2)
        with tempfile.TemporaryDirectory() as d:
            r = save_outputs(self.ex, canvas, both, d, "two", 0.4, mode="full")
            self.assertEqual(len(r["faces"]), 2)

    def test04_very_small_face(self):
        canvas = np.full((400, 400, 3), 200, np.uint8)
        tiny = cv2.resize(FACE, (130, 130), interpolation=cv2.INTER_AREA)
        canvas[100:230, 100:230] = tiny
        res = self.ex.analyze_image(canvas)
        if res["status"] == "no_face":
            self.skipTest("detector does not find faces this small (also acceptable behaviour)")
        self.assertIn("Face too small for reliable emotion classification.", res["faces"][0].warnings)

    def test05_dark_image(self):
        dark = (FACE.astype(np.float32) * 0.08).astype(np.uint8)
        res = self.ex.analyze_image(dark, use_face_detection=False)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(np.isfinite(res["faces"][0].heatmap).all())

    def test06_side_face_documented(self):
        self.skipTest("Needs a real profile photo; reduced reliability for side faces is documented in the README.")

    def test07_cropped_48x48_gray_without_detection(self):
        prepared = prepare_face(FACE, self.ex.metadata, self.ex.device, self.ex.detector.detect(FACE)[0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "face48.png"
            cv2.imwrite(str(path), prepared.face48)
            out = Path(d) / "out"
            code = cli.main(["--image", str(path), "--no-face-detection", "--output-dir", str(out), "--device", "cpu"])
            self.assertEqual(code, 0)
            self.assertTrue((out / "face48_overlay.jpg").exists())
            data = json.loads((out / "face48_result.json").read_text())
            self.assertEqual(data["prediction"], "happy")

    def test08_rgb_bgr_handling(self):
        rgb = cv2.cvtColor(FACE, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(ensure_bgr(rgb, "rgb"), FACE)
        a = self.ex.analyze_image(FACE)["faces"][0]
        b = self.ex.analyze_image(ensure_bgr(rgb, "rgb"))["faces"][0]
        self.assertEqual(a.prediction, b.prediction)
        self.assertAlmostEqual(a.confidence, b.confidence, places=5)
        # Jet in BGR: 0 -> dark blue (128,0,0), 1 -> dark red (0,0,128)
        jet = create_heatmap(np.array([[0.0, 1.0]], np.float32))
        self.assertEqual(jet[0, 0].tolist(), [128, 0, 0])
        self.assertEqual(jet[0, 1].tolist(), [0, 0, 128])
        ov = overlay_heatmap(FACE, a.heatmap, 0.4)
        self.assertEqual(ov.shape, FACE.shape)

    def test09_manual_target_class(self):
        pred = self.ex.analyze_image(FACE)["faces"][0]
        sad = self.ex.analyze_image(FACE, class_idx=parse_class("sad", self.ex.class_names))["faces"][0]
        self.assertEqual(pred.prediction, sad.prediction)       # prediction unchanged
        self.assertEqual(sad.target_name, "sad")
        self.assertFalse(np.allclose(pred.heatmap, sad.heatmap))  # different class -> different map

    def test10_video_gradcam_interval(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "clip.mp4"
            w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 512))
            for _ in range(25):
                w.write(FACE)
            w.release()
            for n, expected in ((10, 3), (0, 0), (1, 25)):
                s = self.ex.process_video(str(video), Path(d) / f"o{n}.mp4", gradcam_every_n=n, frame_width=None)
                self.assertEqual(s["frames"], 25)
                self.assertEqual(s["gradcam_frames"], expected, f"N={n}")

    def test11_hooks_do_not_accumulate(self):
        for _ in range(5):
            self.ex.analyze_image(FACE)
        layer = self.ex.cam.target_layer
        self.assertEqual(len(layer._forward_hooks), 1)
        self.assertEqual(len(layer._backward_hooks), 1)

    def test12_errors_are_clear(self):
        with self.assertRaises(ModelLoadError):
            load_emotion_model(ROOT / "models" / "nope.pt")
        with self.assertRaises(ImageLoadError):
            load_image(ROOT / "missing.jpg")
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.jpg"
            bad.write_bytes(b"not an image")
            with self.assertRaises(ImageLoadError):
                load_image(bad)
            with self.assertRaises(ImageLoadError):
                (Path(d) / "x.gif").write_bytes(b"GIF89a")
                load_image(Path(d) / "x.gif")

    def test13_low_confidence_flag(self):
        strict = EmotionExplainer(verbose=False, device="cpu", confidence_threshold=0.99)
        try:
            f = strict.analyze_image(FACE)["faces"][0]
            self.assertTrue(f.low_confidence)
            self.assertIsNotNone(f.heatmap)  # Grad-CAM still generated
        finally:
            strict.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
