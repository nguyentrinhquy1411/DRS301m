# Dùng: python scripts/study_analysis.py <repo_root> <out_dir>  (ví dụ: . docs/images)
"""Tái lập số liệu cho tài liệu học: ablation TTA/bias, quét t, confusion, Grad-CAM attention shift."""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.append(str(ROOT / "src"))
from dataset import (CLASSES, SamplesDataset, get_transforms, get_train_val_samples, get_test_samples,
                     compute_class_prior, make_loader, source_group)
from models import load_checkpoint_model
from gradcam import GradCAM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, tf = get_transforms(224)


@torch.no_grad()
def logits_of(model, samples):
    dl = make_loader(SamplesDataset(samples, CLASSES, transform=tf), 16, False, 0)
    lo, lf, ys = [], [], []
    for x, y in dl:
        x = x.to(dev)
        lo.append(model(x).float().cpu()); lf.append(model(torch.flip(x, dims=[3])).float().cpu()); ys.append(y)
    return torch.cat(lo), torch.cat(lf), torch.cat(ys)


def metr(logits, y):
    p = logits.argmax(1).numpy(); y = y.numpy()
    return {"acc": float((p == y).mean()), "f1": float(f1_score(y, p, average="macro", zero_division=0))}


res = {}
test_sets = {"unmasked": get_test_samples("original"), "masked": get_test_samples("masked")}
for name in ["best_model_rafdb", "best_model_mask_aware"]:
    model, ck = load_checkpoint_model(ROOT / "output" / f"{name}.pth", dev)
    tr, va = get_train_val_samples(ck["data_mode"], 0.15, 42)
    prior = compute_class_prior(tr); lp = torch.log(prior)
    t_star = ck["logit_bias_info"]["t"]
    r = {"t_star": t_star, "prior": prior.tolist(), "n_train": len(tr), "n_val": len(va)}
    vo, vf, vy = logits_of(model, va)
    vt = 0.5 * (vo + vf)
    grid = [round(-0.5 + 0.05 * i, 2) for i in range(41)]
    r["val_sweep"] = {str(t): metr(vt + t * lp, vy) for t in grid}
    r["test"] = {}
    cache = {}
    for sname, smp in test_sets.items():
        o, f, y = logits_of(model, smp)
        cache[sname] = (o, f, y)
    cache["combined"] = tuple(torch.cat([cache["unmasked"][i], cache["masked"][i]]) for i in range(3))
    for sname, (o, f, y) in cache.items():
        t = 0.5 * (o + f)
        r["test"][sname] = {
            "raw": metr(o, y), "tta": metr(t, y), "bias": metr(o + t_star * lp, y), "tta_bias": metr(t + t_star * lp, y),
            "sweep": {str(tt): metr(t + tt * lp, y) for tt in grid},
        }
        for tag, L in (("t0", t), ("tstar", t + t_star * lp)):
            pred = L.argmax(1).numpy()
            r["test"][sname][f"cm_{tag}"] = confusion_matrix(y.numpy(), pred, labels=range(7)).tolist()
    res[name] = r
    print(name, json.dumps({k: {m: r["test"][k][m] for m in ("raw", "tta", "bias", "tta_bias")} for k in r["test"]}, indent=1))

    # Grad-CAM: tỷ lệ "năng lượng" heatmap ở nửa trên (y < 0.5H) trên các cặp ảnh gốc / khẩu trang
    cam = GradCAM(model, model.get_last_conv_layer())
    orig_by_group = {source_group(p): (p, l) for p, l in test_sets["unmasked"]}
    pairs = [(orig_by_group[source_group(p)], (p, l)) for p, l in test_sets["masked"] if source_group(p) in orig_by_group]
    from PIL import Image
    stats = {"unmasked": [], "masked": []}
    per_class = {c: {"unmasked": [], "masked": []} for c in CLASSES}
    mean_maps = {"unmasked": np.zeros((224, 224)), "masked": np.zeros((224, 224))}
    for (po, lab), (pm, _) in pairs:
        for key, p in (("unmasked", po), ("masked", pm)):
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            h, _, _ = cam.generate_heatmap(x, target_class=lab)   # giải thích cho nhãn thật
            s = h.sum()
            frac = float(h[:112].sum() / s) if s > 0 else float("nan")
            stats[key].append(frac); per_class[CLASSES[lab]][key].append(frac)
            mean_maps[key] += h
    cam.remove()
    r["cam_upper_frac"] = {k: float(np.nanmean(v)) for k, v in stats.items()}
    r["cam_upper_frac_per_class"] = {c: {k: float(np.nanmean(v)) for k, v in d.items()} for c, d in per_class.items()}
    r["cam_n_pairs"] = len(pairs)
    np.save(OUT / f"meanmap_{name}.npy", np.stack([mean_maps["unmasked"], mean_maps["masked"]]) / len(pairs))
    print(name, "CAM upper frac", r["cam_upper_frac"])
    del model; torch.cuda.empty_cache()

(OUT / "study_metrics.json").write_text(json.dumps(res, indent=1))
print("done")
