from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.roi_patch_dataset import make_fixed_condition_dataset
from models.baseline_cnn import BaselineCNN
from models.hf_gated_fusion import HFGatedFusionNet
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_confusion, compute_metrics
from utils.plots import save_confusion_matrix_png
from utils.seed import seed_everything


CLASS_NAMES = ["2peaks_close", "2peaks_far", "3peaks_line", "3peaks_cluster"]


def _build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "baseline":
        return BaselineCNN(in_channels=2, num_classes=num_classes)
    if name == "hf_gated":
        return HFGatedFusionNet(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def _write_metrics_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "L", "accuracy", "macro_f1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summary_mean_std(rows_all: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    key_to_vals: dict[tuple[float, int], list[dict]] = {}
    for r in rows_all:
        key = (float(r["snr_db"]), int(r["L"]))
        key_to_vals.setdefault(key, []).append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["snr_db", "L", "acc_mean", "acc_std", "f1_mean", "f1_std", "n"],
        )
        w.writeheader()
        for (snr_db, L), rs in sorted(key_to_vals.items(), key=lambda x: (x[0][0], x[0][1])):
            acc = np.array([float(x["accuracy"]) for x in rs], dtype=np.float64)
            f1 = np.array([float(x["macro_f1"]) for x in rs], dtype=np.float64)
            w.writerow(
                {
                    "snr_db": snr_db,
                    "L": L,
                    "acc_mean": float(acc.mean()),
                    "acc_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
                    "f1_mean": float(f1.mean()),
                    "f1_std": float(f1.std(ddof=1) if len(f1) > 1 else 0.0),
                    "n": int(len(rs)),
                }
            )


def _collect_checkpoints(args: argparse.Namespace) -> list[str]:
    if args.ckpt:
        return [args.ckpt]
    if args.ckpt_dir:
        paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "repeat_*", "best.pt")))
        if len(paths) == 0:
            paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", "best.pt"), recursive=True))
        if len(paths) == 0:
            paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", "*.pt"), recursive=True))
        return paths
    raise ValueError("Provide --ckpt or --ckpt_dir")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--ckpt_dir", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--snr_list", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 10.0])
    p.add_argument("--L_list", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--hf_mode", type=str, default="auto", choices=["auto", "laplacian", "sobel"])
    p.add_argument("--normalize", type=str, default="auto", choices=["auto", "none", "per_sample"])
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=41)

    p.add_argument("--total_samples", type=int, default=24000)
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    ckpts = _collect_checkpoints(args)

    if args.out_dir:
        root_out = args.out_dir
    elif args.ckpt_dir:
        root_out = os.path.join(args.ckpt_dir, "eval")
    else:
        root_out = os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(root_out, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    rows_all: list[dict] = []
    for ckpt_idx, ckpt_path in enumerate(ckpts):
        payload = load_checkpoint(ckpt_path, map_location=device)
        model_name = str(payload.get("model", "baseline"))
        seed = int(payload.get("seed", 0))
        ckpt_args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
        hf_mode = str(ckpt_args.get("hf_mode", "laplacian")) if args.hf_mode == "auto" else args.hf_mode
        normalize = str(ckpt_args.get("normalize", "per_sample")) if args.normalize == "auto" else args.normalize
        seed_everything(seed)

        model = _build_model(model_name, num_classes=args.num_classes).to(device)
        model.load_state_dict(payload["model_state"])

        out_dir = os.path.join(root_out, f"ckpt_{ckpt_idx:03d}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "ckpt_path.txt"), "w", encoding="utf-8") as f:
            f.write(ckpt_path)

        rows: list[dict] = []
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []

        for snr_db in args.snr_list:
            for L in args.L_list:
                ds = make_fixed_condition_dataset(
                    split="test",
                    total_samples=args.total_samples,
                    split_ratio=args.split,
                    base_seed=seed,
                    snr_db=float(snr_db),
                    L=int(L),
                    hf_mode=hf_mode,
                    normalize=normalize,
                    enable_aug=False,
                    height=args.patch_size,
                    width=args.patch_size,
                )
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                y_true, y_pred = _predict(model, loader, device)
                m = compute_metrics(y_true, y_pred)
                cm = compute_confusion(y_true, y_pred, num_classes=args.num_classes)

                tag = f"snr{float(snr_db):g}_L{int(L)}"
                save_confusion_matrix_png(
                    cm,
                    CLASS_NAMES,
                    os.path.join(out_dir, f"confusion_matrix_{tag}_count.png"),
                    normalize=False,
                    title=f"Confusion (count) | SNR={float(snr_db):g}dB L={int(L)}",
                )
                save_confusion_matrix_png(
                    cm,
                    CLASS_NAMES,
                    os.path.join(out_dir, f"confusion_matrix_{tag}_norm.png"),
                    normalize=True,
                    title=f"Confusion (row-norm) | SNR={float(snr_db):g}dB L={int(L)}",
                )

                row = {"snr_db": float(snr_db), "L": int(L), "accuracy": float(m.accuracy), "macro_f1": float(m.macro_f1)}
                rows.append(row)
                rows_all.append({**row, "ckpt": ckpt_path})
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

        _write_metrics_csv(os.path.join(out_dir, "metrics.csv"), rows)

        y_true_all_np = np.concatenate(y_true_all, axis=0)
        y_pred_all_np = np.concatenate(y_pred_all, axis=0)
        cm_all = compute_confusion(y_true_all_np, y_pred_all_np, num_classes=args.num_classes)
        save_confusion_matrix_png(
            cm_all,
            CLASS_NAMES,
            os.path.join(out_dir, "confusion_matrix_count.png"),
            normalize=False,
            title="Confusion (count) | All conditions",
        )
        save_confusion_matrix_png(
            cm_all,
            CLASS_NAMES,
            os.path.join(out_dir, "confusion_matrix_norm.png"),
            normalize=True,
            title="Confusion (row-norm) | All conditions",
        )

    _summary_mean_std(rows_all, os.path.join(root_out, "summary_mean_std.csv"))
