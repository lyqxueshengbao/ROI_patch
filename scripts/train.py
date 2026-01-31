from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.roi_patch_dataset import ToyROIPatchDataset, make_fixed_condition_dataset
from models.baseline_cnn import BaselineCNN
from models.hf_gated_fusion import HFGatedFusionNet
from utils.checkpoint import save_checkpoint
from utils.logging import CSVLogger, write_kv
from utils.metrics import compute_confusion, compute_metrics
from utils.plots import save_confusion_matrix_png
from utils.seed import seed_everything


CLASS_NAMES = ["2peaks_close", "2peaks_far", "3peaks_line", "3peaks_cluster"]


def _build_model(name: str, num_classes: int) -> nn.Module:
    if name == "baseline":
        return BaselineCNN(in_channels=2, num_classes=num_classes)
    if name == "hf_gated":
        return HFGatedFusionNet(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def _run_eval(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        pred = torch.argmax(logits, dim=1)
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        bs = int(xb.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    m = compute_metrics(y_true, y_pred)
    cm = compute_confusion(y_true, y_pred, num_classes=num_classes)
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": m.accuracy,
        "macro_f1": m.macro_f1,
        "confusion": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _train_one_repeat(args: argparse.Namespace, repeat_idx: int, seed: int) -> dict:
    use_shift_split = bool(args.train_roi_mode or args.test_roi_mode or args.train_aug_profile or args.test_aug_profile)
    subdir = f"rep_{repeat_idx:02d}" if use_shift_split else f"repeat_{repeat_idx:03d}"
    run_dir = os.path.join(args.out_dir, subdir)
    os.makedirs(run_dir, exist_ok=True)
    write_kv(os.path.join(run_dir, "run_args.txt"), {**vars(args), "repeat_idx": repeat_idx, "seed": seed})

    seed_everything(seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    if use_shift_split:
        train_roi_mode = str(args.train_roi_mode or args.roi_mode)
        test_roi_mode = str(args.test_roi_mode or train_roi_mode)
        train_profile = str(args.train_aug_profile or "oracle")
        test_profile = str(args.test_aug_profile or "pipeline")
    else:
        train_roi_mode = str(args.roi_mode)
        test_roi_mode = str(args.roi_mode)
        train_profile = ""
        test_profile = ""

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "repeat_id": int(repeat_idx),
                "seed": int(seed),
                "model": str(args.model),
                "train_roi_mode": train_roi_mode,
                "test_roi_mode": test_roi_mode,
                "train_profile": train_profile or None,
                "test_profile": test_profile or None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    ds_train = ToyROIPatchDataset(
        split="train",
        total_samples=args.total_samples,
        split_ratio=args.split,
        base_seed=seed,
        snr_list=args.snr_list,
        L_list=args.L_list,
        hf_mode=args.hf_mode,
        normalize=args.normalize,
        roi_mode=train_roi_mode,
        aug_profile=train_profile,
        center_sigma_oracle=args.center_sigma_oracle,
        center_sigma_min=args.center_sigma_min,
        center_sigma_max=args.center_sigma_max,
        pseudo_peak_prob=args.pseudo_peak_prob,
        pseudo_peak_max=args.pseudo_peak_max,
        warp_prob=args.warp_prob,
        warp_strength=args.warp_strength,
        corr_noise_prob=args.corr_noise_prob,
        corr_strength=args.corr_strength,
        enable_aug=not args.no_aug,
        height=args.patch_size,
        width=args.patch_size,
    )
    ds_val = ToyROIPatchDataset(
        split="val",
        total_samples=args.total_samples,
        split_ratio=args.split,
        base_seed=seed,
        snr_list=args.snr_list,
        L_list=args.L_list,
        hf_mode=args.hf_mode,
        normalize=args.normalize,
        roi_mode=test_roi_mode,
        aug_profile=test_profile,
        center_sigma_oracle=args.center_sigma_oracle,
        center_sigma_min=args.center_sigma_min,
        center_sigma_max=args.center_sigma_max,
        pseudo_peak_prob=args.pseudo_peak_prob,
        pseudo_peak_max=args.pseudo_peak_max,
        warp_prob=args.warp_prob,
        warp_strength=args.warp_strength,
        corr_noise_prob=args.corr_noise_prob,
        corr_strength=args.corr_strength,
        enable_aug=False,
        height=args.patch_size,
        width=args.patch_size,
    )
    ds_test = ToyROIPatchDataset(
        split="test",
        total_samples=args.total_samples,
        split_ratio=args.split,
        base_seed=seed,
        snr_list=args.snr_list,
        L_list=args.L_list,
        hf_mode=args.hf_mode,
        normalize=args.normalize,
        roi_mode=test_roi_mode,
        aug_profile=test_profile,
        center_sigma_oracle=args.center_sigma_oracle,
        center_sigma_min=args.center_sigma_min,
        center_sigma_max=args.center_sigma_max,
        pseudo_peak_prob=args.pseudo_peak_prob,
        pseudo_peak_max=args.pseudo_peak_max,
        warp_prob=args.warp_prob,
        warp_strength=args.warp_strength,
        corr_noise_prob=args.corr_noise_prob,
        corr_strength=args.corr_strength,
        enable_aug=False,
        height=args.patch_size,
        width=args.patch_size,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(args.model, num_classes=args.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = CSVLogger(
        os.path.join(run_dir, "train_log.csv"),
        ["epoch", "split", "loss", "accuracy", "macro_f1", "lr", "time_sec"],
    )

    best_val_f1 = -1.0
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ds_train.set_epoch(epoch)
        model.train()
        running = 0.0
        n = 0
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

            pred = torch.argmax(logits.detach(), dim=1)
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            bs = int(xb.shape[0])
            running += float(loss.item()) * bs
            n += bs

        scheduler.step()

        y_true_np = np.concatenate(y_true, axis=0)
        y_pred_np = np.concatenate(y_pred, axis=0)
        m_train = compute_metrics(y_true_np, y_pred_np)
        train_loss = running / max(n, 1)

        val_out = _run_eval(model, val_loader, device, num_classes=args.num_classes)
        lr = float(opt.param_groups[0]["lr"])
        dt = float(time.time() - t0)

        print(
            f"[repeat {repeat_idx:02d}] epoch {epoch:03d}/{args.epochs} "
            f"train loss={train_loss:.4f} acc={m_train.accuracy:.3f} f1={m_train.macro_f1:.3f} | "
            f"val loss={val_out['loss']:.4f} acc={val_out['accuracy']:.3f} f1={val_out['macro_f1']:.3f}"
        )

        log.log(
            {
                "epoch": epoch,
                "split": "train",
                "loss": train_loss,
                "accuracy": m_train.accuracy,
                "macro_f1": m_train.macro_f1,
                "lr": lr,
                "time_sec": dt,
            }
        )
        log.log(
            {
                "epoch": epoch,
                "split": "val",
                "loss": val_out["loss"],
                "accuracy": val_out["accuracy"],
                "macro_f1": val_out["macro_f1"],
                "lr": lr,
                "time_sec": dt,
            }
        )

        payload = {"model": args.model, "model_state": model.state_dict(), "epoch": epoch, "args": vars(args), "seed": seed}
        save_checkpoint(last_path, payload)
        if float(val_out["macro_f1"]) > best_val_f1:
            best_val_f1 = float(val_out["macro_f1"])
            save_checkpoint(best_path, payload)

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    test_out_mixed = _run_eval(model, test_loader, device, num_classes=args.num_classes)

    # Per-condition test (SNR,L) — avoids reporting only a mixed-distribution score.
    cond_rows: list[dict] = []
    in_domain_rows: list[dict] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_true_all_in: list[np.ndarray] = []
    y_pred_all_in: list[np.ndarray] = []
    for snr_db in args.snr_list:
        for L in args.L_list:
            ds_cond = make_fixed_condition_dataset(
                split="test",
                total_samples=args.total_samples,
                split_ratio=args.split,
                base_seed=seed,
                snr_db=float(snr_db),
                L=int(L),
                hf_mode=args.hf_mode,
                normalize=args.normalize,
                enable_aug=False,
                height=args.patch_size,
                width=args.patch_size,
                roi_mode=test_roi_mode,
                aug_profile=test_profile,
                center_sigma_oracle=args.center_sigma_oracle,
                center_sigma_min=args.center_sigma_min,
                center_sigma_max=args.center_sigma_max,
                pseudo_peak_prob=args.pseudo_peak_prob,
                pseudo_peak_max=args.pseudo_peak_max,
                warp_prob=args.warp_prob,
                warp_strength=args.warp_strength,
                corr_noise_prob=args.corr_noise_prob,
                corr_strength=args.corr_strength,
            )
            ds_cond_in = make_fixed_condition_dataset(
                split="test",
                total_samples=args.total_samples,
                split_ratio=args.split,
                base_seed=seed,
                snr_db=float(snr_db),
                L=int(L),
                hf_mode=args.hf_mode,
                normalize=args.normalize,
                enable_aug=False,
                height=args.patch_size,
                width=args.patch_size,
                roi_mode=train_roi_mode,
                aug_profile=train_profile,
                center_sigma_oracle=args.center_sigma_oracle,
                center_sigma_min=args.center_sigma_min,
                center_sigma_max=args.center_sigma_max,
                pseudo_peak_prob=args.pseudo_peak_prob,
                pseudo_peak_max=args.pseudo_peak_max,
                warp_prob=args.warp_prob,
                warp_strength=args.warp_strength,
                corr_noise_prob=args.corr_noise_prob,
                corr_strength=args.corr_strength,
            )
            loader_cond = DataLoader(
                ds_cond,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            loader_cond_in = DataLoader(
                ds_cond_in,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            out = _run_eval(model, loader_cond, device, num_classes=args.num_classes)
            out_in = _run_eval(model, loader_cond_in, device, num_classes=args.num_classes)
            cond_rows.append(
                {
                    "snr_db": float(snr_db),
                    "L": int(L),
                    "loss": float(out["loss"]),
                    "accuracy": float(out["accuracy"]),
                    "macro_f1": float(out["macro_f1"]),
                }
            )
            in_domain_rows.append(
                {
                    "snr_db": float(snr_db),
                    "L": int(L),
                    "loss": float(out_in["loss"]),
                    "accuracy": float(out_in["accuracy"]),
                    "macro_f1": float(out_in["macro_f1"]),
                }
            )
            save_confusion_matrix_png(
                out["confusion"],
                CLASS_NAMES,
                os.path.join(run_dir, f"confusion_matrix_snr{float(snr_db):g}_L{int(L)}_count.png"),
                normalize=False,
                title=f"Test confusion (count) | SNR={float(snr_db):g}dB L={int(L)}",
            )
            save_confusion_matrix_png(
                out["confusion"],
                CLASS_NAMES,
                os.path.join(run_dir, f"confusion_matrix_snr{float(snr_db):g}_L{int(L)}_norm.png"),
                normalize=True,
                title=f"Test confusion (row-norm) | SNR={float(snr_db):g}dB L={int(L)}",
            )

            # accumulate for an "all conditions" aggregate (equal test size per condition)
            y_true_all.append(out["y_true"])
            y_pred_all.append(out["y_pred"])
            y_true_all_in.append(out_in["y_true"])
            y_pred_all_in.append(out_in["y_pred"])

    with open(os.path.join(run_dir, "test_metrics_by_condition.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "L", "loss", "accuracy", "macro_f1"])
        w.writeheader()
        for r in cond_rows:
            w.writerow(r)

    with open(os.path.join(run_dir, "in_domain_metrics_by_condition.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "L", "loss", "accuracy", "macro_f1"])
        w.writeheader()
        for r in in_domain_rows:
            w.writerow(r)

    y_true_all_np = np.concatenate(y_true_all, axis=0)
    y_pred_all_np = np.concatenate(y_pred_all, axis=0)
    m_all = compute_metrics(y_true_all_np, y_pred_all_np)
    cm_all = compute_confusion(y_true_all_np, y_pred_all_np, num_classes=args.num_classes)
    save_confusion_matrix_png(
        cm_all,
        CLASS_NAMES,
        os.path.join(run_dir, "confusion_matrix_count.png"),
        normalize=False,
        title="Test confusion (count) | All conditions",
    )
    save_confusion_matrix_png(
        cm_all,
        CLASS_NAMES,
        os.path.join(run_dir, "confusion_matrix_norm.png"),
        normalize=True,
        title="Test confusion (row-normalized) | All conditions",
    )

    y_true_all_in_np = np.concatenate(y_true_all_in, axis=0)
    y_pred_all_in_np = np.concatenate(y_pred_all_in, axis=0)
    m_all_in = compute_metrics(y_true_all_in_np, y_pred_all_in_np)
    with open(os.path.join(run_dir, "in_domain_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["in_domain_accuracy_all_conditions", "in_domain_macro_f1_all_conditions"])
        w.writeheader()
        w.writerow(
            {
                "in_domain_accuracy_all_conditions": float(m_all_in.accuracy),
                "in_domain_macro_f1_all_conditions": float(m_all_in.macro_f1),
            }
        )

    with open(os.path.join(run_dir, "test_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "test_accuracy_all_conditions",
                "test_macro_f1_all_conditions",
                "test_accuracy_mixed_sampler",
                "test_macro_f1_mixed_sampler",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "test_accuracy_all_conditions": float(m_all.accuracy),
                "test_macro_f1_all_conditions": float(m_all.macro_f1),
                "test_accuracy_mixed_sampler": float(test_out_mixed["accuracy"]),
                "test_macro_f1_mixed_sampler": float(test_out_mixed["macro_f1"]),
            }
        )

    return {
        "repeat": repeat_idx,
        "seed": seed,
        "test_accuracy": float(m_all.accuracy),
        "test_macro_f1": float(m_all.macro_f1),
    }


def _summarize_repeats(out_dir: str, rows: list[dict]) -> None:
    accs = np.array([r["test_accuracy"] for r in rows], dtype=np.float64)
    f1s = np.array([r["test_macro_f1"] for r in rows], dtype=np.float64)
    out_path = os.path.join(out_dir, "summary_mean_std.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n"])
        w.writeheader()
        w.writerow({"metric": "test_accuracy", "mean": float(accs.mean()), "std": float(accs.std(ddof=1) if len(accs) > 1 else 0.0), "n": int(len(accs))})
        w.writerow({"metric": "test_macro_f1", "mean": float(f1s.mean()), "std": float(f1s.std(ddof=1) if len(f1s) > 1 else 0.0), "n": int(len(f1s))})


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="baseline", choices=["baseline", "hf_gated"])
    p.add_argument("--hf_mode", type=str, default="laplacian", choices=["laplacian", "sobel"])
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=41)
    p.add_argument("--normalize", type=str, default="per_sample", choices=["none", "per_sample"])
    p.add_argument("--no_aug", action="store_true")

    p.add_argument("--roi_mode", type=str, default="oracle", choices=["oracle", "pipeline"])
    p.add_argument("--train_roi_mode", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--test_roi_mode", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--train_aug_profile", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--test_aug_profile", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--center_sigma_oracle", type=float, default=1.0)
    p.add_argument("--center_sigma_min", type=float, default=1.5)
    p.add_argument("--center_sigma_max", type=float, default=6.0)
    p.add_argument("--pseudo_peak_prob", type=float, default=0.35)
    p.add_argument("--pseudo_peak_max", type=int, default=2)
    p.add_argument("--warp_prob", type=float, default=0.25)
    p.add_argument("--warp_strength", type=float, default=0.6)
    p.add_argument("--corr_noise_prob", type=float, default=0.25)
    p.add_argument("--corr_strength", type=float, default=0.6)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)

    p.add_argument("--snr_list", type=float, nargs="+", default=[5.0])
    p.add_argument("--L_list", type=int, nargs="+", default=[1])

    p.add_argument("--total_samples", type=int, default=24000)
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for i in range(int(args.repeat)):
        seed = int(args.seed) + i * 1000
        rows.append(_train_one_repeat(args, repeat_idx=i, seed=seed))
    _summarize_repeats(args.out_dir, rows)
