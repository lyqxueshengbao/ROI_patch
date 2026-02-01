from __future__ import annotations

import argparse
import csv
import json
import os
import time
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.fdamimo_patch_generator import FdaMimoGenConfig, generate_fdamimo_batch_torch
from datasets.roi_patch_dataset import (
    ToyROIPatchDataset,
    apply_border_overrides,
    apply_near_peak_overrides,
    make_cfg,
    make_fixed_condition_dataset,
)
from datasets.toy_generator import ToyGenConfig
from models.baseline_cnn import BaselineCNN
from models.hf_gated_fusion import HFGatedFusionNet
from utils.checkpoint import save_checkpoint
from utils.logging import CSVLogger, write_kv
from utils.metrics import compute_confusion, compute_metrics
from utils.plots import save_confusion_matrix_png
from utils.seed import seed_everything


CLASS_NAMES = ["2peaks_close", "2peaks_far", "3peaks_line", "3peaks_cluster"]


def _build_model(name: str, num_classes: int, in_channels: int = 2) -> nn.Module:
    if name == "baseline":
        return BaselineCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "hf_gated":
        return HFGatedFusionNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def _preview_cfg(
    args: argparse.Namespace,
    *,
    profile: str,
    roi_mode: str,
    occlude_mode: str | None,
    border_prob: float | None,
    border_sides: str | None,
    border_min: int | None,
    border_max: int | None,
    border_fill: str | None,
    border_sat_q: float | None,
    border_sat_strength: float | None,
    border_sat_noise: float | None,
    border_sat_clip: bool | None,
    near_peak_prob: float | None,
    near_peak_per_true_max: int | None,
    near_peak_radius_min: float | None,
    near_peak_radius_max: float | None,
    near_peak_amp_min: float | None,
    near_peak_amp_max: float | None,
    near_peak_sigma_scale_min: float | None,
    near_peak_sigma_scale_max: float | None,
    near_peak_mode: str | None,
) -> ToyGenConfig:
    snr0 = float(args.snr_list[0]) if len(args.snr_list) else float(args.snr_list)
    L0 = int(args.L_list[0]) if len(args.L_list) else int(args.L_list)
    if profile:
        cfg = make_cfg(
            profile=profile,  # type: ignore[arg-type]
            roi_mode=roi_mode,  # type: ignore[arg-type]
            snr_db=snr0,
            L=L0,
            hf_mode=args.hf_mode,
            normalize=args.normalize,
            enable_aug=True,
            height=args.patch_size,
            width=args.patch_size,
            center_sigma_oracle=args.center_sigma_oracle,
            center_sigma_min=args.center_sigma_min,
            center_sigma_max=args.center_sigma_max,
            pseudo_peak_max=args.pseudo_peak_max,
        )
    else:
        cfg = ToyGenConfig(
            height=args.patch_size,
            width=args.patch_size,
            snr_db=snr0,
            L=L0,
            hf_mode=args.hf_mode,
            normalize=args.normalize,
            roi_mode=roi_mode,  # type: ignore[arg-type]
            center_sigma_oracle=args.center_sigma_oracle,
            center_sigma_min=args.center_sigma_min,
            center_sigma_max=args.center_sigma_max,
            pseudo_peak_prob=args.pseudo_peak_prob,
            pseudo_peak_max=args.pseudo_peak_max,
            warp_prob=args.warp_prob,
            warp_strength=args.warp_strength,
            corr_noise_prob=args.corr_noise_prob,
            corr_strength=args.corr_strength,
            enable_aug=True,
        )

    cfg = apply_border_overrides(
        cfg,
        occlude_mode=occlude_mode,  # type: ignore[arg-type]
        border_prob=border_prob,
        border_sides=border_sides,  # type: ignore[arg-type]
        border_min=border_min,
        border_max=border_max,
        border_fill=border_fill,  # type: ignore[arg-type]
        border_sat_q=border_sat_q,
        border_sat_strength=border_sat_strength,
        border_sat_noise=border_sat_noise,
        border_sat_clip=border_sat_clip,
    )
    cfg = apply_near_peak_overrides(
        cfg,
        near_peak_prob=near_peak_prob,
        near_peak_per_true_max=near_peak_per_true_max,
        near_peak_radius_min=near_peak_radius_min,
        near_peak_radius_max=near_peak_radius_max,
        near_peak_amp_min=near_peak_amp_min,
        near_peak_amp_max=near_peak_amp_max,
        near_peak_sigma_scale_min=near_peak_sigma_scale_min,
        near_peak_sigma_scale_max=near_peak_sigma_scale_max,
        near_peak_mode=near_peak_mode,  # type: ignore[arg-type]
    )
    return cfg


def _print_border_cfg(tag: str, cfg: ToyGenConfig) -> None:
    print(
        f"[{tag}] data_source=toy occlude_mode={cfg.occlude_mode} border_prob={cfg.border_prob:g} "
        f"border_min={cfg.border_min} border_max={cfg.border_max} border_sides={cfg.border_sides} "
        f"border_fill={cfg.border_fill} sat_strength={cfg.border_sat_strength:g} sat_q={cfg.border_sat_q:g} "
        f"sat_noise={cfg.border_sat_noise:g} sat_clip={int(bool(cfg.border_sat_clip))} | "
        f"near_peak_prob={cfg.near_peak_prob:g} per_true_max={cfg.near_peak_per_true_max} "
        f"r=[{cfg.near_peak_radius_min:g},{cfg.near_peak_radius_max:g}] "
        f"amp=[{cfg.near_peak_amp_min:g},{cfg.near_peak_amp_max:g}] "
        f"sigma=[{cfg.near_peak_sigma_scale_min:g},{cfg.near_peak_sigma_scale_max:g}] mode={cfg.near_peak_mode}"
    )


def _preview_fdamimo_cfg(args: argparse.Namespace, *, roi_mode: str) -> FdaMimoGenConfig:
    snr0 = float(args.snr_list[0]) if len(args.snr_list) else float(args.snr_list)
    L0 = int(args.L_list[0]) if len(args.L_list) else int(args.L_list)
    return FdaMimoGenConfig(
        patch_h=int(args.patch_size),
        patch_w=int(args.patch_size),
        snr_db=float(snr0),
        L=int(L0),
        hf_mode=str(args.hf_mode),  # type: ignore[arg-type]
        normalize=str(args.normalize),  # type: ignore[arg-type]
        spec_mode=str(args.fdamimo_spec_mode),  # type: ignore[arg-type]
        roi_mode=str(roi_mode),  # type: ignore[arg-type]
        center_sigma_oracle=float(args.center_sigma_oracle),
        center_sigma_min=float(args.center_sigma_min),
        center_sigma_max=float(args.center_sigma_max),
        theta_span_deg=float(args.fdamimo_theta_span_deg),
        r_span_m=float(args.fdamimo_r_span_m),
        f0=float(args.fdamimo_f0),
        M=int(args.fdamimo_M),
        N=int(args.fdamimo_N),
        delta_f=float(args.fdamimo_delta_f),
        d=float(args.fdamimo_d),
        c=float(args.fdamimo_c),
    )


def _print_fdamimo_cfg(tag: str, cfg: FdaMimoGenConfig) -> None:
    print(
        f"[{tag}] data_source=fdamimo roi_mode={cfg.roi_mode} spec_mode={cfg.spec_mode} "
        f"snr_db={cfg.snr_db:g} L={cfg.L} patch={cfg.patch_h}x{cfg.patch_w} "
        f"theta_span_deg={cfg.theta_span_deg:g} r_span_m={cfg.r_span_m:g} "
        f"M={cfg.M} N={cfg.N} f0={cfg.f0:g} delta_f={cfg.delta_f:g} d={cfg.d:g}"
    )


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


def _make_torch_generator(seed: int, device: torch.device) -> torch.Generator | None:
    # Torch generator device support differs across versions; allow fallback to global RNG.
    seed = int(seed) % (2**63 - 1)
    if device.type != "cuda":
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g
    try:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        return g
    except TypeError:
        torch.cuda.manual_seed_all(int(seed))
        return None


@torch.no_grad()
def _run_eval_fdamimo_gen(
    model: nn.Module,
    *,
    cfg: FdaMimoGenConfig,
    device: torch.device,
    num_classes: int,
    n_samples: int,
    batch_size: int,
    seed: int,
    snr_db_fixed: float | None,
    L_fixed: int | None,
    snr_list: list[float],
    L_list: list[int],
) -> dict:
    model.eval()
    crit = nn.CrossEntropyLoss()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    loss_sum = 0.0
    n = 0

    snr_vals = torch.tensor(snr_list, device=device, dtype=torch.float32)
    L_vals = torch.tensor(L_list, device=device, dtype=torch.int64)

    steps = int(math.ceil(int(n_samples) / max(int(batch_size), 1)))
    for step in range(steps):
        bs = int(min(int(batch_size), int(n_samples) - step * int(batch_size)))
        if bs <= 0:
            break
        g = _make_torch_generator(int(seed) + step * 10007, device=device)
        labels = torch.randint(0, int(num_classes), (bs,), device=device, generator=g, dtype=torch.int64)
        if snr_db_fixed is None:
            idx = torch.randint(0, int(snr_vals.numel()), (bs,), device=device, generator=g, dtype=torch.int64)
            snr_db = snr_vals[idx]
        else:
            snr_db = torch.full((bs,), float(snr_db_fixed), device=device, dtype=torch.float32)
        if L_fixed is None:
            idx = torch.randint(0, int(L_vals.numel()), (bs,), device=device, generator=g, dtype=torch.int64)
            L = L_vals[idx]
        else:
            L = torch.full((bs,), int(L_fixed), device=device, dtype=torch.int64)

        xb = generate_fdamimo_batch_torch(
            labels=labels,
            snr_db=snr_db,
            L=L,
            cfg=cfg,
            generator=g,
            device=device,
        )
        logits = model(xb)
        loss = crit(logits, labels)
        pred = torch.argmax(logits, dim=1)
        ys.append(labels.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        loss_sum += float(loss.item()) * bs
        n += bs

    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(ps, axis=0) if ps else np.zeros((0,), dtype=np.int64)
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

    if str(args.data_source) == "toy":
        cfg_train_preview = _preview_cfg(
            args,
            profile=train_profile,
            roi_mode=train_roi_mode,
            occlude_mode=args.train_occlude_mode,
            border_prob=args.train_border_prob,
            border_sides=args.train_border_sides,
            border_min=args.train_border_min,
            border_max=args.train_border_max,
            border_fill=args.train_border_fill,
            border_sat_q=args.train_border_sat_q,
            border_sat_strength=args.train_border_sat_strength,
            border_sat_noise=args.train_border_sat_noise,
            border_sat_clip=args.train_border_sat_clip,
            near_peak_prob=args.train_near_peak_prob,
            near_peak_per_true_max=args.train_near_peak_per_true_max,
            near_peak_radius_min=args.train_near_peak_radius_min,
            near_peak_radius_max=args.train_near_peak_radius_max,
            near_peak_amp_min=args.train_near_peak_amp_min,
            near_peak_amp_max=args.train_near_peak_amp_max,
            near_peak_sigma_scale_min=args.train_near_peak_sigma_scale_min,
            near_peak_sigma_scale_max=args.train_near_peak_sigma_scale_max,
            near_peak_mode=args.train_near_peak_mode,
        )
        cfg_test_preview = _preview_cfg(
            args,
            profile=test_profile,
            roi_mode=test_roi_mode,
            occlude_mode=args.test_occlude_mode,
            border_prob=args.test_border_prob,
            border_sides=args.test_border_sides,
            border_min=args.test_border_min,
            border_max=args.test_border_max,
            border_fill=args.test_border_fill,
            border_sat_q=args.test_border_sat_q,
            border_sat_strength=args.test_border_sat_strength,
            border_sat_noise=args.test_border_sat_noise,
            border_sat_clip=args.test_border_sat_clip,
            near_peak_prob=args.test_near_peak_prob,
            near_peak_per_true_max=args.test_near_peak_per_true_max,
            near_peak_radius_min=args.test_near_peak_radius_min,
            near_peak_radius_max=args.test_near_peak_radius_max,
            near_peak_amp_min=args.test_near_peak_amp_min,
            near_peak_amp_max=args.test_near_peak_amp_max,
            near_peak_sigma_scale_min=args.test_near_peak_sigma_scale_min,
            near_peak_sigma_scale_max=args.test_near_peak_sigma_scale_max,
            near_peak_mode=args.test_near_peak_mode,
        )
        _print_border_cfg(f"repeat {repeat_idx:02d} train_cfg", cfg_train_preview)
        _print_border_cfg(f"repeat {repeat_idx:02d} test_cfg ", cfg_test_preview)
    else:
        cfg_train_preview_f = _preview_fdamimo_cfg(args, roi_mode=train_roi_mode)
        cfg_test_preview_f = _preview_fdamimo_cfg(args, roi_mode=test_roi_mode)
        _print_fdamimo_cfg(f"repeat {repeat_idx:02d} train_cfg", cfg_train_preview_f)
        _print_fdamimo_cfg(f"repeat {repeat_idx:02d} test_cfg ", cfg_test_preview_f)

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "repeat_id": int(repeat_idx),
                "seed": int(seed),
                "model": str(args.model),
                "data_source": str(args.data_source),
                "train_roi_mode": train_roi_mode,
                "test_roi_mode": test_roi_mode,
                "train_profile": train_profile or None,
                "test_profile": test_profile or None,
                "fdamimo_theta_span_deg": float(args.fdamimo_theta_span_deg),
                "fdamimo_r_span_m": float(args.fdamimo_r_span_m),
                "fdamimo_M": int(args.fdamimo_M),
                "fdamimo_N": int(args.fdamimo_N),
                "fdamimo_f0": float(args.fdamimo_f0),
                "fdamimo_delta_f": float(args.fdamimo_delta_f),
                "fdamimo_spec_mode": str(args.fdamimo_spec_mode),
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
        data_source=str(args.data_source),  # type: ignore[arg-type]
        roi_mode=train_roi_mode,
        aug_profile=train_profile,
        center_sigma_oracle=args.center_sigma_oracle,
        center_sigma_min=args.center_sigma_min,
        center_sigma_max=args.center_sigma_max,
        fdamimo_theta_span_deg=args.fdamimo_theta_span_deg,
        fdamimo_r_span_m=args.fdamimo_r_span_m,
        fdamimo_f0=args.fdamimo_f0,
        fdamimo_M=args.fdamimo_M,
        fdamimo_N=args.fdamimo_N,
        fdamimo_delta_f=args.fdamimo_delta_f,
        fdamimo_d=args.fdamimo_d,
        fdamimo_c=args.fdamimo_c,
        fdamimo_spec_mode=args.fdamimo_spec_mode,
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
    ds_train.set_border_overrides(
        occlude_mode=args.train_occlude_mode,  # type: ignore[arg-type]
        border_prob=args.train_border_prob,
        border_sides=args.train_border_sides,  # type: ignore[arg-type]
        border_min=args.train_border_min,
        border_max=args.train_border_max,
        border_fill=args.train_border_fill,  # type: ignore[arg-type]
        border_sat_q=args.train_border_sat_q,
        border_sat_strength=args.train_border_sat_strength,
        border_sat_noise=args.train_border_sat_noise,
        border_sat_clip=args.train_border_sat_clip,
    )
    ds_train.set_near_peak_overrides(
        near_peak_prob=args.train_near_peak_prob,
        near_peak_per_true_max=args.train_near_peak_per_true_max,
        near_peak_radius_min=args.train_near_peak_radius_min,
        near_peak_radius_max=args.train_near_peak_radius_max,
        near_peak_amp_min=args.train_near_peak_amp_min,
        near_peak_amp_max=args.train_near_peak_amp_max,
        near_peak_sigma_scale_min=args.train_near_peak_sigma_scale_min,
        near_peak_sigma_scale_max=args.train_near_peak_sigma_scale_max,
        near_peak_mode=args.train_near_peak_mode,  # type: ignore[arg-type]
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
        data_source=str(args.data_source),  # type: ignore[arg-type]
        roi_mode=test_roi_mode,
        aug_profile=test_profile,
        center_sigma_oracle=args.center_sigma_oracle,
        center_sigma_min=args.center_sigma_min,
        center_sigma_max=args.center_sigma_max,
        fdamimo_theta_span_deg=args.fdamimo_theta_span_deg,
        fdamimo_r_span_m=args.fdamimo_r_span_m,
        fdamimo_f0=args.fdamimo_f0,
        fdamimo_M=args.fdamimo_M,
        fdamimo_N=args.fdamimo_N,
        fdamimo_delta_f=args.fdamimo_delta_f,
        fdamimo_d=args.fdamimo_d,
        fdamimo_c=args.fdamimo_c,
        fdamimo_spec_mode=args.fdamimo_spec_mode,
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
    ds_val.set_border_overrides(
        occlude_mode=args.test_occlude_mode,  # type: ignore[arg-type]
        border_prob=args.test_border_prob,
        border_sides=args.test_border_sides,  # type: ignore[arg-type]
        border_min=args.test_border_min,
        border_max=args.test_border_max,
        border_fill=args.test_border_fill,  # type: ignore[arg-type]
        border_sat_q=args.test_border_sat_q,
        border_sat_strength=args.test_border_sat_strength,
        border_sat_noise=args.test_border_sat_noise,
        border_sat_clip=args.test_border_sat_clip,
    )
    ds_val.set_near_peak_overrides(
        near_peak_prob=args.test_near_peak_prob,
        near_peak_per_true_max=args.test_near_peak_per_true_max,
        near_peak_radius_min=args.test_near_peak_radius_min,
        near_peak_radius_max=args.test_near_peak_radius_max,
        near_peak_amp_min=args.test_near_peak_amp_min,
        near_peak_amp_max=args.test_near_peak_amp_max,
        near_peak_sigma_scale_min=args.test_near_peak_sigma_scale_min,
        near_peak_sigma_scale_max=args.test_near_peak_sigma_scale_max,
        near_peak_mode=args.test_near_peak_mode,  # type: ignore[arg-type]
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
        data_source=str(args.data_source),  # type: ignore[arg-type]
        roi_mode=test_roi_mode,
        aug_profile=test_profile,
        center_sigma_oracle=args.center_sigma_oracle,
        center_sigma_min=args.center_sigma_min,
        center_sigma_max=args.center_sigma_max,
        fdamimo_theta_span_deg=args.fdamimo_theta_span_deg,
        fdamimo_r_span_m=args.fdamimo_r_span_m,
        fdamimo_f0=args.fdamimo_f0,
        fdamimo_M=args.fdamimo_M,
        fdamimo_N=args.fdamimo_N,
        fdamimo_delta_f=args.fdamimo_delta_f,
        fdamimo_d=args.fdamimo_d,
        fdamimo_c=args.fdamimo_c,
        fdamimo_spec_mode=args.fdamimo_spec_mode,
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
    ds_test.set_border_overrides(
        occlude_mode=args.test_occlude_mode,  # type: ignore[arg-type]
        border_prob=args.test_border_prob,
        border_sides=args.test_border_sides,  # type: ignore[arg-type]
        border_min=args.test_border_min,
        border_max=args.test_border_max,
        border_fill=args.test_border_fill,  # type: ignore[arg-type]
        border_sat_q=args.test_border_sat_q,
        border_sat_strength=args.test_border_sat_strength,
        border_sat_noise=args.test_border_sat_noise,
        border_sat_clip=args.test_border_sat_clip,
    )
    ds_test.set_near_peak_overrides(
        near_peak_prob=args.test_near_peak_prob,
        near_peak_per_true_max=args.test_near_peak_per_true_max,
        near_peak_radius_min=args.test_near_peak_radius_min,
        near_peak_radius_max=args.test_near_peak_radius_max,
        near_peak_amp_min=args.test_near_peak_amp_min,
        near_peak_amp_max=args.test_near_peak_amp_max,
        near_peak_sigma_scale_min=args.test_near_peak_sigma_scale_min,
        near_peak_sigma_scale_max=args.test_near_peak_sigma_scale_max,
        near_peak_mode=args.test_near_peak_mode,  # type: ignore[arg-type]
    )

    use_gpu_gen = bool(args.gen_on_gpu) and str(args.data_source) == "fdamimo"
    if use_gpu_gen and device.type != "cuda":
        raise ValueError("--gen_on_gpu requires CUDA (device.type == 'cuda').")
    if use_gpu_gen and int(args.num_workers) != 0:
        print("[warn] --gen_on_gpu forces --num_workers=0 (DataLoader workers are not used).")

    train_loader = None if use_gpu_gen else DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = None if use_gpu_gen else DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = None if use_gpu_gen else DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Detect input channels from dataset sample
    sample_x, _ = ds_train[0]
    in_channels = int(sample_x.shape[0])
    print(f"[repeat {repeat_idx:02d}] Detected in_channels={in_channels}")

    model = _build_model(args.model, num_classes=args.num_classes, in_channels=in_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    fcfg_train = _preview_fdamimo_cfg(args, roi_mode=train_roi_mode) if use_gpu_gen else None
    fcfg_test = _preview_fdamimo_cfg(args, roi_mode=test_roi_mode) if use_gpu_gen else None

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

        if use_gpu_gen:
            assert fcfg_train is not None
            snr_list = [float(x) for x in args.snr_list]
            L_list = [int(x) for x in args.L_list]
            snr_vals = torch.tensor(snr_list, device=device, dtype=torch.float32)
            L_vals = torch.tensor(L_list, device=device, dtype=torch.int64)
            steps = int(math.ceil(len(ds_train) / max(int(args.batch_size), 1)))
            for step in range(steps):
                bs = int(min(int(args.batch_size), len(ds_train) - step * int(args.batch_size)))
                if bs <= 0:
                    break
                g = _make_torch_generator(seed * 1000003 + epoch * 9176 + step * 7919 + 17, device=device)
                yb = torch.randint(0, int(args.num_classes), (bs,), device=device, generator=g, dtype=torch.int64)
                i_snr = torch.randint(0, int(snr_vals.numel()), (bs,), device=device, generator=g, dtype=torch.int64)
                i_L = torch.randint(0, int(L_vals.numel()), (bs,), device=device, generator=g, dtype=torch.int64)
                snr_db = snr_vals[i_snr]
                L = L_vals[i_L]
                xb = generate_fdamimo_batch_torch(labels=yb, snr_db=snr_db, L=L, cfg=fcfg_train, generator=g, device=device)

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
                running += float(loss.item()) * bs
                n += bs
        else:
            assert train_loader is not None
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

        if use_gpu_gen:
            assert fcfg_test is not None
            val_out = _run_eval_fdamimo_gen(
                model,
                cfg=fcfg_test,
                device=device,
                num_classes=args.num_classes,
                n_samples=len(ds_val),
                batch_size=int(args.batch_size),
                seed=seed * 10007 + epoch * 31 + 101,
                snr_db_fixed=None,
                L_fixed=None,
                snr_list=[float(x) for x in args.snr_list],
                L_list=[int(x) for x in args.L_list],
            )
        else:
            assert val_loader is not None
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
    if use_gpu_gen:
        assert fcfg_test is not None
        test_out_mixed = _run_eval_fdamimo_gen(
            model,
            cfg=fcfg_test,
            device=device,
            num_classes=args.num_classes,
            n_samples=len(ds_test),
            batch_size=int(args.batch_size),
            seed=seed * 10007 + 202,
            snr_db_fixed=None,
            L_fixed=None,
            snr_list=[float(x) for x in args.snr_list],
            L_list=[int(x) for x in args.L_list],
        )
    else:
        assert test_loader is not None
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
                data_source=str(args.data_source),  # type: ignore[arg-type]
                roi_mode=test_roi_mode,
                aug_profile=test_profile,
                center_sigma_oracle=args.center_sigma_oracle,
                center_sigma_min=args.center_sigma_min,
                center_sigma_max=args.center_sigma_max,
                fdamimo_theta_span_deg=args.fdamimo_theta_span_deg,
                fdamimo_r_span_m=args.fdamimo_r_span_m,
                fdamimo_f0=args.fdamimo_f0,
                fdamimo_M=args.fdamimo_M,
                fdamimo_N=args.fdamimo_N,
                fdamimo_delta_f=args.fdamimo_delta_f,
                fdamimo_d=args.fdamimo_d,
                fdamimo_c=args.fdamimo_c,
                fdamimo_spec_mode=args.fdamimo_spec_mode,
                pseudo_peak_prob=args.pseudo_peak_prob,
                pseudo_peak_max=args.pseudo_peak_max,
                warp_prob=args.warp_prob,
                warp_strength=args.warp_strength,
                corr_noise_prob=args.corr_noise_prob,
                corr_strength=args.corr_strength,
                occlude_mode_override=args.test_occlude_mode,  # type: ignore[arg-type]
                border_prob_override=args.test_border_prob,
                border_sides_override=args.test_border_sides,  # type: ignore[arg-type]
                border_min_override=args.test_border_min,
                border_max_override=args.test_border_max,
                border_fill_override=args.test_border_fill,  # type: ignore[arg-type]
                border_sat_q_override=args.test_border_sat_q,
                border_sat_strength_override=args.test_border_sat_strength,
                border_sat_noise_override=args.test_border_sat_noise,
                border_sat_clip_override=args.test_border_sat_clip,
                near_peak_prob_override=args.test_near_peak_prob,
                near_peak_per_true_max_override=args.test_near_peak_per_true_max,
                near_peak_radius_min_override=args.test_near_peak_radius_min,
                near_peak_radius_max_override=args.test_near_peak_radius_max,
                near_peak_amp_min_override=args.test_near_peak_amp_min,
                near_peak_amp_max_override=args.test_near_peak_amp_max,
                near_peak_sigma_scale_min_override=args.test_near_peak_sigma_scale_min,
                near_peak_sigma_scale_max_override=args.test_near_peak_sigma_scale_max,
                near_peak_mode_override=args.test_near_peak_mode,  # type: ignore[arg-type]
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
                data_source=str(args.data_source),  # type: ignore[arg-type]
                roi_mode=train_roi_mode,
                aug_profile=train_profile,
                center_sigma_oracle=args.center_sigma_oracle,
                center_sigma_min=args.center_sigma_min,
                center_sigma_max=args.center_sigma_max,
                fdamimo_theta_span_deg=args.fdamimo_theta_span_deg,
                fdamimo_r_span_m=args.fdamimo_r_span_m,
                fdamimo_f0=args.fdamimo_f0,
                fdamimo_M=args.fdamimo_M,
                fdamimo_N=args.fdamimo_N,
                fdamimo_delta_f=args.fdamimo_delta_f,
                fdamimo_d=args.fdamimo_d,
                fdamimo_c=args.fdamimo_c,
                fdamimo_spec_mode=args.fdamimo_spec_mode,
                pseudo_peak_prob=args.pseudo_peak_prob,
                pseudo_peak_max=args.pseudo_peak_max,
                warp_prob=args.warp_prob,
                warp_strength=args.warp_strength,
                corr_noise_prob=args.corr_noise_prob,
                corr_strength=args.corr_strength,
                occlude_mode_override=args.train_occlude_mode,  # type: ignore[arg-type]
                border_prob_override=args.train_border_prob,
                border_sides_override=args.train_border_sides,  # type: ignore[arg-type]
                border_min_override=args.train_border_min,
                border_max_override=args.train_border_max,
                border_fill_override=args.train_border_fill,  # type: ignore[arg-type]
                border_sat_q_override=args.train_border_sat_q,
                border_sat_strength_override=args.train_border_sat_strength,
                border_sat_noise_override=args.train_border_sat_noise,
                border_sat_clip_override=args.train_border_sat_clip,
                near_peak_prob_override=args.train_near_peak_prob,
                near_peak_per_true_max_override=args.train_near_peak_per_true_max,
                near_peak_radius_min_override=args.train_near_peak_radius_min,
                near_peak_radius_max_override=args.train_near_peak_radius_max,
                near_peak_amp_min_override=args.train_near_peak_amp_min,
                near_peak_amp_max_override=args.train_near_peak_amp_max,
                near_peak_sigma_scale_min_override=args.train_near_peak_sigma_scale_min,
                near_peak_sigma_scale_max_override=args.train_near_peak_sigma_scale_max,
                near_peak_mode_override=args.train_near_peak_mode,  # type: ignore[arg-type]
            )
            if use_gpu_gen:
                assert fcfg_test is not None
                assert fcfg_train is not None
                out = _run_eval_fdamimo_gen(
                    model,
                    cfg=fcfg_test,
                    device=device,
                    num_classes=args.num_classes,
                    n_samples=len(ds_cond),
                    batch_size=int(args.batch_size),
                    seed=seed * 10007 + int(float(snr_db) * 10) + int(L) * 101 + 303,
                    snr_db_fixed=float(snr_db),
                    L_fixed=int(L),
                    snr_list=[float(snr_db)],
                    L_list=[int(L)],
                )
                out_in = _run_eval_fdamimo_gen(
                    model,
                    cfg=fcfg_train,
                    device=device,
                    num_classes=args.num_classes,
                    n_samples=len(ds_cond_in),
                    batch_size=int(args.batch_size),
                    seed=seed * 10007 + int(float(snr_db) * 10) + int(L) * 101 + 404,
                    snr_db_fixed=float(snr_db),
                    L_fixed=int(L),
                    snr_list=[float(snr_db)],
                    L_list=[int(L)],
                )
            else:
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
                    "data_source": str(args.data_source),
                    "snr_db": float(snr_db),
                    "L": int(L),
                    "loss": float(out["loss"]),
                    "accuracy": float(out["accuracy"]),
                    "macro_f1": float(out["macro_f1"]),
                }
            )
            in_domain_rows.append(
                {
                    "data_source": str(args.data_source),
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
        w = csv.DictWriter(f, fieldnames=["data_source", "snr_db", "L", "loss", "accuracy", "macro_f1"])
        w.writeheader()
        for r in cond_rows:
            w.writerow(r)

    with open(os.path.join(run_dir, "in_domain_metrics_by_condition.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["data_source", "snr_db", "L", "loss", "accuracy", "macro_f1"])
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
        w = csv.DictWriter(
            f,
            fieldnames=["data_source", "in_domain_accuracy_all_conditions", "in_domain_macro_f1_all_conditions"],
        )
        w.writeheader()
        w.writerow(
            {
                "data_source": str(args.data_source),
                "in_domain_accuracy_all_conditions": float(m_all_in.accuracy),
                "in_domain_macro_f1_all_conditions": float(m_all_in.macro_f1),
            }
        )

    with open(os.path.join(run_dir, "test_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "data_source",
                "test_accuracy_all_conditions",
                "test_macro_f1_all_conditions",
                "test_accuracy_mixed_sampler",
                "test_macro_f1_mixed_sampler",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "data_source": str(args.data_source),
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

    p.add_argument("--data_source", type=str, default="toy", choices=["toy", "fdamimo"])
    p.add_argument("--gen_on_gpu", action="store_true", help="Generate FDA-MIMO batches on GPU inside the train loop (no DataLoader workers).")

    # FDA-MIMO local patch parameters (only used when --data_source=fdamimo).
    p.add_argument("--fdamimo_theta_span_deg", type=float, default=20.0)
    p.add_argument("--fdamimo_r_span_m", type=float, default=400.0)
    p.add_argument("--fdamimo_f0", type=float, default=1e9)
    p.add_argument("--fdamimo_M", type=int, default=10)
    p.add_argument("--fdamimo_N", type=int, default=10)
    p.add_argument("--fdamimo_delta_f", type=float, default=30e3)
    p.add_argument("--fdamimo_d", type=float, default=0.5)
    p.add_argument("--fdamimo_c", type=float, default=3e8)
    p.add_argument("--fdamimo_spec_mode", type=str, default="power", choices=["power", "z_sincos"],
                   help="Spectrum mode: 'power' = |z|^2 (2-ch), 'z_sincos' = |z| + phase as sin/cos (4-ch)")

    p.add_argument("--roi_mode", type=str, default="oracle", choices=["oracle", "pipeline"])
    p.add_argument("--train_roi_mode", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--test_roi_mode", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--train_aug_profile", type=str, default="", choices=["", "oracle", "pipeline"])
    p.add_argument("--test_aug_profile", type=str, default="", choices=["", "oracle", "pipeline"])

    # Border-jam / occlusion overrides (when provided, override profile(make_cfg) defaults).
    p.add_argument("--train_occlude_mode", type=str, default=None, choices=["none", "block", "border"])
    p.add_argument("--train_border_prob", type=float, default=None)
    p.add_argument("--train_border_sides", type=str, default=None, choices=["one", "two", "rand12"])
    p.add_argument("--train_border_min", type=int, default=None)
    p.add_argument("--train_border_max", type=int, default=None)
    p.add_argument(
        "--train_border_fill",
        type=str,
        default=None,
        choices=["zero", "mean", "min", "sat_const", "sat_noise", "sat_quantile"],
    )
    p.add_argument("--train_border_sat_q", type=float, default=None)
    p.add_argument("--train_border_sat_strength", type=float, default=None)
    p.add_argument("--train_border_sat_noise", type=float, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--train_border_sat_clip", dest="train_border_sat_clip", action="store_true")
    g.add_argument("--train_no_border_sat_clip", dest="train_border_sat_clip", action="store_false")
    p.set_defaults(train_border_sat_clip=None)

    p.add_argument("--test_occlude_mode", type=str, default=None, choices=["none", "block", "border"])
    p.add_argument("--test_border_prob", type=float, default=None)
    p.add_argument("--test_border_sides", type=str, default=None, choices=["one", "two", "rand12"])
    p.add_argument("--test_border_min", type=int, default=None)
    p.add_argument("--test_border_max", type=int, default=None)
    p.add_argument(
        "--test_border_fill",
        type=str,
        default=None,
        choices=["zero", "mean", "min", "sat_const", "sat_noise", "sat_quantile"],
    )
    p.add_argument("--test_border_sat_q", type=float, default=None)
    p.add_argument("--test_border_sat_strength", type=float, default=None)
    p.add_argument("--test_border_sat_noise", type=float, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--test_border_sat_clip", dest="test_border_sat_clip", action="store_true")
    g.add_argument("--test_no_border_sat_clip", dest="test_border_sat_clip", action="store_false")
    p.set_defaults(test_border_sat_clip=None)

    # Near-peak pseudo peaks (targeted peak splitting/merging) overrides (default=None => no override).
    p.add_argument("--train_near_peak_prob", type=float, default=None)
    p.add_argument("--train_near_peak_per_true_max", type=int, default=None)
    p.add_argument("--train_near_peak_radius_min", type=float, default=None)
    p.add_argument("--train_near_peak_radius_max", type=float, default=None)
    p.add_argument("--train_near_peak_amp_min", type=float, default=None)
    p.add_argument("--train_near_peak_amp_max", type=float, default=None)
    p.add_argument("--train_near_peak_sigma_scale_min", type=float, default=None)
    p.add_argument("--train_near_peak_sigma_scale_max", type=float, default=None)
    p.add_argument("--train_near_peak_mode", type=str, default=None, choices=["around_each", "around_random_true"])

    p.add_argument("--test_near_peak_prob", type=float, default=None)
    p.add_argument("--test_near_peak_per_true_max", type=int, default=None)
    p.add_argument("--test_near_peak_radius_min", type=float, default=None)
    p.add_argument("--test_near_peak_radius_max", type=float, default=None)
    p.add_argument("--test_near_peak_amp_min", type=float, default=None)
    p.add_argument("--test_near_peak_amp_max", type=float, default=None)
    p.add_argument("--test_near_peak_sigma_scale_min", type=float, default=None)
    p.add_argument("--test_near_peak_sigma_scale_max", type=float, default=None)
    p.add_argument("--test_near_peak_mode", type=str, default=None, choices=["around_each", "around_random_true"])
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
