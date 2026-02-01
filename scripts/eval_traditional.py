"""
eval_traditional.py

传统方法（sklearn）评测脚本：在 toy ROI patch 数据上按 (SNR, L) 条件分别训练/测试，
并输出 metrics csv 与混淆矩阵图片，用于和深度网络做公平对比。

示例：
  py eval_traditional.py --out_dir runs/trad_L1 --L_list 1 --snr_list -15 -10 -5 0 5 10 15 20 --methods svm_rbf knn rf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Subset

from datasets.roi_patch_dataset import apply_border_overrides, apply_near_peak_overrides, make_cfg, make_fixed_condition_dataset
from datasets.toy_generator import ToyGenConfig
from utils.metrics import compute_confusion, compute_metrics
from utils.plots import save_confusion_matrix_png
from utils.seed import seed_everything


CLASS_NAMES = ["2peaks_close", "2peaks_far", "3peaks_line", "3peaks_cluster"]


def _sanitize_tag(x: str) -> str:
    # Keep filenames short and Windows-safe.
    x = x.replace(" ", "")
    x = x.replace(".", "p")
    x = x.replace("+", "")
    return x


def _method_run_name(method: str, handfeat_mode: str) -> str:
    if method.startswith("handfeat") and str(handfeat_mode) != "full":
        return f"{method}__{handfeat_mode}"
    return method


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
            enable_aug=False,
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
            enable_aug=False,
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


def _to_numpy_xy(dataset, batch_size: int, num_workers: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xb, yb in loader:
        # xb: [B,2,H,W] on CPU
        xb_np = xb.reshape(xb.shape[0], -1).numpy().astype(np.float32, copy=False)
        yb_np = yb.numpy().astype(np.int64, copy=False)
        xs.append(xb_np)
        ys.append(yb_np)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _spectral_entropy_2d(img: np.ndarray) -> float:
    p = np.abs(np.fft.fft2(img)) ** 2
    p = p.astype(np.float64)
    s = float(p.sum() + 1e-12)
    p = p / s
    return float(-(p * np.log(p + 1e-12)).sum())


def extract_handcrafted_features(
    x_chw: np.ndarray,
    *,
    mode: Literal["full", "peaks_only", "stats_only"] = "full",
    topk: int = 3,
) -> np.ndarray:
    # x_chw: [2,H,W] (X0, Xhf). Default mode="full" is backward-compatible with the previous
    # handfeat_svm feature vector ordering and values.
    x0 = x_chw[0].astype(np.float32, copy=False)
    xhf = x_chw[1].astype(np.float32, copy=False)
    h, w = x0.shape
    k = int(topk)

    flat = x0.reshape(-1)
    if flat.size >= k:
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
    else:
        idx = np.arange(flat.size)
        idx = idx[np.argsort(flat[idx])[::-1]]
        idx = np.pad(idx, (0, k - idx.size), mode="edge")

    ys = (idx // w).astype(np.float32)
    xs = (idx % w).astype(np.float32)
    vals = flat[idx].astype(np.float32)

    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    ny = (ys - cy) / (cy + 1e-6)
    nx = (xs - cx) / (cx + 1e-6)

    coords = np.stack([ys, xs], axis=1).astype(np.float32)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2) + 1e-12).astype(np.float32)
    d12, d13, d23 = float(d[0, 1]), float(d[0, 2]), float(d[1, 2])
    diag = float(np.sqrt((h - 1) ** 2 + (w - 1) ** 2) + 1e-6)
    d_norm = (np.array([d12, d13, d23], dtype=np.float32) / diag).astype(np.float32)

    # peaks-only features (explicit top-k peak values/coords/geometry)
    peaks_feat = np.concatenate(
        [
            vals,  # k
            np.stack([ny, nx], axis=1).reshape(-1).astype(np.float32),  # 2k
            d_norm.astype(np.float32),  # 3
        ],
        axis=0,
    ).astype(np.float32)

    # stats-only features (no explicit peak coordinates)
    z = x0 - float(x0.min())
    e = (z.astype(np.float64) ** 2).reshape(-1)
    total_e = float(e.sum() + 1e-12)
    topn = max(1, int(round(0.10 * e.size)))
    idxe = np.argpartition(e, -topn)[-topn:]
    conc = float(e[idxe].sum() / total_e)

    ent = _spectral_entropy_2d(x0)
    stats = np.array([float(x0.mean()), float(x0.std()), float(xhf.mean()), float(xhf.std())], dtype=np.float32)
    stats_feat = np.concatenate([np.array([conc, ent], dtype=np.float32), stats], axis=0).astype(np.float32)

    if mode == "peaks_only":
        return peaks_feat
    if mode == "stats_only":
        return stats_feat
    if mode == "full":
        return np.concatenate([peaks_feat, stats_feat], axis=0).astype(np.float32)
    raise ValueError(f"Unknown handfeat mode: {mode}")


def _to_numpy_handfeat_xy(
    dataset,
    batch_size: int,
    num_workers: int,
    *,
    mode: Literal["full", "peaks_only", "stats_only"] = "full",
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xb, yb in loader:
        # xb: [B,2,H,W]
        xb_np = xb.numpy().astype(np.float32, copy=False)
        feats = np.stack([extract_handcrafted_features(xb_np[i], mode=mode) for i in range(xb_np.shape[0])], axis=0)
        xs.append(feats)
        ys.append(yb.numpy().astype(np.int64, copy=False))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class ProtoClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, metric: str = "cosine") -> None:
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric
        self.prototypes_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProtoClassifier":
        classes = np.unique(y).astype(np.int64)
        protos = []
        for c in classes:
            protos.append(X[y == c].mean(axis=0))
        self.classes_ = classes
        self.prototypes_ = np.stack(protos, axis=0).astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.prototypes_ is None or self.classes_ is None:
            raise RuntimeError("ProtoClassifier not fitted")
        P = self.prototypes_
        if self.metric == "euclidean":
            # (x - p)^2 = x^2 + p^2 - 2xp
            x2 = (X**2).sum(axis=1, keepdims=True)
            p2 = (P**2).sum(axis=1, keepdims=True).T
            d2 = x2 + p2 - 2.0 * (X @ P.T)
            idx = np.argmin(d2, axis=1)
            return self.classes_[idx]
        # cosine distance = 1 - cos sim
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        sim = Xn @ Pn.T
        idx = np.argmax(sim, axis=1)
        return self.classes_[idx]


def _build_method(method: str, seed: int) -> Any:
    if method == "svm_rbf":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))])
    if method == "svm_linear":
        return Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(dual=True, max_iter=5000))])
    if method == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])
    if method == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if method == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    if method == "proto":
        return Pipeline([("scaler", StandardScaler()), ("clf", ProtoClassifier(metric="cosine"))])
    if method == "handfeat_svm":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))])
    if method == "handfeat_svm_linear":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear"))])
    if method == "handfeat_svm_rbf":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))])
    raise ValueError(f"Unknown method: {method}")


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _choose_test_subset(ds_test, cond_test_samples: int, seed: int) -> Subset:
    n = len(ds_test)
    k = int(cond_test_samples)
    if k <= 0:
        raise ValueError("--cond_test_samples must be > 0")
    if k >= n:
        return Subset(ds_test, list(range(n)))
    seed32 = int(seed) % (2**32 - 1)
    rng = np.random.default_rng(seed32)
    idx = rng.permutation(n)[:k].tolist()
    return Subset(ds_test, idx)


def _choose_subset(ds, k: int, seed: int) -> Subset:
    n = len(ds)
    k = int(k)
    if k <= 0 or k >= n:
        return Subset(ds, list(range(n)))
    seed32 = int(seed) % (2**32 - 1)
    rng = np.random.default_rng(seed32)
    idx = rng.permutation(n)[:k].tolist()
    return Subset(ds, idx)


@dataclass
class RepeatAggregate:
    method: str
    accuracy: float
    macro_f1: float


def _eval_one_repeat(args: argparse.Namespace, repeat_idx: int, seed: int) -> list[RepeatAggregate]:
    seed_everything(seed)
    use_shift_split = bool(args.train_roi_mode or args.test_roi_mode or args.train_aug_profile or args.test_aug_profile)
    subdir = f"rep_{repeat_idx:02d}" if use_shift_split else f"repeat_{repeat_idx:03d}"
    repeat_dir = os.path.join(args.out_dir, subdir)
    os.makedirs(repeat_dir, exist_ok=True)
    with open(os.path.join(repeat_dir, "run_args.txt"), "w", encoding="utf-8") as f:
        for k, v in {**vars(args), "repeat_idx": repeat_idx, "seed": seed}.items():
            f.write(f"{k}: {v}\n")

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

    with open(os.path.join(repeat_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "repeat_id": int(repeat_idx),
                "seed": int(seed),
                "data_source": str(args.data_source),
                "train_roi_mode": train_roi_mode,
                "test_roi_mode": test_roi_mode,
                "train_profile": train_profile or None,
                "test_profile": test_profile or None,
                "methods": list(args.methods),
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
        print(
            f"[repeat {repeat_idx:02d}] data_source=fdamimo "
            f"train_roi_mode={train_roi_mode} test_roi_mode={test_roi_mode} "
            f"theta_span_deg={float(args.fdamimo_theta_span_deg):g} r_span_m={float(args.fdamimo_r_span_m):g} "
            f"M={int(args.fdamimo_M)} N={int(args.fdamimo_N)} f0={float(args.fdamimo_f0):g} delta_f={float(args.fdamimo_delta_f):g}"
        )

    # Evaluate per (SNR,L) condition. For speed: generate/flatten features once per condition,
    # then fit/predict for each method using the same arrays.
    by_condition_rows: list[dict[str, Any]] = []
    by_condition_rows_in: list[dict[str, Any]] = []
    handfeat_mode = str(args.handfeat_mode)
    need_hand = any(str(m).startswith("handfeat") for m in args.methods)
    run_methods = [_method_run_name(str(m), handfeat_mode) for m in args.methods]
    method_to_y_true_all: dict[str, list[np.ndarray]] = {m: [] for m in run_methods}
    method_to_y_pred_all: dict[str, list[np.ndarray]] = {m: [] for m in run_methods}
    method_to_y_true_all_in: dict[str, list[np.ndarray]] = {m: [] for m in run_methods}
    method_to_y_pred_all_in: dict[str, list[np.ndarray]] = {m: [] for m in run_methods}

    cache_dir = os.path.join(repeat_dir, "cache_npz") if bool(args.cache_condition_npz) else ""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    total_conds = len(args.snr_list) * len(args.L_list)
    cond_idx = 0
    for snr_db in args.snr_list:
        for L in args.L_list:
            cond_idx += 1
            cond_seed = seed + int(float(snr_db) * 100) + int(L) * 10000
            ds_train_full = make_fixed_condition_dataset(
                split="train",
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
            ds_test_full = make_fixed_condition_dataset(
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
            ds_test_in_full = make_fixed_condition_dataset(
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

            ds_train = _choose_subset(ds_train_full, k=args.cond_train_samples, seed=cond_seed + 7)
            ds_test = _choose_test_subset(ds_test_full, cond_test_samples=args.cond_test_samples, seed=cond_seed + 13)
            ds_test_in = _choose_test_subset(ds_test_in_full, cond_test_samples=args.cond_test_samples, seed=cond_seed + 17)

            cache_key = (
                f"ds{str(args.data_source)}_snr{float(snr_db):g}_L{int(L)}_seed{int(seed)}_hand{int(need_hand)}_"
                f"handmode{args.handfeat_mode}_"
                f"fdth{float(args.fdamimo_theta_span_deg):g}_fdr{float(args.fdamimo_r_span_m):g}_fdM{int(args.fdamimo_M)}_fdN{int(args.fdamimo_N)}_"
                f"trm{args.train_occlude_mode}_trbp{args.train_border_prob}_trbf{args.train_border_fill}_"
                f"tem{args.test_occlude_mode}_tebp{args.test_border_prob}_tebf{args.test_border_fill}_"
                f"trss{args.train_border_sat_strength}_tess{args.test_border_sat_strength}_"
                f"trnp{args.train_near_peak_prob}_trnpm{args.train_near_peak_mode}_trnpt{args.train_near_peak_per_true_max}_"
                f"tenp{args.test_near_peak_prob}_tenpm{args.test_near_peak_mode}_tenpt{args.test_near_peak_per_true_max}"
            )
            cache_path = os.path.join(cache_dir, f"{_sanitize_tag(cache_key)}.npz") if cache_dir else ""
            if cache_path and os.path.exists(cache_path):
                z = np.load(cache_path, allow_pickle=False)
                Xtr_flat = z["Xtr_flat"]
                ytr = z["ytr"]
                Xte_flat = z["Xte_flat"]
                yte = z["yte"]
                Xte_in_flat = z["Xte_in_flat"]
                yte_in = z["yte_in"]
                if need_hand:
                    Xtr_hand = z["Xtr_hand"]
                    Xte_hand = z["Xte_hand"]
                    Xte_in_hand = z["Xte_in_hand"]
            else:
                Xtr_flat, ytr = _to_numpy_xy(ds_train, batch_size=args.batch_size, num_workers=args.num_workers)
                Xte_flat, yte = _to_numpy_xy(ds_test, batch_size=args.batch_size, num_workers=args.num_workers)
                Xte_in_flat, yte_in = _to_numpy_xy(ds_test_in, batch_size=args.batch_size, num_workers=args.num_workers)
                if need_hand:
                    Xtr_hand, _ = _to_numpy_handfeat_xy(ds_train, batch_size=args.batch_size, num_workers=args.num_workers, mode=args.handfeat_mode)
                    Xte_hand, _ = _to_numpy_handfeat_xy(ds_test, batch_size=args.batch_size, num_workers=args.num_workers, mode=args.handfeat_mode)
                    Xte_in_hand, _ = _to_numpy_handfeat_xy(ds_test_in, batch_size=args.batch_size, num_workers=args.num_workers, mode=args.handfeat_mode)
                if cache_path:
                    payload: dict[str, Any] = {
                        "Xtr_flat": Xtr_flat,
                        "ytr": ytr,
                        "Xte_flat": Xte_flat,
                        "yte": yte,
                        "Xte_in_flat": Xte_in_flat,
                        "yte_in": yte_in,
                    }
                    if need_hand:
                        payload.update({"Xtr_hand": Xtr_hand, "Xte_hand": Xte_hand, "Xte_in_hand": Xte_in_hand})
                    np.savez_compressed(cache_path, **payload)

            if args.verbose:
                print(
                    f"  [cond {cond_idx:02d}/{total_conds}] SNR={float(snr_db):g}dB L={int(L)} "
                    f"train_n={len(ds_train)} test_n={len(ds_test)} feat_dim={Xtr_flat.shape[1]}"
                )

            for method in args.methods:
                method = str(method)
                run_method = _method_run_name(method, handfeat_mode)
                if method.startswith("handfeat"):
                    Xtr = Xtr_hand
                    Xte = Xte_hand
                    Xte_in = Xte_in_hand
                else:
                    Xtr = Xtr_flat
                    Xte = Xte_flat
                    Xte_in = Xte_in_flat
                clf = _build_method(method, seed=seed)
                clf.fit(Xtr, ytr)
                y_pred = clf.predict(Xte).astype(np.int64, copy=False)
                y_pred_in = clf.predict(Xte_in).astype(np.int64, copy=False)

                m = compute_metrics(yte, y_pred)
                cm = compute_confusion(yte, y_pred, num_classes=args.num_classes)
                m_in = compute_metrics(yte_in, y_pred_in)

                by_condition_rows.append(
                    {
                        "data_source": str(args.data_source),
                        "method": run_method,
                        "snr_db": float(snr_db),
                        "L": int(L),
                        "accuracy": float(m.accuracy),
                        "macro_f1": float(m.macro_f1),
                    }
                )
                by_condition_rows_in.append(
                    {
                        "data_source": str(args.data_source),
                        "method": run_method,
                        "snr_db": float(snr_db),
                        "L": int(L),
                        "accuracy": float(m_in.accuracy),
                        "macro_f1": float(m_in.macro_f1),
                    }
                )

                if not args.no_plots and not args.no_per_condition_plots:
                    tag = f"{_sanitize_tag(run_method)}_snr{float(snr_db):g}_L{int(L)}"
                    save_confusion_matrix_png(
                        cm,
                        CLASS_NAMES,
                        os.path.join(repeat_dir, f"confusion_{tag}_count.png"),
                        normalize=False,
                        title=f"{method} | count | SNR={float(snr_db):g}dB L={int(L)}",
                    )
                    save_confusion_matrix_png(
                        cm,
                        CLASS_NAMES,
                        os.path.join(repeat_dir, f"confusion_{tag}_norm.png"),
                        normalize=True,
                        title=f"{method} | row-norm | SNR={float(snr_db):g}dB L={int(L)}",
                    )

                method_to_y_true_all[run_method].append(yte.astype(np.int64, copy=False))
                method_to_y_pred_all[run_method].append(y_pred)
                method_to_y_true_all_in[run_method].append(yte_in.astype(np.int64, copy=False))
                method_to_y_pred_all_in[run_method].append(y_pred_in)

    aggregates: list[RepeatAggregate] = []
    for method in run_methods:
        y_true_all_np = np.concatenate(method_to_y_true_all[method], axis=0)
        y_pred_all_np = np.concatenate(method_to_y_pred_all[method], axis=0)
        m_all = compute_metrics(y_true_all_np, y_pred_all_np)
        cm_all = compute_confusion(y_true_all_np, y_pred_all_np, num_classes=args.num_classes)

        if not args.no_plots:
            tag_all = f"{_sanitize_tag(method)}_all_conditions"
            save_confusion_matrix_png(
                cm_all,
                CLASS_NAMES,
                os.path.join(repeat_dir, f"confusion_{tag_all}_count.png"),
                normalize=False,
                title=f"{method} | count | all conditions",
            )
            save_confusion_matrix_png(
                cm_all,
                CLASS_NAMES,
                os.path.join(repeat_dir, f"confusion_{tag_all}_norm.png"),
                normalize=True,
                title=f"{method} | row-norm | all conditions",
            )

        aggregates.append(RepeatAggregate(method=method, accuracy=float(m_all.accuracy), macro_f1=float(m_all.macro_f1)))
        print(f"  [all-conditions] {method:10s} acc={float(m_all.accuracy):.4f} macro_f1={float(m_all.macro_f1):.4f}")

        y_true_all_in_np = np.concatenate(method_to_y_true_all_in[method], axis=0)
        y_pred_all_in_np = np.concatenate(method_to_y_pred_all_in[method], axis=0)
        m_all_in = compute_metrics(y_true_all_in_np, y_pred_all_in_np)
        print(f"  [in-domain]     {method:10s} acc={float(m_all_in.accuracy):.4f} macro_f1={float(m_all_in.macro_f1):.4f}")

    _write_csv(
        os.path.join(repeat_dir, "traditional_metrics_by_condition.csv"),
        ["data_source", "method", "snr_db", "L", "accuracy", "macro_f1"],
        by_condition_rows,
    )
    _write_csv(
        os.path.join(repeat_dir, "traditional_in_domain_metrics_by_condition.csv"),
        ["data_source", "method", "snr_db", "L", "accuracy", "macro_f1"],
        by_condition_rows_in,
    )
    _write_csv(
        os.path.join(repeat_dir, "traditional_metrics_all_conditions.csv"),
        ["data_source", "method", "accuracy", "macro_f1"],
        [{"data_source": str(args.data_source), "method": a.method, "accuracy": a.accuracy, "macro_f1": a.macro_f1} for a in aggregates],
    )
    _write_csv(
        os.path.join(repeat_dir, "traditional_in_domain_metrics_all_conditions.csv"),
        ["data_source", "method", "accuracy", "macro_f1"],
        [
            {
                "data_source": str(args.data_source),
                "method": method,
                "accuracy": float(
                    compute_metrics(
                        np.concatenate(method_to_y_true_all_in[method], axis=0),
                        np.concatenate(method_to_y_pred_all_in[method], axis=0),
                    ).accuracy
                ),
                "macro_f1": float(
                    compute_metrics(
                        np.concatenate(method_to_y_true_all_in[method], axis=0),
                        np.concatenate(method_to_y_pred_all_in[method], axis=0),
                    ).macro_f1
                ),
            }
            for method in run_methods
        ],
    )

    return aggregates


def _summarize_mean_std(out_dir: str, all_repeats: list[list[RepeatAggregate]]) -> None:
    # Flatten into per-method arrays
    method_to_acc: dict[str, list[float]] = {}
    method_to_f1: dict[str, list[float]] = {}
    for rep in all_repeats:
        for a in rep:
            method_to_acc.setdefault(a.method, []).append(float(a.accuracy))
            method_to_f1.setdefault(a.method, []).append(float(a.macro_f1))

    rows: list[dict[str, Any]] = []
    for method in sorted(method_to_acc.keys()):
        acc = np.array(method_to_acc[method], dtype=np.float64)
        f1 = np.array(method_to_f1[method], dtype=np.float64)
        rows.append(
            {
                "method": method,
                "acc_mean": float(acc.mean()),
                "acc_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
                "f1_mean": float(f1.mean()),
                "f1_std": float(f1.std(ddof=1) if len(f1) > 1 else 0.0),
                "n": int(len(acc)),
            }
        )
    _write_csv(os.path.join(out_dir, "summary_mean_std.csv"), ["method", "acc_mean", "acc_std", "f1_mean", "f1_std", "n"], rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr_list", type=float, nargs="+", default=[-15, -10, -5, 0, 5, 10, 15, 20])
    p.add_argument("--L_list", type=int, nargs="+", default=[1])
    p.add_argument("--hf_mode", type=str, default="sobel", choices=["laplacian", "sobel"])
    p.add_argument("--normalize", type=str, default="per_sample", choices=["none", "per_sample"])

    p.add_argument("--data_source", type=str, default="toy", choices=["toy", "fdamimo"])
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

    # Near-peak pseudo peaks overrides (default=None => no override).
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
    p.add_argument("--patch_size", type=int, default=41)
    p.add_argument("--total_samples", type=int, default=24000)
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--cond_test_samples", type=int, default=6000)
    p.add_argument(
        "--cond_train_samples",
        type=int,
        default=0,
        help="Per-condition train subsample size (0 = use full train split). Useful to speed up heavy methods (e.g., svm_rbf).",
    )

    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["svm_rbf", "svm_linear", "knn", "rf", "logreg", "proto"],
        choices=[
            "svm_rbf",
            "svm_linear",
            "knn",
            "rf",
            "logreg",
            "proto",
            "handfeat_svm",
            "handfeat_svm_linear",
            "handfeat_svm_rbf",
        ],
    )
    p.add_argument(
        "--handfeat_mode",
        type=str,
        default="full",
        choices=["full", "peaks_only", "stats_only"],
        help="Only affects handfeat_* methods. full keeps backward-compatible feature vector.",
    )
    p.add_argument(
        "--max_methods",
        type=int,
        default=0,
        help="If >0, only run the first N methods from --methods (useful for speed).",
    )

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--no_plots", action="store_true", help="Disable saving all confusion matrix PNGs.")
    p.add_argument(
        "--no_per_condition_plots",
        action="store_true",
        help="Only save all-conditions confusion matrices (skip per-(SNR,L) PNGs).",
    )
    p.add_argument(
        "--cache_condition_npz",
        action="store_true",
        help="Cache per-condition (X_train/y_train/X_test/y_test) arrays to NPZ to avoid regenerating data on re-runs.",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-condition progress.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if int(args.max_methods) > 0:
        args.methods = list(args.methods)[: int(args.max_methods)]

    all_repeats: list[list[RepeatAggregate]] = []
    for i in range(int(args.repeat)):
        seed = int(args.seed) + i * 1000
        print(f"[repeat {i:02d}] seed={seed} out_dir={args.out_dir}")
        all_repeats.append(_eval_one_repeat(args, repeat_idx=i, seed=seed))

    _summarize_mean_std(args.out_dir, all_repeats)
    print(f"Saved summary: {os.path.join(args.out_dir, 'summary_mean_std.csv')}")


if __name__ == "__main__":
    main()
