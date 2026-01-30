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
import os
from dataclasses import dataclass
from typing import Any, Iterable

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

from datasets.roi_patch_dataset import make_fixed_condition_dataset
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
        return Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(dual="auto", max_iter=5000))])
    if method == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))])
    if method == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if method == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    if method == "proto":
        return Pipeline([("scaler", StandardScaler()), ("clf", ProtoClassifier(metric="cosine"))])
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
    repeat_dir = os.path.join(args.out_dir, f"repeat_{repeat_idx:03d}")
    os.makedirs(repeat_dir, exist_ok=True)
    with open(os.path.join(repeat_dir, "run_args.txt"), "w", encoding="utf-8") as f:
        for k, v in {**vars(args), "repeat_idx": repeat_idx, "seed": seed}.items():
            f.write(f"{k}: {v}\n")

    # Evaluate per (SNR,L) condition. For speed: generate/flatten features once per condition,
    # then fit/predict for each method using the same arrays.
    by_condition_rows: list[dict[str, Any]] = []
    method_to_y_true_all: dict[str, list[np.ndarray]] = {m: [] for m in args.methods}
    method_to_y_pred_all: dict[str, list[np.ndarray]] = {m: [] for m in args.methods}

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
            )

            ds_train = _choose_subset(ds_train_full, k=args.cond_train_samples, seed=cond_seed + 7)
            ds_test = _choose_test_subset(ds_test_full, cond_test_samples=args.cond_test_samples, seed=cond_seed + 13)

            Xtr, ytr = _to_numpy_xy(ds_train, batch_size=args.batch_size, num_workers=args.num_workers)
            Xte, yte = _to_numpy_xy(ds_test, batch_size=args.batch_size, num_workers=args.num_workers)

            if args.verbose:
                print(
                    f"  [cond {cond_idx:02d}/{total_conds}] SNR={float(snr_db):g}dB L={int(L)} "
                    f"train_n={len(ds_train)} test_n={len(ds_test)} feat_dim={Xtr.shape[1]}"
                )

            for method in args.methods:
                clf = _build_method(method, seed=seed)
                clf.fit(Xtr, ytr)
                y_pred = clf.predict(Xte).astype(np.int64, copy=False)

                m = compute_metrics(yte, y_pred)
                cm = compute_confusion(yte, y_pred, num_classes=args.num_classes)

                by_condition_rows.append(
                    {
                        "method": method,
                        "snr_db": float(snr_db),
                        "L": int(L),
                        "accuracy": float(m.accuracy),
                        "macro_f1": float(m.macro_f1),
                    }
                )

                if not args.no_plots and not args.no_per_condition_plots:
                    tag = f"{_sanitize_tag(method)}_snr{float(snr_db):g}_L{int(L)}"
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

                method_to_y_true_all[method].append(yte.astype(np.int64, copy=False))
                method_to_y_pred_all[method].append(y_pred)

    aggregates: list[RepeatAggregate] = []
    for method in args.methods:
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

    _write_csv(
        os.path.join(repeat_dir, "traditional_metrics_by_condition.csv"),
        ["method", "snr_db", "L", "accuracy", "macro_f1"],
        by_condition_rows,
    )
    _write_csv(
        os.path.join(repeat_dir, "traditional_metrics_all_conditions.csv"),
        ["method", "accuracy", "macro_f1"],
        [{"method": a.method, "accuracy": a.accuracy, "macro_f1": a.macro_f1} for a in aggregates],
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
        choices=["svm_rbf", "svm_linear", "knn", "rf", "logreg", "proto"],
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
    p.add_argument("--verbose", action="store_true", help="Print per-condition progress.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_repeats: list[list[RepeatAggregate]] = []
    for i in range(int(args.repeat)):
        seed = int(args.seed) + i * 1000
        print(f"[repeat {i:02d}] seed={seed} out_dir={args.out_dir}")
        all_repeats.append(_eval_one_repeat(args, repeat_idx=i, seed=seed))

    _summarize_mean_std(args.out_dir, all_repeats)
    print(f"Saved summary: {os.path.join(args.out_dir, 'summary_mean_std.csv')}")


if __name__ == "__main__":
    main()
