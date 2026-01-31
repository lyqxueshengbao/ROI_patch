from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset

from datasets.roi_patch_dataset import make_fixed_condition_dataset
from utils.metrics import compute_metrics
from utils.seed import seed_everything


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _choose_subset(ds, k: int, seed: int) -> Subset:
    n = len(ds)
    k = int(k)
    if k <= 0 or k >= n:
        return Subset(ds, list(range(n)))
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    idx = rng.permutation(n)[:k].tolist()
    return Subset(ds, idx)


def _choose_test_subset(ds, cond_test_samples: int, seed: int) -> Subset:
    n = len(ds)
    k = int(cond_test_samples)
    if k <= 0 or k >= n:
        return Subset(ds, list(range(n)))
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    idx = rng.permutation(n)[:k].tolist()
    return Subset(ds, idx)


def _region_stats(a: np.ndarray) -> np.ndarray:
    # mean, std, max, p95
    a = a.astype(np.float32, copy=False)
    return np.array(
        [
            float(a.mean()),
            float(a.std()),
            float(a.max()),
            float(np.quantile(a.astype(np.float64, copy=False), 0.95)),
        ],
        dtype=np.float32,
    )


def _edge_mask(H: int, W: int, b: int) -> np.ndarray:
    b = int(max(b, 1))
    m = np.zeros((H, W), dtype=bool)
    m[:b, :] = True
    m[H - b :, :] = True
    m[:, :b] = True
    m[:, W - b :] = True
    return m


def _sobel_mag(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    pad = 1
    p = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    gx = (
        kx[0, 0] * p[:-2, :-2]
        + kx[0, 1] * p[:-2, 1:-1]
        + kx[0, 2] * p[:-2, 2:]
        + kx[1, 0] * p[1:-1, :-2]
        + kx[1, 1] * p[1:-1, 1:-1]
        + kx[1, 2] * p[1:-1, 2:]
        + kx[2, 0] * p[2:, :-2]
        + kx[2, 1] * p[2:, 1:-1]
        + kx[2, 2] * p[2:, 2:]
    )
    gy = (
        ky[0, 0] * p[:-2, :-2]
        + ky[0, 1] * p[:-2, 1:-1]
        + ky[0, 2] * p[:-2, 2:]
        + ky[1, 0] * p[1:-1, :-2]
        + ky[1, 1] * p[1:-1, 1:-1]
        + ky[1, 2] * p[1:-1, 2:]
        + ky[2, 0] * p[2:, :-2]
        + ky[2, 1] * p[2:, 1:-1]
        + ky[2, 2] * p[2:, 2:]
    )
    return np.sqrt(gx**2 + gy**2 + 1e-12).astype(np.float32)


def extract_edge_features(
    x0: np.ndarray,
    *,
    edge_band: int,
    feat_set: Literal["simple", "extended"],
    eps: float = 1e-6,
) -> np.ndarray:
    x0 = x0.astype(np.float32, copy=False)
    H, W = x0.shape
    b = int(max(int(edge_band), 1))
    b = int(min(b, min(H, W)))

    top = x0[:b, :]
    bottom = x0[H - b :, :]
    left = x0[:, :b]
    right = x0[:, W - b :]

    feats = [
        _region_stats(top),
        _region_stats(bottom),
        _region_stats(left),
        _region_stats(right),
    ]

    mask = _edge_mask(H, W, b)
    edge = x0[mask]
    full_mean = float(x0.mean())
    full_max = float(x0.max())
    edge_mean = float(edge.mean())
    edge_max = float(edge.max())
    feats.append(np.array([edge_mean / (full_mean + eps), edge_max / (full_max + eps)], dtype=np.float32))

    if feat_set == "extended":
        tl = x0[:b, :b]
        tr = x0[:b, W - b :]
        bl = x0[H - b :, :b]
        br = x0[H - b :, W - b :]
        feats.extend([_region_stats(tl), _region_stats(tr), _region_stats(bl), _region_stats(br)])

        g = _sobel_mag(x0)
        gedge = g[mask]
        feats.append(_region_stats(gedge))

    return np.concatenate(feats, axis=0).astype(np.float32)


def _to_numpy_edge_xy(
    dataset,
    *,
    edge_band: int,
    feat_set: Literal["simple", "extended"],
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xb, yb in loader:
        xb_np = xb.numpy().astype(np.float32, copy=False)  # [B,2,H,W]
        x0 = xb_np[:, 0]  # [B,H,W]
        feats = np.stack(
            [extract_edge_features(x0[i], edge_band=edge_band, feat_set=feat_set) for i in range(x0.shape[0])],
            axis=0,
        )
        xs.append(feats)
        ys.append(yb.numpy().astype(np.int64, copy=False))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _build_clf(name: str, seed: int):
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=int(seed))),
            ]
        )
    if name == "linear_svm":
        return Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(random_state=int(seed)))])
    raise ValueError(f"Unknown clf={name}")


@dataclass(frozen=True)
class RepRow:
    rep: int
    snr_db: float
    L: int
    accuracy: float
    macro_f1: float


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = x.astype(np.float64, copy=False)
    if x.size <= 1:
        return float(x.mean()) if x.size else 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=1))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--snr_list", type=float, nargs="+", default=[-15, -10, -5, 0, 5, 10, 15, 20])
    p.add_argument("--L_list", type=int, nargs="+", default=[1])
    p.add_argument("--hf_mode", type=str, default="sobel", choices=["laplacian", "sobel"])
    p.add_argument("--normalize", type=str, default="per_sample", choices=["none", "per_sample"])

    p.add_argument("--train_roi_mode", type=str, default="oracle", choices=["oracle", "pipeline"])
    p.add_argument("--test_roi_mode", type=str, default="pipeline", choices=["oracle", "pipeline"])
    p.add_argument("--train_aug_profile", type=str, default="oracle", choices=["oracle", "pipeline"])
    p.add_argument("--test_aug_profile", type=str, default="pipeline", choices=["oracle", "pipeline"])

    p.add_argument("--patch_size", type=int, default=41)
    p.add_argument("--total_samples", type=int, default=24000)
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument("--cond_test_samples", type=int, default=6000)
    p.add_argument("--cond_train_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--edge_band", type=int, default=6)
    p.add_argument("--edge_feat_set", type=str, default="simple", choices=["simple", "extended"])
    p.add_argument("--clf", type=str, default="logreg", choices=["logreg", "linear_svm"])

    # Border/occlusion overrides (optional) - apply to train/test data generation
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

    return p


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rep_rows: list[RepRow] = []
    per_rep_paths: list[str] = []

    for rep in range(int(args.repeat)):
        seed = int(args.seed) + rep * 1000
        seed_everything(seed)
        rep_dir = os.path.join(args.out_dir, f"repeat_{rep:03d}")
        os.makedirs(rep_dir, exist_ok=True)
        per_rep_paths.append(rep_dir)

        with open(os.path.join(rep_dir, "run_args.txt"), "w", encoding="utf-8") as f:
            for k, v in {**vars(args), "repeat_idx": rep, "seed": seed}.items():
                f.write(f"{k}: {v}\n")
        with open(os.path.join(rep_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "repeat_idx": int(rep),
                    "seed": int(seed),
                    "edge_band": int(args.edge_band),
                    "edge_feat_set": str(args.edge_feat_set),
                    "clf": str(args.clf),
                    "train_roi_mode": str(args.train_roi_mode),
                    "test_roi_mode": str(args.test_roi_mode),
                    "train_profile": str(args.train_aug_profile),
                    "test_profile": str(args.test_aug_profile),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        rows: list[dict[str, Any]] = []
        for snr_db in args.snr_list:
            for L in args.L_list:
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
                    roi_mode=str(args.train_roi_mode),  # type: ignore[arg-type]
                    aug_profile=str(args.train_aug_profile),
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
                    roi_mode=str(args.test_roi_mode),  # type: ignore[arg-type]
                    aug_profile=str(args.test_aug_profile),
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
                )

                ds_train = _choose_subset(ds_train_full, k=args.cond_train_samples, seed=cond_seed + 7)
                ds_test = _choose_test_subset(ds_test_full, cond_test_samples=args.cond_test_samples, seed=cond_seed + 13)

                Xtr, ytr = _to_numpy_edge_xy(
                    ds_train,
                    edge_band=args.edge_band,
                    feat_set=args.edge_feat_set,  # type: ignore[arg-type]
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                Xte, yte = _to_numpy_edge_xy(
                    ds_test,
                    edge_band=args.edge_band,
                    feat_set=args.edge_feat_set,  # type: ignore[arg-type]
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

                clf = _build_clf(str(args.clf), seed=seed)
                clf.fit(Xtr, ytr)
                y_pred = clf.predict(Xte).astype(np.int64, copy=False)
                m = compute_metrics(yte, y_pred)

                rows.append(
                    {
                        "snr_db": float(snr_db),
                        "L": int(L),
                        "accuracy": float(m.accuracy),
                        "macro_f1": float(m.macro_f1),
                        "edge_band": int(args.edge_band),
                        "edge_feat_set": str(args.edge_feat_set),
                        "clf": str(args.clf),
                        "train_roi_mode": str(args.train_roi_mode),
                        "test_roi_mode": str(args.test_roi_mode),
                        "train_profile": str(args.train_aug_profile),
                        "test_profile": str(args.test_aug_profile),
                        "repeat_idx": int(rep),
                    }
                )
                all_rep_rows.append(RepRow(rep=rep, snr_db=float(snr_db), L=int(L), accuracy=float(m.accuracy), macro_f1=float(m.macro_f1)))

        _write_csv(
            os.path.join(rep_dir, "edge_shortcut_metrics_by_condition.csv"),
            [
                "snr_db",
                "L",
                "accuracy",
                "macro_f1",
                "edge_band",
                "edge_feat_set",
                "clf",
                "train_roi_mode",
                "test_roi_mode",
                "train_profile",
                "test_profile",
                "repeat_idx",
            ],
            rows,
        )

    # Summary by condition (mean/std across repeats)
    key_to_rows: dict[tuple[float, int], list[RepRow]] = {}
    for r in all_rep_rows:
        key_to_rows.setdefault((float(r.snr_db), int(r.L)), []).append(r)

    out_rows: list[dict[str, Any]] = []
    for (snr_db, L), rs in sorted(key_to_rows.items(), key=lambda x: (x[0][0], x[0][1])):
        acc = np.array([x.accuracy for x in rs], dtype=np.float64)
        f1 = np.array([x.macro_f1 for x in rs], dtype=np.float64)
        acc_m, acc_s = _mean_std(acc)
        f1_m, f1_s = _mean_std(f1)
        out_rows.append(
            {
                "snr_db": float(snr_db),
                "L": int(L),
                "acc_mean": acc_m,
                "acc_std": acc_s,
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "n": int(len(rs)),
                "edge_band": int(args.edge_band),
                "edge_feat_set": str(args.edge_feat_set),
                "clf": str(args.clf),
                "train_roi_mode": str(args.train_roi_mode),
                "test_roi_mode": str(args.test_roi_mode),
                "train_profile": str(args.train_aug_profile),
                "test_profile": str(args.test_aug_profile),
            }
        )

    _write_csv(
        os.path.join(args.out_dir, "edge_shortcut_summary_by_condition_mean_std.csv"),
        [
            "snr_db",
            "L",
            "acc_mean",
            "acc_std",
            "f1_mean",
            "f1_std",
            "n",
            "edge_band",
            "edge_feat_set",
            "clf",
            "train_roi_mode",
            "test_roi_mode",
            "train_profile",
            "test_profile",
        ],
        out_rows,
    )

    # Overall summary (mean/std across all conditions per repeat, then across repeats)
    rep_to_vals: dict[int, list[RepRow]] = {}
    for r in all_rep_rows:
        rep_to_vals.setdefault(int(r.rep), []).append(r)

    rep_acc: list[float] = []
    rep_f1: list[float] = []
    for rep, rs in sorted(rep_to_vals.items()):
        rep_acc.append(float(np.mean([x.accuracy for x in rs])))
        rep_f1.append(float(np.mean([x.macro_f1 for x in rs])))

    acc_m, acc_s = _mean_std(np.array(rep_acc, dtype=np.float64))
    f1_m, f1_s = _mean_std(np.array(rep_f1, dtype=np.float64))
    _write_csv(
        os.path.join(args.out_dir, "edge_shortcut_summary_mean_std.csv"),
        ["metric", "mean", "std", "n", "edge_band", "edge_feat_set", "clf", "train_roi_mode", "test_roi_mode", "train_profile", "test_profile"],
        [
            {
                "metric": "accuracy_mean_over_conditions",
                "mean": acc_m,
                "std": acc_s,
                "n": int(len(rep_acc)),
                "edge_band": int(args.edge_band),
                "edge_feat_set": str(args.edge_feat_set),
                "clf": str(args.clf),
                "train_roi_mode": str(args.train_roi_mode),
                "test_roi_mode": str(args.test_roi_mode),
                "train_profile": str(args.train_aug_profile),
                "test_profile": str(args.test_aug_profile),
            },
            {
                "metric": "macro_f1_mean_over_conditions",
                "mean": f1_m,
                "std": f1_s,
                "n": int(len(rep_f1)),
                "edge_band": int(args.edge_band),
                "edge_feat_set": str(args.edge_feat_set),
                "clf": str(args.clf),
                "train_roi_mode": str(args.train_roi_mode),
                "test_roi_mode": str(args.test_roi_mode),
                "train_profile": str(args.train_aug_profile),
                "test_profile": str(args.test_aug_profile),
            },
        ],
    )

    print(f"Saved: {os.path.join(args.out_dir, 'edge_shortcut_summary_by_condition_mean_std.csv')}")
    print(f"Saved: {os.path.join(args.out_dir, 'edge_shortcut_summary_mean_std.csv')}")


if __name__ == "__main__":
    main()

