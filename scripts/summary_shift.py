from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricRow:
    rep: str
    method: str
    snr_db: float
    L: int
    accuracy: float
    macro_f1: float


def _read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_metadata(rep_dir: str) -> dict[str, Any]:
    path = os.path.join(rep_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            v = json.load(f)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}


def _collect_deep(rep_dir: str, rep: str) -> tuple[list[MetricRow], list[MetricRow]]:
    md = _load_metadata(rep_dir)
    method = str(md.get("model", "deep"))

    shifted_path = os.path.join(rep_dir, "test_metrics_by_condition.csv")
    in_domain_path = os.path.join(rep_dir, "in_domain_metrics_by_condition.csv")
    if not os.path.exists(shifted_path) or not os.path.exists(in_domain_path):
        return [], []

    shifted_rows: list[MetricRow] = []
    for r in _read_csv(shifted_path):
        shifted_rows.append(
            MetricRow(
                rep=rep,
                method=method,
                snr_db=float(r["snr_db"]),
                L=int(r["L"]),
                accuracy=float(r["accuracy"]),
                macro_f1=float(r["macro_f1"]),
            )
        )

    in_rows: list[MetricRow] = []
    for r in _read_csv(in_domain_path):
        in_rows.append(
            MetricRow(
                rep=rep,
                method=method,
                snr_db=float(r["snr_db"]),
                L=int(r["L"]),
                accuracy=float(r["accuracy"]),
                macro_f1=float(r["macro_f1"]),
            )
        )
    return shifted_rows, in_rows


def _collect_traditional(rep_dir: str, rep: str) -> tuple[list[MetricRow], list[MetricRow]]:
    shifted_path = os.path.join(rep_dir, "traditional_metrics_by_condition.csv")
    in_domain_path = os.path.join(rep_dir, "traditional_in_domain_metrics_by_condition.csv")
    if not os.path.exists(shifted_path) or not os.path.exists(in_domain_path):
        return [], []

    shifted_rows: list[MetricRow] = []
    for r in _read_csv(shifted_path):
        shifted_rows.append(
            MetricRow(
                rep=rep,
                method=str(r["method"]),
                snr_db=float(r["snr_db"]),
                L=int(r["L"]),
                accuracy=float(r["accuracy"]),
                macro_f1=float(r["macro_f1"]),
            )
        )

    in_rows: list[MetricRow] = []
    for r in _read_csv(in_domain_path):
        in_rows.append(
            MetricRow(
                rep=rep,
                method=str(r["method"]),
                snr_db=float(r["snr_db"]),
                L=int(r["L"]),
                accuracy=float(r["accuracy"]),
                macro_f1=float(r["macro_f1"]),
            )
        )
    return shifted_rows, in_rows


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = x.astype(np.float64, copy=False)
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Run root containing rep_*/ or repeat_*/ subdirectories.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = args.run_dir

    rep_dirs = sorted(glob.glob(os.path.join(run_dir, "rep_*")))
    if len(rep_dirs) == 0:
        rep_dirs = sorted(glob.glob(os.path.join(run_dir, "repeat_*")))
    if len(rep_dirs) == 0:
        raise FileNotFoundError(f"No rep_*/repeat_* subdirectories found under: {run_dir}")

    shifted_all: list[MetricRow] = []
    in_domain_all: list[MetricRow] = []
    for rep_dir in rep_dirs:
        rep = os.path.basename(rep_dir)
        s_deep, in_deep = _collect_deep(rep_dir, rep=rep)
        s_trad, in_trad = _collect_traditional(rep_dir, rep=rep)
        shifted_all.extend(s_deep)
        shifted_all.extend(s_trad)
        in_domain_all.extend(in_deep)
        in_domain_all.extend(in_trad)

    # Summary mean/std for shifted test.
    key_to_shift: dict[tuple[str, float, int], list[MetricRow]] = {}
    for r in shifted_all:
        key_to_shift.setdefault((r.method, float(r.snr_db), int(r.L)), []).append(r)

    summary_rows: list[dict[str, Any]] = []
    for (method, snr_db, L), rows in sorted(key_to_shift.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        acc = np.array([r.accuracy for r in rows], dtype=np.float64)
        f1 = np.array([r.macro_f1 for r in rows], dtype=np.float64)
        acc_m, acc_s = _mean_std(acc)
        f1_m, f1_s = _mean_std(f1)
        summary_rows.append(
            {
                "method": method,
                "snr_db": float(snr_db),
                "L": int(L),
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "macro_f1_mean": f1_m,
                "macro_f1_std": f1_s,
                "n": int(len(rows)),
            }
        )

    _write_csv(
        os.path.join(run_dir, "summary_by_condition_mean_std.csv"),
        ["method", "snr_db", "L", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "n"],
        summary_rows,
    )

    # Drop = MacroF1_in_domain - MacroF1_shifted (per rep, per condition, per method).
    key_to_in: dict[tuple[str, str, float, int], MetricRow] = {}
    for r in in_domain_all:
        key_to_in[(r.rep, r.method, float(r.snr_db), int(r.L))] = r

    drop_vals: dict[tuple[str, float, int], list[float]] = {}
    for r in shifted_all:
        k = (r.rep, r.method, float(r.snr_db), int(r.L))
        rin = key_to_in.get(k)
        if rin is None:
            continue
        drop_vals.setdefault((r.method, float(r.snr_db), int(r.L)), []).append(float(rin.macro_f1 - r.macro_f1))

    drop_rows: list[dict[str, Any]] = []
    for (method, snr_db, L), vals in sorted(drop_vals.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        v = np.array(vals, dtype=np.float64)
        m, s = _mean_std(v)
        drop_rows.append(
            {
                "method": method,
                "snr_db": float(snr_db),
                "L": int(L),
                "drop_macro_f1_mean": float(m),
                "drop_macro_f1_std": float(s),
                "n": int(len(vals)),
            }
        )

    _write_csv(
        os.path.join(run_dir, "drop_by_condition_mean_std.csv"),
        ["method", "snr_db", "L", "drop_macro_f1_mean", "drop_macro_f1_std", "n"],
        drop_rows,
    )

    print(f"Saved: {os.path.join(run_dir, 'summary_by_condition_mean_std.csv')}")
    print(f"Saved: {os.path.join(run_dir, 'drop_by_condition_mean_std.csv')}")


if __name__ == "__main__":
    main()

