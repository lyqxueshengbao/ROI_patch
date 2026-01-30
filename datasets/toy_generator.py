from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ToyGenConfig:
    height: int = 41
    width: int = 41
    eps: float = 1e-6
    snr_db: float = 5.0
    L: int = 1
    hf_mode: Literal["laplacian", "sobel"] = "laplacian"
    normalize: Literal["none", "per_sample"] = "per_sample"
    enable_aug: bool = True
    d_close: float = 6.0
    d_far: float = 14.0
    line_max_dist_to_line: float = 1.3
    cluster_radius: float = 4.5
    margin: int = 5


def _meshgrid_xy(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy, xx


def _render_gaussian_peaks(
    height: int,
    width: int,
    centers_yx: np.ndarray,
    amplitudes: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    yy, xx = _meshgrid_xy(height, width)
    x = np.zeros((height, width), dtype=np.float32)
    for (cy, cx), a, s in zip(centers_yx, amplitudes, sigmas, strict=True):
        dy2 = (yy - cy) ** 2
        dx2 = (xx - cx) ** 2
        x += a * np.exp(-(dx2 + dy2) / (2.0 * (s**2)))
    return x


def _pairwise_dists(points_yx: np.ndarray) -> np.ndarray:
    diffs = points_yx[:, None, :] - points_yx[None, :, :]
    d2 = (diffs[..., 0] ** 2 + diffs[..., 1] ** 2).astype(np.float32)
    return np.sqrt(d2 + 1e-12, dtype=np.float32)


def _dist_point_to_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    denom = float(np.linalg.norm(ab) + 1e-12)
    area = float(np.abs(ab[1] * ap[0] - ab[0] * ap[1]))
    return area / denom


def _sample_point(rng: np.random.Generator, height: int, width: int, margin: int) -> np.ndarray:
    y = rng.uniform(margin, height - 1 - margin)
    x = rng.uniform(margin, width - 1 - margin)
    return np.array([y, x], dtype=np.float32)


def _rotate_flip(points_yx: np.ndarray, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(0, 4))
    do_h = bool(rng.integers(0, 2))
    do_v = bool(rng.integers(0, 2))
    p = points_yx.copy()
    h, w = height, width
    for _ in range(k):
        p = np.stack([p[:, 1], (w - 1) - p[:, 0]], axis=1).astype(np.float32)
        h, w = w, h
    if do_h:
        p[:, 1] = (w - 1) - p[:, 1]
    if do_v:
        p[:, 0] = (h - 1) - p[:, 0]
    return p


def _make_class_centers(label: int, cfg: ToyGenConfig, rng: np.random.Generator) -> np.ndarray:
    h, w = cfg.height, cfg.width
    margin = cfg.margin
    max_tries = 200

    if label in (0, 1):
        for _ in range(max_tries):
            p1 = _sample_point(rng, h, w, margin)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            if label == 0:
                # Keep a safety margin so post-jitter still stays "close".
                dist = rng.uniform(2.0, max(2.2, cfg.d_close - 1.2))
            else:
                max_dist = min(float(min(h, w)) * 0.9, 26.0)
                # Keep a safety margin so post-jitter still stays "far".
                dist = rng.uniform(cfg.d_far + 1.2, max(cfg.d_far + 1.6, max_dist))
            delta = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * dist
            p2 = p1 + delta
            if (
                margin <= p2[0] <= (h - 1 - margin)
                and margin <= p2[1] <= (w - 1 - margin)
            ):
                centers = np.stack([p1, p2], axis=0).astype(np.float32)
                d = float(_pairwise_dists(centers)[0, 1])
                if label == 0 and d < cfg.d_close:
                    return centers
                if label == 1 and d > cfg.d_far:
                    return centers
        raise RuntimeError("Failed to sample 2-peak geometry.")

    if label == 2:
        for _ in range(max_tries):
            a = _sample_point(rng, h, w, margin)
            b = _sample_point(rng, h, w, margin)
            if float(np.linalg.norm(b - a)) < 10.0:
                continue
            t = rng.uniform(0.2, 0.8)
            base = a + t * (b - a)
            ab = b - a
            ab_norm = ab / (np.linalg.norm(ab) + 1e-12)
            perp = np.array([ab_norm[1], -ab_norm[0]], dtype=np.float32)
            offset = perp * rng.normal(0.0, cfg.line_max_dist_to_line * 0.4)
            c = base + offset
            if not (
                margin <= c[0] <= (h - 1 - margin) and margin <= c[1] <= (w - 1 - margin)
            ):
                continue
            centers = np.stack([a, b, c], axis=0).astype(np.float32)
            d_line = _dist_point_to_line(c, a, b)
            if d_line <= cfg.line_max_dist_to_line:
                return centers
        raise RuntimeError("Failed to sample 3-peak line geometry.")

    if label == 3:
        for _ in range(max_tries):
            center = _sample_point(rng, h, w, margin)
            angles = rng.uniform(0.0, 2.0 * np.pi, size=(3,))
            radii = rng.uniform(0.0, cfg.cluster_radius, size=(3,))
            offsets = np.stack([np.sin(angles) * radii, np.cos(angles) * radii], axis=1).astype(
                np.float32
            )
            points = center[None, :] + offsets
            if np.any(points[:, 0] < margin) or np.any(points[:, 0] > (h - 1 - margin)):
                continue
            if np.any(points[:, 1] < margin) or np.any(points[:, 1] > (w - 1 - margin)):
                continue
            d = _pairwise_dists(points)
            max_pair = float(np.max(d[np.triu_indices(3, 1)]))
            if max_pair <= (cfg.cluster_radius * 1.4):
                return points.astype(np.float32)
        raise RuntimeError("Failed to sample 3-peak cluster geometry.")

    raise ValueError(f"Unknown label={label}")


def _add_noise_snr(x_signal: np.ndarray, snr_db: float, L: int, rng: np.random.Generator) -> np.ndarray:
    if L < 1:
        raise ValueError("L must be >= 1")
    sig_power = float(np.mean(x_signal**2) + 1e-12)
    snr_lin = float(10.0 ** (snr_db / 10.0))
    noise_power = sig_power / max(snr_lin, 1e-12)
    noise_std = float(np.sqrt(noise_power))
    noise = rng.normal(0.0, noise_std, size=(L, *x_signal.shape)).astype(np.float32)
    noise = noise.mean(axis=0)
    return x_signal + noise


def _laplacian(img: np.ndarray) -> np.ndarray:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = 1
    p = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    out = (
        k[0, 0] * p[:-2, :-2]
        + k[0, 1] * p[:-2, 1:-1]
        + k[0, 2] * p[:-2, 2:]
        + k[1, 0] * p[1:-1, :-2]
        + k[1, 1] * p[1:-1, 1:-1]
        + k[1, 2] * p[1:-1, 2:]
        + k[2, 0] * p[2:, :-2]
        + k[2, 1] * p[2:, 1:-1]
        + k[2, 2] * p[2:, 2:]
    )
    return out.astype(np.float32)


def _sobel_mag(img: np.ndarray) -> np.ndarray:
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


def generate_sample(label: int, cfg: ToyGenConfig, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    def valid_geometry(points: np.ndarray) -> bool:
        if label in (0, 1):
            d = float(_pairwise_dists(points)[0, 1])
            return (d < cfg.d_close) if label == 0 else (d > cfg.d_far)
        if label == 2:
            # define line by the farthest pair; the remaining point should be close to this line
            d = _pairwise_dists(points)
            tri = np.triu_indices(3, 1)
            pair_idx = int(np.argmax(d[tri]))
            pairs = [(0, 1), (0, 2), (1, 2)]
            i, j = pairs[pair_idx]
            k = ({0, 1, 2} - {i, j}).pop()
            return _dist_point_to_line(points[k], points[i], points[j]) <= cfg.line_max_dist_to_line
        if label == 3:
            d = _pairwise_dists(points)
            max_pair = float(np.max(d[np.triu_indices(3, 1)]))
            return max_pair <= (cfg.cluster_radius * 1.7)
        return False

    centers = None
    for _ in range(50):
        c = _make_class_centers(label, cfg, rng)
        if cfg.enable_aug:
            c = _rotate_flip(c, cfg.height, cfg.width, rng)
        c = c + rng.normal(0.0, 0.25, size=c.shape).astype(np.float32)
        if valid_geometry(c):
            centers = c
            break
    if centers is None:
        centers = _make_class_centers(label, cfg, rng)

    n_peaks = centers.shape[0]
    amplitudes = rng.uniform(0.7, 1.4, size=(n_peaks,)).astype(np.float32) * rng.uniform(0.8, 1.2)
    sigmas = rng.uniform(1.1, 2.4, size=(n_peaks,)).astype(np.float32)

    x_signal = _render_gaussian_peaks(cfg.height, cfg.width, centers, amplitudes, sigmas)
    x = _add_noise_snr(x_signal, cfg.snr_db, cfg.L, rng)
    x = np.clip(x, 0.0, None).astype(np.float32)

    x0 = np.log(x + cfg.eps).astype(np.float32)
    if cfg.hf_mode == "laplacian":
        xhf = _laplacian(x0)
    elif cfg.hf_mode == "sobel":
        xhf = _sobel_mag(x0)
    else:
        raise ValueError(f"Unknown hf_mode={cfg.hf_mode}")

    x_stack = np.stack([x0, xhf], axis=0).astype(np.float32)
    if cfg.normalize == "per_sample":
        mean = x_stack.mean(axis=(1, 2), keepdims=True).astype(np.float32)
        std = x_stack.std(axis=(1, 2), keepdims=True).astype(np.float32) + 1e-6
        x_stack = (x_stack - mean) / std
    return x_stack, int(label)
