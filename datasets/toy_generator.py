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

    # ROI crop/align error model.
    #
    # In "pipeline" mode, the underlying structure is generated around the (oracle-ish) true center,
    # then a ROI patch is extracted with a center offset caused by detection/localization errors.
    # This is implemented as a global subpixel shift on the final noisy image `x` (after noise,
    # before log). The offset scale (sigma) increases as SNR decreases and as L gets smaller.
    roi_mode: Literal["oracle", "pipeline"] = "oracle"
    enable_roi_shift: bool = True
    center_sigma_oracle: float = 1.0
    center_sigma_min: float = 1.5
    center_sigma_max: float = 6.0
    center_trunc: float = 8.0  # truncate offsets to +/- this many pixels
    center_snr_low: float = -15.0
    center_snr_high: float = 20.0
    center_L_min: int = 1
    center_L_max: int = 16

    # Stronger intra-class perturbations
    pseudo_peak_prob: float = 0.35
    pseudo_peak_max: int = 2
    warp_prob: float = 0.25
    warp_strength: float = 0.6
    corr_noise_prob: float = 0.25
    corr_strength: float = 0.6

    # Occlusion / truncation.
    #
    # v2 default uses "border" cut to mimic ROI crop errors: due to pipeline detection/localization
    # errors, the true structure can be close to the ROI boundary and gets truncated ("贴边被裁掉").
    # Filling with zeros before log creates a learnable high-frequency boundary cue for CNNs,
    # while peak-geometry hand-crafted features become less stable.
    enable_occlude: bool = True
    occlude_mode: Literal["none", "block", "border"] = "border"

    # Border-cut (preferred v2)
    border_prob: float = 0.35
    border_sides: Literal["one", "two", "rand12"] = "rand12"
    border_min: int = 4
    border_max: int = 14
    border_fill: Literal["zero", "min", "mean"] = "zero"

    # Block cutout (legacy/v1; keep available but default weaker)
    occlude_prob: float = 0.05
    occlude_max_blocks: int = 1
    occlude_min_size: int = 4
    occlude_max_size: int = 10
    occlude_fill: Literal["zero", "mean"] = "mean"
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
    for (cy, cx), a, s in zip(centers_yx, amplitudes, sigmas):
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


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _center_sigma(cfg: ToyGenConfig) -> float:
    if cfg.roi_mode == "oracle":
        return float(cfg.center_sigma_oracle)
    if cfg.roi_mode != "pipeline":
        raise ValueError(f"Unknown roi_mode={cfg.roi_mode}")

    snr_norm = _clip01(
        (float(cfg.snr_db) - float(cfg.center_snr_low))
        / (float(cfg.center_snr_high) - float(cfg.center_snr_low) + 1e-12)
    )
    Lmin = max(int(cfg.center_L_min), 1)
    Lmax = max(int(cfg.center_L_max), Lmin + 1)
    Lc = max(int(cfg.L), 1)
    L_norm = _clip01(
        (np.log2(Lc) - np.log2(Lmin)) / (np.log2(Lmax) - np.log2(Lmin) + 1e-12)
    )
    mix = 0.6 * snr_norm + 0.4 * float(L_norm)
    sigma = float(cfg.center_sigma_max) - (float(cfg.center_sigma_max) - float(cfg.center_sigma_min)) * float(mix)
    return float(np.clip(sigma, float(cfg.center_sigma_min), float(cfg.center_sigma_max)))


def _sample_center(cfg: ToyGenConfig, rng: np.random.Generator) -> np.ndarray:
    h, w, m = cfg.height, cfg.width, cfg.margin
    cy0 = (h - 1) * 0.5
    cx0 = (w - 1) * 0.5
    y = float(np.clip(cy0, m, h - 1 - m))
    x = float(np.clip(cx0, m, w - 1 - m))
    return np.array([y, x], dtype=np.float32)


def _sample_uniform_point(cfg: ToyGenConfig, rng: np.random.Generator) -> np.ndarray:
    h, w, m = cfg.height, cfg.width, cfg.margin
    y = rng.uniform(m, h - 1 - m)
    x = rng.uniform(m, w - 1 - m)
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
            center = _sample_center(cfg, rng)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            if label == 0:
                dist = rng.uniform(2.0, max(2.2, cfg.d_close - 1.2))
            else:
                max_dist = min(float(min(h, w)) * 0.9, 26.0)
                dist = rng.uniform(cfg.d_far + 1.2, max(cfg.d_far + 1.6, max_dist))

            half = 0.5 * dist
            delta = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32) * half
            p1 = center - delta
            p2 = center + delta
            if (
                margin <= p1[0] <= (h - 1 - margin)
                and margin <= p1[1] <= (w - 1 - margin)
                and margin <= p2[0] <= (h - 1 - margin)
                and margin <= p2[1] <= (w - 1 - margin)
            ):
                return np.stack([p1, p2], axis=0).astype(np.float32)
        raise RuntimeError("Failed to sample 2-peak geometry.")

    if label == 2:
        for _ in range(max_tries):
            center = _sample_center(cfg, rng)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
            step = rng.uniform(4.8, 8.5)

            a = center - direction * step
            b = center + direction * step
            c = center + direction * rng.uniform(-0.6, 0.6) * step
            perp = np.array([direction[1], -direction[0]], dtype=np.float32)
            c = c + perp * rng.normal(0.0, cfg.line_max_dist_to_line * 0.35)

            pts = np.stack([a, b, c], axis=0).astype(np.float32)
            if np.any(pts[:, 0] < margin) or np.any(pts[:, 0] > (h - 1 - margin)):
                continue
            if np.any(pts[:, 1] < margin) or np.any(pts[:, 1] > (w - 1 - margin)):
                continue
            if _dist_point_to_line(c, a, b) <= cfg.line_max_dist_to_line:
                return pts
        raise RuntimeError("Failed to sample 3-peak line geometry.")

    if label == 3:
        for _ in range(max_tries):
            center = _sample_center(cfg, rng)
            angles = rng.uniform(0.0, 2.0 * np.pi, size=(3,))
            radii = rng.uniform(0.0, cfg.cluster_radius, size=(3,))
            offsets = np.stack([np.sin(angles) * radii, np.cos(angles) * radii], axis=1).astype(np.float32)
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


def _mean_filter_3x3(img: np.ndarray) -> np.ndarray:
    pad = 1
    p = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    out = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    ) / 9.0
    return out.astype(np.float32)


def _shift_bilinear(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    # 2D float32 image, subpixel bilinear sampling with boundary clamp.
    h, w = img.shape
    if abs(float(dy)) <= 1e-12 and abs(float(dx)) <= 1e-12:
        return img.astype(np.float32, copy=False)

    yy, xx = _meshgrid_xy(h, w)
    yy = np.clip(yy + float(dy), 0.0, h - 1.0)
    xx = np.clip(xx + float(dx), 0.0, w - 1.0)

    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (yy - y0).astype(np.float32)
    wx = (xx - x0).astype(np.float32)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    out = (1 - wy) * ((1 - wx) * Ia + wx * Ib) + wy * ((1 - wx) * Ic + wx * Id)
    return out.astype(np.float32)


def _sample_roi_offset(cfg: ToyGenConfig, rng: np.random.Generator) -> tuple[float, float]:
    if cfg.roi_mode == "oracle":
        sigma = float(cfg.center_sigma_oracle)
    elif cfg.roi_mode == "pipeline":
        sigma = _center_sigma(cfg)
    else:
        raise ValueError(f"Unknown roi_mode={cfg.roi_mode}")

    trunc = float(cfg.center_trunc)
    dy = float(np.clip(rng.normal(0.0, sigma), -trunc, trunc))
    dx = float(np.clip(rng.normal(0.0, sigma), -trunc, trunc))
    return dy, dx


def _apply_border_cut(x: np.ndarray, cfg: ToyGenConfig, rng: np.random.Generator) -> np.ndarray:
    # Border truncation (ROI crop error): fill a border band with zeros/mean/min.
    # Apply on log-pre intensity map to emulate measurement/cropping information loss.
    if rng.random() >= float(cfg.border_prob):
        return x.astype(np.float32, copy=False)

    H, W = x.shape
    bmin = int(max(int(cfg.border_min), 1))
    bmax = int(max(int(cfg.border_max), bmin))
    bw = int(rng.integers(bmin, bmax + 1))
    bw = int(min(bw, min(H, W)))

    fill_mode = str(cfg.border_fill)
    if fill_mode == "mean":
        fill_val = float(x.mean())
    elif fill_mode == "min":
        fill_val = float(x.min())
    else:
        fill_val = 0.0

    sides_mode = str(cfg.border_sides)
    if sides_mode == "rand12":
        sides_mode = "one" if (rng.random() < 0.5) else "two"

    out = x.astype(np.float32, copy=True)

    if sides_mode == "one":
        side = str(rng.choice(["top", "bottom", "left", "right"]))
        if side == "top":
            out[:bw, :] = np.float32(fill_val)
        elif side == "bottom":
            out[H - bw :, :] = np.float32(fill_val)
        elif side == "left":
            out[:, :bw] = np.float32(fill_val)
        else:
            out[:, W - bw :] = np.float32(fill_val)
        return out.astype(np.float32, copy=False)

    if sides_mode == "two":
        # Two opposite sides: either top+bottom or left+right.
        if bool(rng.integers(0, 2)):
            out[:bw, :] = np.float32(fill_val)
            out[H - bw :, :] = np.float32(fill_val)
        else:
            out[:, :bw] = np.float32(fill_val)
            out[:, W - bw :] = np.float32(fill_val)
        return out.astype(np.float32, copy=False)

    raise ValueError(f"Unknown border_sides={cfg.border_sides}")


def _apply_block_occlusion(x: np.ndarray, cfg: ToyGenConfig, rng: np.random.Generator) -> np.ndarray:
    # Legacy v1 random block cutout inside the patch.
    if rng.random() >= float(cfg.occlude_prob):
        return x.astype(np.float32, copy=False)

    H, W = x.shape
    max_blocks = int(max(int(cfg.occlude_max_blocks), 1))
    blocks = int(rng.integers(1, max_blocks + 1))

    hmin = int(max(int(cfg.occlude_min_size), 1))
    hmax = int(max(int(cfg.occlude_max_size), hmin))
    fill_mode = str(cfg.occlude_fill)
    fill_val = float(x.mean()) if fill_mode == "mean" else 0.0

    out = x.astype(np.float32, copy=True)
    for _ in range(blocks):
        bh = int(rng.integers(hmin, hmax + 1))
        bw = int(rng.integers(hmin, hmax + 1))
        bh = int(min(bh, H))
        bw = int(min(bw, W))
        y0 = int(rng.integers(0, H - bh + 1))
        x0 = int(rng.integers(0, W - bw + 1))
        out[y0 : y0 + bh, x0 : x0 + bw] = np.float32(fill_val)
    return out.astype(np.float32, copy=False)


def _apply_occlusion(x: np.ndarray, cfg: ToyGenConfig, rng: np.random.Generator) -> np.ndarray:
    if not bool(cfg.enable_occlude):
        return x.astype(np.float32, copy=False)

    mode = str(cfg.occlude_mode)
    if mode == "none":
        return x.astype(np.float32, copy=False)
    if mode == "border":
        return _apply_border_cut(x, cfg, rng)
    if mode == "block":
        return _apply_block_occlusion(x, cfg, rng)
    raise ValueError(f"Unknown occlude_mode={cfg.occlude_mode}")


def _warp_image(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    # Lightweight "mismatch": (a) mild subpixel translation; (b) mild blur/sharpen.
    s = float(max(strength, 0.0))
    if s <= 1e-6:
        return img

    if bool(rng.integers(0, 2)):
        dy = float(rng.normal(0.0, s))
        dx = float(rng.normal(0.0, s))
        return _shift_bilinear(img, dy=dy, dx=dx)

    blur = _mean_filter_3x3(img)
    alpha = float(np.clip(s, 0.0, 1.0))
    if bool(rng.integers(0, 2)):
        return ((1 - alpha) * img + alpha * blur).astype(np.float32)
    return (img + alpha * (img - blur)).astype(np.float32)


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
    amplitudes = rng.uniform(0.6, 1.8, size=(n_peaks,)).astype(np.float32) * rng.uniform(0.75, 1.25)
    sigmas = rng.uniform(0.9, 3.0, size=(n_peaks,)).astype(np.float32)

    pseudo_centers = np.zeros((0, 2), dtype=np.float32)
    pseudo_amp = np.zeros((0,), dtype=np.float32)
    pseudo_sig = np.zeros((0,), dtype=np.float32)
    if float(cfg.pseudo_peak_prob) > 0.0 and rng.random() < float(cfg.pseudo_peak_prob):
        k = int(rng.integers(0, int(cfg.pseudo_peak_max) + 1))
        if k > 0:
            pseudo_centers = np.stack([_sample_uniform_point(cfg, rng) for _ in range(k)], axis=0).astype(np.float32)
            pseudo_sig = rng.uniform(2.4, 4.8, size=(k,)).astype(np.float32)
            base = float(np.median(amplitudes))
            pseudo_amp = (rng.uniform(0.08, 0.28, size=(k,)).astype(np.float32) * base).astype(np.float32)

    x_signal = _render_gaussian_peaks(cfg.height, cfg.width, centers, amplitudes, sigmas)
    if pseudo_centers.shape[0] > 0:
        x_signal = x_signal + _render_gaussian_peaks(cfg.height, cfg.width, pseudo_centers, pseudo_amp, pseudo_sig)

    if float(cfg.warp_prob) > 0.0 and rng.random() < float(cfg.warp_prob):
        x_signal = _warp_image(x_signal, rng=rng, strength=float(cfg.warp_strength))

    if float(cfg.corr_noise_prob) > 0.0 and rng.random() < float(cfg.corr_noise_prob):
        sig_power = float(np.mean(x_signal**2) + 1e-12)
        snr_lin = float(10.0 ** (float(cfg.snr_db) / 10.0))
        noise_power = sig_power / max(snr_lin, 1e-12)
        noise_std = float(np.sqrt(noise_power))
        L = int(max(int(cfg.L), 1))
        noise = rng.normal(0.0, noise_std, size=(L, *x_signal.shape)).astype(np.float32).mean(axis=0)
        smooth = _mean_filter_3x3(noise)
        a = float(np.clip(float(cfg.corr_strength), 0.0, 1.0))
        corr = (1 - a) * noise + a * smooth
        corr = corr * (float(noise.std() + 1e-6) / float(corr.std() + 1e-6))
        x = x_signal + corr.astype(np.float32)
    else:
        x = _add_noise_snr(x_signal, cfg.snr_db, cfg.L, rng)
    x = np.clip(x, 0.0, None).astype(np.float32)

    if bool(cfg.enable_roi_shift):
        # Pipeline-ROI crop/align error (detection/localization error propagation).
        # In "pipeline" mode, the offset scale is SNR/L-dependent via `_center_sigma(cfg)`.
        dy, dx = _sample_roi_offset(cfg, rng)
        x = _shift_bilinear(x, dy=dy, dx=dx)

    # Occlusion/truncation (v2 default: border-cut):
    # simulate ROI alignment error propagation causing border truncation of the true structure.
    x = _apply_occlusion(x, cfg, rng)

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
