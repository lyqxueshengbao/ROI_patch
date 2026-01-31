from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from datasets.toy_generator import _laplacian, _sobel_mag


@dataclass(frozen=True)
class FdaMimoGenConfig:
    patch_h: int = 41
    patch_w: int = 41
    snr_db: float = 5.0
    L: int = 1
    eps: float = 1e-6
    hf_mode: Literal["laplacian", "sobel"] = "laplacian"
    normalize: Literal["none", "per_sample"] = "per_sample"

    # ROI mode (consistent with toy)
    roi_mode: Literal["oracle", "pipeline"] = "oracle"
    center_sigma_oracle: float = 1.0
    center_sigma_min: float = 1.5
    center_sigma_max: float = 6.0
    center_trunc: float = 8.0
    center_snr_low: float = -15.0
    center_snr_high: float = 20.0
    center_L_min: int = 1
    center_L_max: int = 16

    # FDA-MIMO parameters (v0: simplified model, replace with your real pipeline later)
    f0: float = 1e9
    M: int = 10
    N: int = 10
    delta_f: float = 30e3
    d: float = 0.5  # spacing in wavelengths (relative to lambda0)
    c: float = 3e8

    # local patch grid (compute 41x41 directly, no full-image)
    theta_span_deg: float = 20.0  # total span (e.g., +/-10deg)
    r_span_m: float = 400.0  # total span (e.g., +/-200m)

    # structure center in physical coords (kept fixed; ROI offset moves around it)
    scene_theta0_deg: float = 0.0
    scene_r0_m: float = 1000.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _center_sigma(snr_db: float, L: int, cfg: FdaMimoGenConfig) -> float:
    if cfg.roi_mode == "oracle":
        return float(cfg.center_sigma_oracle)
    if cfg.roi_mode != "pipeline":
        raise ValueError(f"Unknown roi_mode={cfg.roi_mode}")

    snr_norm = _clip01((float(snr_db) - float(cfg.center_snr_low)) / (float(cfg.center_snr_high) - float(cfg.center_snr_low) + 1e-12))
    Lmin = max(int(cfg.center_L_min), 1)
    Lmax = max(int(cfg.center_L_max), Lmin + 1)
    Lc = max(int(L), 1)
    L_norm = _clip01((np.log2(Lc) - np.log2(Lmin)) / (np.log2(Lmax) - np.log2(Lmin) + 1e-12))
    mix = 0.6 * snr_norm + 0.4 * float(L_norm)
    sigma = float(cfg.center_sigma_max) - (float(cfg.center_sigma_max) - float(cfg.center_sigma_min)) * float(mix)
    return float(np.clip(sigma, float(cfg.center_sigma_min), float(cfg.center_sigma_max)))


def _sample_roi_offset_pix(cfg: FdaMimoGenConfig, rng: np.random.Generator, *, snr_db: float, L: int) -> tuple[float, float]:
    # ROI crop/align error (detection/localization error propagation).
    # sigma is SNR/L-dependent in "pipeline" mode via `_center_sigma`.
    s = float(_center_sigma(snr_db, L, cfg))
    trunc = float(cfg.center_trunc)
    dy = float(rng.normal(0.0, s))
    dx = float(rng.normal(0.0, s))
    dy = float(np.clip(dy, -trunc, trunc))
    dx = float(np.clip(dx, -trunc, trunc))
    return dy, dx


def _sample_scatterers(
    label: int,
    cfg: FdaMimoGenConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (theta_deg[K], r_m[K], alpha_complex[K]) describing the true structure.
    Labels match the toy task semantics:
      0: 2 scatters close
      1: 2 scatters far
      2: 3 scatters line
      3: 3 scatters cluster
    """
    theta0 = float(cfg.scene_theta0_deg)
    r0 = float(cfg.scene_r0_m)

    if label == 0:
        dtheta = float(rng.uniform(0.6, 1.6))
        dr = float(rng.uniform(12.0, 42.0))
        thetas = np.array([theta0 - 0.5 * dtheta, theta0 + 0.5 * dtheta], dtype=np.float32)
        rs = np.array([r0 - 0.5 * dr, r0 + 0.5 * dr], dtype=np.float32)
    elif label == 1:
        dtheta = float(rng.uniform(6.0, 9.0))
        dr = float(rng.uniform(120.0, 180.0))
        thetas = np.array([theta0 - 0.5 * dtheta, theta0 + 0.5 * dtheta], dtype=np.float32)
        rs = np.array([r0 - 0.5 * dr, r0 + 0.5 * dr], dtype=np.float32)
    elif label == 2:
        # 3 points on a line in normalized (theta,r) coords
        s_th = float(cfg.theta_span_deg) * 0.5
        s_r = float(cfg.r_span_m) * 0.5
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        u_th = float(np.cos(ang))
        u_r = float(np.sin(ang))
        d = float(rng.uniform(0.25, 0.55))
        t = np.array([-d, 0.0, d], dtype=np.float32)
        thetas = (theta0 + (t * u_th * s_th)).astype(np.float32)
        rs = (r0 + (t * u_r * s_r)).astype(np.float32)
    elif label == 3:
        s_th = float(cfg.theta_span_deg) * 0.5
        s_r = float(cfg.r_span_m) * 0.5
        th_off = rng.normal(0.0, 0.18, size=(3,)).astype(np.float32)
        r_off = rng.normal(0.0, 0.18, size=(3,)).astype(np.float32)
        th_off = np.clip(th_off, -0.45, 0.45)
        r_off = np.clip(r_off, -0.45, 0.45)
        thetas = (theta0 + th_off * s_th).astype(np.float32)
        rs = (r0 + r_off * s_r).astype(np.float32)
    else:
        raise ValueError(f"Unknown label={label}")

    K = int(thetas.shape[0])
    amp = rng.uniform(0.8, 1.2, size=(K,)).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(K,)).astype(np.float32)
    alpha = (amp * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)
    return thetas.astype(np.float32), rs.astype(np.float32), alpha


def _steering_numpy(theta_deg: np.ndarray, r_m: np.ndarray, cfg: FdaMimoGenConfig) -> np.ndarray:
    """
    Return steering vectors for targets at (theta_deg, r_m).
    Output: [K, MN] complex64.
    """
    M = int(cfg.M)
    N = int(cfg.N)
    f0 = float(cfg.f0)
    c = float(cfg.c)
    delta_f = float(cfg.delta_f)
    lambda0 = float(c / f0)
    k0 = float(2.0 * np.pi / lambda0)

    m = np.arange(M, dtype=np.float32)
    n = np.arange(N, dtype=np.float32)
    mm, nn = np.meshgrid(m, n, indexing="ij")
    x_mn = (mm + nn).astype(np.float32) * float(cfg.d) * float(lambda0)
    f_m = (f0 + mm.astype(np.float32) * delta_f).astype(np.float32)

    theta = np.deg2rad(theta_deg.astype(np.float32)).reshape(-1, 1, 1)
    r = r_m.astype(np.float32).reshape(-1, 1, 1)

    phase_r = (-4.0 * np.pi * f_m / c) * r  # [K,M,N]
    phase_th = (k0 * x_mn) * np.sin(theta)  # [K,M,N]
    phase = phase_r + phase_th
    a = np.exp(1j * phase.astype(np.float32)).astype(np.complex64)  # [K,M,N]
    return a.reshape(a.shape[0], -1)  # [K,MN]


def _response_patch_numpy(
    theta_grid_deg: np.ndarray,  # [W]
    r_grid_m: np.ndarray,  # [H]
    y: np.ndarray,  # [MN] complex64
    cfg: FdaMimoGenConfig,
) -> np.ndarray:
    # Compute P(theta,r) on the local grid via matched-filter power: |a^H y|^2.
    H = int(r_grid_m.shape[0])
    W = int(theta_grid_deg.shape[0])
    thetas = np.repeat(theta_grid_deg.reshape(1, W), H, axis=0).reshape(-1)
    rs = np.repeat(r_grid_m.reshape(H, 1), W, axis=1).reshape(-1)
    a = _steering_numpy(thetas, rs, cfg)  # [HW,MN]
    z = np.sum(np.conj(a) * y.reshape(1, -1), axis=1)  # [HW]
    P = (np.abs(z) ** 2).astype(np.float32) / float(y.size + 1e-12)
    return P.reshape(H, W).astype(np.float32)


def generate_fdamimo_sample(
    label: int,
    cfg: FdaMimoGenConfig,
    rng: np.random.Generator,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, int]:
    """
    v0 reference implementation (numpy): generate a single (2,H,W) ROI patch.
    ROI offset is applied by shifting the (theta_center, r_center) used to define the local grid.
    """
    if device is not None:
        # Keep a single-sample numpy path; training uses `generate_fdamimo_batch_torch` when gen_on_gpu=True.
        pass

    theta_t, r_t, alpha = _sample_scatterers(int(label), cfg, rng)

    # Oracle center: geometric mean of true structure.
    theta_center = float(theta_t.mean())
    r_center = float(r_t.mean())

    # Pipeline center: add SNR/L-dependent ROI offset (in pixel units mapped to theta/r units).
    if cfg.roi_mode in ("oracle", "pipeline"):
        dy, dx = _sample_roi_offset_pix(cfg, rng, snr_db=float(cfg.snr_db), L=int(cfg.L))
        dtheta = float(cfg.theta_span_deg) / float(max(int(cfg.patch_w) - 1, 1))
        dr = float(cfg.r_span_m) / float(max(int(cfg.patch_h) - 1, 1))
        theta_center = float(theta_center + dx * dtheta)
        r_center = float(r_center + dy * dr)
    else:
        raise ValueError(f"Unknown roi_mode={cfg.roi_mode}")

    theta_grid = np.linspace(
        theta_center - 0.5 * float(cfg.theta_span_deg),
        theta_center + 0.5 * float(cfg.theta_span_deg),
        int(cfg.patch_w),
        dtype=np.float32,
    )
    r_grid = np.linspace(
        r_center - 0.5 * float(cfg.r_span_m),
        r_center + 0.5 * float(cfg.r_span_m),
        int(cfg.patch_h),
        dtype=np.float32,
    )

    A_t = _steering_numpy(theta_t, r_t, cfg)  # [K,MN]
    y_sig = (alpha.reshape(-1, 1) * A_t).sum(axis=0).astype(np.complex64)  # [MN]

    sig_power = float(np.mean(np.abs(y_sig) ** 2) + 1e-12)
    snr_lin = float(10.0 ** (float(cfg.snr_db) / 10.0))
    noise_power = sig_power / max(snr_lin, 1e-12)
    noise_std = float(np.sqrt(noise_power)) / float(np.sqrt(max(int(cfg.L), 1)))
    noise = (
        rng.normal(0.0, noise_std / np.sqrt(2.0), size=y_sig.shape).astype(np.float32)
        + 1j * rng.normal(0.0, noise_std / np.sqrt(2.0), size=y_sig.shape).astype(np.float32)
    ).astype(np.complex64)
    y = (y_sig + noise).astype(np.complex64)

    P = _response_patch_numpy(theta_grid, r_grid, y, cfg)  # [H,W] float32, >=0
    x = np.clip(P, 0.0, None).astype(np.float32)

    x0 = np.log(x + float(cfg.eps)).astype(np.float32)
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
    return x_stack.astype(np.float32), int(label)


def _torch_reflect_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    # x: [B,1,H,W]
    return F.pad(x, (pad, pad, pad, pad), mode="reflect")


def _torch_laplacian(x0: torch.Tensor) -> torch.Tensor:
    # x0: [B,H,W] float32
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x0.dtype, device=x0.device).view(1, 1, 3, 3)
    x = _torch_reflect_pad(x0.unsqueeze(1), 1)
    out = F.conv2d(x, k).squeeze(1)
    return out


def _torch_sobel_mag(x0: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x0.dtype, device=x0.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x0.dtype, device=x0.device).view(1, 1, 3, 3)
    x = _torch_reflect_pad(x0.unsqueeze(1), 1)
    gx = F.conv2d(x, kx).squeeze(1)
    gy = F.conv2d(x, ky).squeeze(1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _center_sigma_torch(snr_db: torch.Tensor, L: torch.Tensor, cfg: FdaMimoGenConfig) -> torch.Tensor:
    if cfg.roi_mode == "oracle":
        return torch.full_like(snr_db, float(cfg.center_sigma_oracle))
    if cfg.roi_mode != "pipeline":
        raise ValueError(f"Unknown roi_mode={cfg.roi_mode}")

    snr_norm = torch.clamp(
        (snr_db - float(cfg.center_snr_low)) / (float(cfg.center_snr_high) - float(cfg.center_snr_low) + 1e-12),
        0.0,
        1.0,
    )
    Lmin = float(max(int(cfg.center_L_min), 1))
    Lmax = float(max(int(cfg.center_L_max), int(Lmin) + 1))
    Lc = torch.clamp(L.to(dtype=torch.float32), min=1.0)
    L_norm = torch.clamp((torch.log2(Lc) - np.log2(Lmin)) / (np.log2(Lmax) - np.log2(Lmin) + 1e-12), 0.0, 1.0)
    mix = 0.6 * snr_norm + 0.4 * L_norm
    sigma = float(cfg.center_sigma_max) - (float(cfg.center_sigma_max) - float(cfg.center_sigma_min)) * mix
    return torch.clamp(sigma, float(cfg.center_sigma_min), float(cfg.center_sigma_max))


def generate_fdamimo_batch_torch(
    *,
    labels: torch.Tensor,  # [B] int64
    snr_db: torch.Tensor,  # [B] float32
    L: torch.Tensor,  # [B] int64
    cfg: FdaMimoGenConfig,
    generator: torch.Generator | None,
    device: torch.device,
    hw_chunk: int = 256,
) -> torch.Tensor:
    """
    GPU batch generator: returns x_stack [B,2,H,W] float32.
    v0: simplified FDA-MIMO matched-filter response on a local (theta,r) grid.
    """
    B = int(labels.numel())
    H = int(cfg.patch_h)
    W = int(cfg.patch_w)
    M = int(cfg.M)
    N = int(cfg.N)
    Kch = M * N

    labels = labels.to(device=device, dtype=torch.int64)
    snr_db = snr_db.to(device=device, dtype=torch.float32)
    L = L.to(device=device, dtype=torch.int64)

    def _uniform_(t: torch.Tensor, a: float, b: float) -> torch.Tensor:
        if generator is None:
            return t.uniform_(a, b)
        return t.uniform_(a, b, generator=generator)

    def _randn(shape: tuple[int, ...]) -> torch.Tensor:
        if generator is None:
            return torch.randn(shape, device=device, dtype=torch.float32)
        return torch.randn(shape, device=device, generator=generator, dtype=torch.float32)

    # Sample scatterer offsets in normalized coords and map to physical theta/r.
    s_th = float(cfg.theta_span_deg) * 0.5
    s_r = float(cfg.r_span_m) * 0.5
    theta0 = float(cfg.scene_theta0_deg)
    r0 = float(cfg.scene_r0_m)

    # Allocate max 3 targets per sample.
    th = torch.zeros((B, 3), device=device, dtype=torch.float32)
    rr = torch.zeros((B, 3), device=device, dtype=torch.float32)
    num_t = torch.full((B,), 3, device=device, dtype=torch.int64)

    # label 0/1 => 2 targets (mask the 3rd by zero amplitude later)
    mask2 = (labels == 0) | (labels == 1)
    num_t = torch.where(mask2, torch.tensor(2, device=device, dtype=torch.int64), num_t)

    # label 0: close
    m0 = labels == 0
    if bool(m0.any()):
        dtheta = _uniform_(torch.empty((int(m0.sum()),), device=device), 0.6, 1.6)
        dr = _uniform_(torch.empty((int(m0.sum()),), device=device), 12.0, 42.0)
        th[m0, 0] = theta0 - 0.5 * dtheta
        th[m0, 1] = theta0 + 0.5 * dtheta
        rr[m0, 0] = r0 - 0.5 * dr
        rr[m0, 1] = r0 + 0.5 * dr

    # label 1: far
    m1 = labels == 1
    if bool(m1.any()):
        dtheta = _uniform_(torch.empty((int(m1.sum()),), device=device), 6.0, 9.0)
        dr = _uniform_(torch.empty((int(m1.sum()),), device=device), 120.0, 180.0)
        th[m1, 0] = theta0 - 0.5 * dtheta
        th[m1, 1] = theta0 + 0.5 * dtheta
        rr[m1, 0] = r0 - 0.5 * dr
        rr[m1, 1] = r0 + 0.5 * dr

    # label 2: line (3 targets)
    m2 = labels == 2
    if bool(m2.any()):
        cnt = int(m2.sum())
        ang = _uniform_(torch.empty((cnt,), device=device), 0.0, float(2.0 * np.pi))
        u_th = torch.cos(ang)
        u_r = torch.sin(ang)
        d = _uniform_(torch.empty((cnt,), device=device), 0.25, 0.55)
        t = torch.stack([-d, torch.zeros_like(d), d], dim=1)  # [cnt,3]
        th[m2] = theta0 + t * u_th[:, None] * s_th
        rr[m2] = r0 + t * u_r[:, None] * s_r

    # label 3: cluster (3 targets)
    m3 = labels == 3
    if bool(m3.any()):
        cnt = int(m3.sum())
        th_off = _randn((cnt, 3)) * 0.18
        r_off = _randn((cnt, 3)) * 0.18
        th_off = torch.clamp(th_off, -0.45, 0.45)
        r_off = torch.clamp(r_off, -0.45, 0.45)
        th[m3] = theta0 + th_off * s_th
        rr[m3] = r0 + r_off * s_r

    # Complex amplitudes (3 max targets); 3rd is masked for 2-target labels.
    amp = _uniform_(torch.empty((B, 3), device=device), 0.8, 1.2)
    ph = _uniform_(torch.empty((B, 3), device=device), 0.0, float(2.0 * np.pi))
    alpha = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph)).to(torch.complex64)
    alpha = torch.where(num_t[:, None] >= torch.arange(1, 4, device=device)[None, :], alpha, torch.zeros_like(alpha))

    # Build channel geometry.
    f0 = float(cfg.f0)
    c = float(cfg.c)
    delta_f = float(cfg.delta_f)
    lambda0 = float(c / f0)
    k0 = float(2.0 * np.pi / lambda0)
    m = torch.arange(M, device=device, dtype=torch.float32)
    n = torch.arange(N, device=device, dtype=torch.float32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    x_mn = (mm + nn) * float(cfg.d) * float(lambda0)  # [M,N]
    f_m = f0 + mm * delta_f  # [M,N]
    x_mn = x_mn.reshape(1, 1, Kch)  # [1,1,K]
    f_m = f_m.reshape(1, 1, Kch)  # [1,1,K]

    theta_t = torch.deg2rad(th).reshape(B, 3, 1)
    r_t = rr.reshape(B, 3, 1)
    phase_r = (-4.0 * np.pi * f_m / c) * r_t  # [B,3,K]
    phase_th = (k0 * x_mn) * torch.sin(theta_t)  # [B,3,K]
    a_t = torch.exp(torch.complex(torch.zeros_like(phase_r), (phase_r + phase_th))).to(torch.complex64)
    y_sig = torch.sum(alpha.reshape(B, 3, 1) * a_t, dim=1)  # [B,K]

    sig_power = torch.mean(torch.abs(y_sig) ** 2, dim=1) + 1e-12
    snr_lin = torch.pow(torch.tensor(10.0, device=device), snr_db / 10.0)
    noise_power = sig_power / torch.clamp(snr_lin, min=1e-12)
    noise_std = torch.sqrt(noise_power) / torch.sqrt(torch.clamp(L.to(torch.float32), min=1.0))
    noise_re = _randn((B, Kch)) * (noise_std[:, None] / np.sqrt(2.0))
    noise_im = _randn((B, Kch)) * (noise_std[:, None] / np.sqrt(2.0))
    y = (y_sig + torch.complex(noise_re, noise_im).to(torch.complex64)).to(torch.complex64)  # [B,K]

    # Oracle center from true structure; then apply ROI offset for oracle/pipeline (oracle sigma small).
    theta_center = th.mean(dim=1)  # [B] deg
    r_center = rr.mean(dim=1)  # [B] m
    sigma = _center_sigma_torch(snr_db, L, cfg)
    trunc = float(cfg.center_trunc)
    dy = torch.clamp(_randn((B,)) * sigma, -trunc, trunc)
    dx = torch.clamp(_randn((B,)) * sigma, -trunc, trunc)
    dtheta_pix = float(cfg.theta_span_deg) / float(max(W - 1, 1))
    dr_pix = float(cfg.r_span_m) / float(max(H - 1, 1))
    theta_center = theta_center + dx * dtheta_pix
    r_center = r_center + dy * dr_pix

    # Local grid for each sample.
    theta_axis = torch.linspace(-0.5 * float(cfg.theta_span_deg), 0.5 * float(cfg.theta_span_deg), W, device=device, dtype=torch.float32)
    r_axis = torch.linspace(-0.5 * float(cfg.r_span_m), 0.5 * float(cfg.r_span_m), H, device=device, dtype=torch.float32)
    theta_grid = theta_center[:, None] + theta_axis[None, :]  # [B,W] deg
    r_grid = r_center[:, None] + r_axis[None, :]  # [B,H] m

    # Matched-filter response in HW chunks to reduce peak memory.
    out = torch.empty((B, H, W), device=device, dtype=torch.float32)
    theta_grid_rad = torch.deg2rad(theta_grid)  # [B,W]
    for y0 in range(0, H * W, hw_chunk):
        y1 = min(H * W, y0 + hw_chunk)
        idx = torch.arange(y0, y1, device=device)
        iy = idx // W
        ix = idx % W
        th_q = theta_grid_rad.gather(1, ix[None, :].expand(B, -1))  # [B,chunk]
        r_q = r_grid.gather(1, iy[None, :].expand(B, -1))  # [B,chunk]
        # steering for query grid points: [B,chunk,K]
        phase_rq = (-4.0 * np.pi * f_m / c) * r_q[:, :, None]
        phase_thq = (k0 * x_mn) * torch.sin(th_q[:, :, None])
        a_q = torch.exp(torch.complex(torch.zeros_like(phase_rq), (phase_rq + phase_thq))).to(torch.complex64)
        z = torch.sum(torch.conj(a_q) * y[:, None, :], dim=2)  # [B,chunk]
        p = (torch.abs(z) ** 2).to(torch.float32) / float(Kch)
        out.view(B, -1)[:, y0:y1] = p

    x = torch.clamp(out, min=0.0)
    x0 = torch.log(x + float(cfg.eps))
    if cfg.hf_mode == "laplacian":
        xhf = _torch_laplacian(x0)
    elif cfg.hf_mode == "sobel":
        xhf = _torch_sobel_mag(x0)
    else:
        raise ValueError(f"Unknown hf_mode={cfg.hf_mode}")

    x_stack = torch.stack([x0, xhf], dim=1).to(torch.float32)  # [B,2,H,W]
    if cfg.normalize == "per_sample":
        mean = x_stack.mean(dim=(2, 3), keepdim=True)
        std = x_stack.std(dim=(2, 3), keepdim=True) + 1e-6
        x_stack = (x_stack - mean) / std
    return x_stack
