from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.toy_generator import ToyGenConfig, generate_sample


AugProfile = Literal["oracle", "pipeline"]


@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


def _normalize_split(split: Sequence[float]) -> SplitSpec:
    if len(split) != 3:
        raise ValueError("--split must have 3 floats: train val test")
    s = float(sum(split))
    if s <= 0:
        raise ValueError("split sum must be > 0")
    return SplitSpec(train=float(split[0]) / s, val=float(split[1]) / s, test=float(split[2]) / s)


def _profile_params(profile: AugProfile) -> dict:
    if profile == "oracle":
        return {
            "pseudo_peak_prob": 0.10,
            "corr_noise_prob": 0.10,
            "warp_prob": 0.10,
            "warp_strength": 0.30,
            "corr_strength": 0.30,
            # v2 border-cut (train-time nonzero to improve shifted robustness)
            "border_prob": 0.20,
            "border_min": 2,
            "border_max": 6,
            "border_sides": "one",
            "border_fill": "sat_noise",
            "border_sat_strength": 1.4,
            "border_sat_q": 0.992,
            "border_sat_noise": 0.10,
            "border_sat_clip": True,
        }
    if profile == "pipeline":
        return {
            "pseudo_peak_prob": 0.35,
            "corr_noise_prob": 0.25,
            "warp_prob": 0.25,
            "warp_strength": 0.60,
            "corr_strength": 0.60,
            # v2 border-cut (stronger, mimics pipeline ROI crop/truncation)
            "border_prob": 0.45,
            "border_min": 4,
            "border_max": 12,
            "border_sides": "rand12",
            "border_fill": "sat_noise",
            "border_sat_strength": 2.5,
            "border_sat_q": 0.995,
            "border_sat_noise": 0.10,
            "border_sat_clip": True,
        }
    raise ValueError(f"Unknown profile={profile}")


def make_cfg(
    *,
    profile: AugProfile,
    roi_mode: Literal["oracle", "pipeline"],
    snr_db: float,
    L: int,
    hf_mode: Literal["laplacian", "sobel"],
    normalize: Literal["none", "per_sample"],
    enable_aug: bool,
    height: int,
    width: int,
    center_sigma_oracle: float = 1.0,
    center_sigma_min: float = 1.5,
    center_sigma_max: float = 6.0,
    pseudo_peak_max: int = 2,
) -> ToyGenConfig:
    p = _profile_params(profile)
    return ToyGenConfig(
        height=int(height),
        width=int(width),
        snr_db=float(snr_db),
        L=int(L),
        hf_mode=hf_mode,
        normalize=normalize,
        roi_mode=roi_mode,
        center_sigma_oracle=float(center_sigma_oracle),
        center_sigma_min=float(center_sigma_min),
        center_sigma_max=float(center_sigma_max),
        pseudo_peak_prob=float(p["pseudo_peak_prob"]),
        pseudo_peak_max=int(pseudo_peak_max),
        warp_prob=float(p["warp_prob"]),
        warp_strength=float(p["warp_strength"]),
        corr_noise_prob=float(p["corr_noise_prob"]),
        corr_strength=float(p["corr_strength"]),
        enable_occlude=True,
        occlude_mode="border",
        border_prob=float(p["border_prob"]),
        border_min=int(p["border_min"]),
        border_max=int(p["border_max"]),
        border_sides=str(p["border_sides"]),  # type: ignore[arg-type]
        border_fill=str(p["border_fill"]),  # type: ignore[arg-type]
        border_sat_strength=float(p["border_sat_strength"]),
        border_sat_q=float(p["border_sat_q"]),
        border_sat_noise=float(p["border_sat_noise"]),
        border_sat_clip=bool(p["border_sat_clip"]),
        enable_aug=bool(enable_aug),
    )


class ToyROIPatchDataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "val", "test"],
        total_samples: int,
        split_ratio: Sequence[float] = (0.7, 0.15, 0.15),
        base_seed: int = 0,
        snr_list: Iterable[float] = (5.0,),
        L_list: Iterable[int] = (1,),
        hf_mode: Literal["laplacian", "sobel"] = "laplacian",
        normalize: Literal["none", "per_sample"] = "per_sample",
        roi_mode: Literal["oracle", "pipeline"] = "oracle",
        aug_profile: AugProfile | str = "",
        center_sigma_oracle: float = 1.0,
        center_sigma_min: float = 1.5,
        center_sigma_max: float = 6.0,
        pseudo_peak_prob: float = 0.35,
        pseudo_peak_max: int = 2,
        warp_prob: float = 0.25,
        warp_strength: float = 0.6,
        corr_noise_prob: float = 0.25,
        corr_strength: float = 0.6,
        enable_occlude: bool = True,
        occlude_mode: Literal["none", "block", "border"] = "border",
        border_prob: float = 0.35,
        border_sides: Literal["one", "two", "rand12"] = "rand12",
        border_min: int = 4,
        border_max: int = 14,
        border_fill: Literal["zero", "mean", "min", "sat_const", "sat_noise", "sat_quantile"] = "sat_noise",
        border_sat_q: float = 0.995,
        border_sat_strength: float = 2.0,
        border_sat_noise: float = 0.10,
        border_sat_clip: bool = True,
        occlude_prob: float = 0.05,
        occlude_max_blocks: int = 1,
        occlude_min_size: int = 4,
        occlude_max_size: int = 10,
        occlude_fill: Literal["zero", "mean"] = "mean",
        enable_aug: bool = True,
        num_classes: int = 4,
        height: int = 41,
        width: int = 41,
    ) -> None:
        super().__init__()
        self.split = split
        self.total_samples = int(total_samples)
        self.split_spec = _normalize_split(split_ratio)
        self.base_seed = int(base_seed)
        self.snr_list = [float(x) for x in snr_list]
        self.L_list = [int(x) for x in L_list]
        self.hf_mode = hf_mode
        self.normalize = normalize
        self.roi_mode = roi_mode
        self.aug_profile = str(aug_profile)
        self.center_sigma_oracle = float(center_sigma_oracle)
        self.center_sigma_min = float(center_sigma_min)
        self.center_sigma_max = float(center_sigma_max)
        self.pseudo_peak_prob = float(pseudo_peak_prob)
        self.pseudo_peak_max = int(pseudo_peak_max)
        self.warp_prob = float(warp_prob)
        self.warp_strength = float(warp_strength)
        self.corr_noise_prob = float(corr_noise_prob)
        self.corr_strength = float(corr_strength)
        self.enable_occlude = bool(enable_occlude)
        self.occlude_mode = occlude_mode
        self.border_prob = float(border_prob)
        self.border_sides = border_sides
        self.border_min = int(border_min)
        self.border_max = int(border_max)
        self.border_fill = border_fill
        self.border_sat_q = float(border_sat_q)
        self.border_sat_strength = float(border_sat_strength)
        self.border_sat_noise = float(border_sat_noise)
        self.border_sat_clip = bool(border_sat_clip)
        self.occlude_prob = float(occlude_prob)
        self.occlude_max_blocks = int(occlude_max_blocks)
        self.occlude_min_size = int(occlude_min_size)
        self.occlude_max_size = int(occlude_max_size)
        self.occlude_fill = occlude_fill
        self.enable_aug = bool(enable_aug)
        self.num_classes = int(num_classes)
        self.height = int(height)
        self.width = int(width)
        self._epoch = 0

        if self.total_samples <= 0:
            raise ValueError("total_samples must be > 0")
        if self.num_classes != 4:
            raise ValueError("This toy generator currently supports num_classes=4")
        if len(self.snr_list) == 0 or len(self.L_list) == 0:
            raise ValueError("snr_list and L_list must be non-empty")

        n_train = int(round(self.total_samples * self.split_spec.train))
        n_val = int(round(self.total_samples * self.split_spec.val))
        n_test = self.total_samples - n_train - n_val
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)
        diff = self.total_samples - (n_train + n_val + n_test)
        n_test += diff

        self._split_sizes = {"train": n_train, "val": n_val, "test": n_test}
        offsets = {"train": 0, "val": n_train, "test": n_train + n_val}
        self._offset = offsets[split]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return int(self._split_sizes[self.split])

    def _rng_for_index(self, global_index: int) -> np.random.Generator:
        seed = (self.base_seed * 1000003 + self._epoch * 9176 + global_index * 7919) % (2**32 - 1)
        return np.random.default_rng(int(seed))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        gi = self._offset + int(index)
        rng = self._rng_for_index(gi)
        label = int(rng.integers(0, self.num_classes))
        snr_db = float(rng.choice(self.snr_list))
        L = int(rng.choice(self.L_list))
        if self.aug_profile:
            cfg = make_cfg(
                profile=self.aug_profile,  # type: ignore[arg-type]
                roi_mode=self.roi_mode,
                snr_db=snr_db,
                L=L,
                hf_mode=self.hf_mode,
                normalize=self.normalize,
                enable_aug=self.enable_aug and (self.split == "train"),
                height=self.height,
                width=self.width,
                center_sigma_oracle=self.center_sigma_oracle,
                center_sigma_min=self.center_sigma_min,
                center_sigma_max=self.center_sigma_max,
                pseudo_peak_max=self.pseudo_peak_max,
            )
        else:
            cfg = ToyGenConfig(
                height=self.height,
                width=self.width,
                snr_db=snr_db,
                L=L,
                hf_mode=self.hf_mode,
                normalize=self.normalize,
                roi_mode=self.roi_mode,
                center_sigma_oracle=self.center_sigma_oracle,
                center_sigma_min=self.center_sigma_min,
                center_sigma_max=self.center_sigma_max,
                pseudo_peak_prob=self.pseudo_peak_prob,
                pseudo_peak_max=self.pseudo_peak_max,
                warp_prob=self.warp_prob,
                warp_strength=self.warp_strength,
                corr_noise_prob=self.corr_noise_prob,
                corr_strength=self.corr_strength,
                enable_occlude=self.enable_occlude,
                occlude_mode=self.occlude_mode,
                border_prob=self.border_prob,
                border_sides=self.border_sides,
                border_min=self.border_min,
                border_max=self.border_max,
                border_fill=self.border_fill,
                border_sat_q=self.border_sat_q,
                border_sat_strength=self.border_sat_strength,
                border_sat_noise=self.border_sat_noise,
                border_sat_clip=self.border_sat_clip,
                occlude_prob=self.occlude_prob,
                occlude_max_blocks=self.occlude_max_blocks,
                occlude_min_size=self.occlude_min_size,
                occlude_max_size=self.occlude_max_size,
                occlude_fill=self.occlude_fill,
                enable_aug=self.enable_aug and (self.split == "train"),
            )
        x_np, y = generate_sample(label, cfg, rng)
        x = torch.from_numpy(x_np)
        return x, int(y)


def make_fixed_condition_dataset(
    split: Literal["train", "val", "test"],
    total_samples: int,
    split_ratio: Sequence[float],
    base_seed: int,
    snr_db: float,
    L: int,
    hf_mode: Literal["laplacian", "sobel"],
    normalize: Literal["none", "per_sample"],
    enable_aug: bool,
    height: int,
    width: int,
    *,
    roi_mode: Literal["oracle", "pipeline"] = "oracle",
    aug_profile: AugProfile | str = "",
    center_sigma_oracle: float = 1.0,
    center_sigma_min: float = 1.5,
    center_sigma_max: float = 6.0,
    pseudo_peak_prob: float = 0.35,
    pseudo_peak_max: int = 2,
    warp_prob: float = 0.25,
    warp_strength: float = 0.6,
    corr_noise_prob: float = 0.25,
    corr_strength: float = 0.6,
    enable_occlude: bool = True,
    occlude_mode: Literal["none", "block", "border"] = "border",
    border_prob: float = 0.35,
    border_sides: Literal["one", "two", "rand12"] = "rand12",
    border_min: int = 4,
    border_max: int = 14,
    border_fill: Literal["zero", "mean", "min", "sat_const", "sat_noise", "sat_quantile"] = "sat_noise",
    border_sat_q: float = 0.995,
    border_sat_strength: float = 2.0,
    border_sat_noise: float = 0.10,
    border_sat_clip: bool = True,
    occlude_prob: float = 0.05,
    occlude_max_blocks: int = 1,
    occlude_min_size: int = 4,
    occlude_max_size: int = 10,
    occlude_fill: Literal["zero", "mean"] = "mean",
) -> ToyROIPatchDataset:
    return ToyROIPatchDataset(
        split=split,
        total_samples=total_samples,
        split_ratio=split_ratio,
        base_seed=base_seed,
        snr_list=(float(snr_db),),
        L_list=(int(L),),
        hf_mode=hf_mode,
        normalize=normalize,
        roi_mode=roi_mode,
        aug_profile=aug_profile,
        center_sigma_oracle=center_sigma_oracle,
        center_sigma_min=center_sigma_min,
        center_sigma_max=center_sigma_max,
        pseudo_peak_prob=pseudo_peak_prob,
        pseudo_peak_max=pseudo_peak_max,
        warp_prob=warp_prob,
        warp_strength=warp_strength,
        corr_noise_prob=corr_noise_prob,
        corr_strength=corr_strength,
        enable_occlude=enable_occlude,
        occlude_mode=occlude_mode,
        border_prob=border_prob,
        border_sides=border_sides,
        border_min=border_min,
        border_max=border_max,
        border_fill=border_fill,
        border_sat_q=border_sat_q,
        border_sat_strength=border_sat_strength,
        border_sat_noise=border_sat_noise,
        border_sat_clip=border_sat_clip,
        occlude_prob=occlude_prob,
        occlude_max_blocks=occlude_max_blocks,
        occlude_min_size=occlude_min_size,
        occlude_max_size=occlude_max_size,
        occlude_fill=occlude_fill,
        enable_aug=enable_aug,
        height=height,
        width=width,
    )
