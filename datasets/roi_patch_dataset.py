from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.toy_generator import ToyGenConfig, generate_sample


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
        center_sigma_oracle: float = 1.0,
        center_sigma_min: float = 1.5,
        center_sigma_max: float = 6.0,
        pseudo_peak_prob: float = 0.35,
        pseudo_peak_max: int = 2,
        warp_prob: float = 0.25,
        warp_strength: float = 0.6,
        corr_noise_prob: float = 0.25,
        corr_strength: float = 0.6,
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
        self.center_sigma_oracle = float(center_sigma_oracle)
        self.center_sigma_min = float(center_sigma_min)
        self.center_sigma_max = float(center_sigma_max)
        self.pseudo_peak_prob = float(pseudo_peak_prob)
        self.pseudo_peak_max = int(pseudo_peak_max)
        self.warp_prob = float(warp_prob)
        self.warp_strength = float(warp_strength)
        self.corr_noise_prob = float(corr_noise_prob)
        self.corr_strength = float(corr_strength)
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
    center_sigma_oracle: float = 1.0,
    center_sigma_min: float = 1.5,
    center_sigma_max: float = 6.0,
    pseudo_peak_prob: float = 0.35,
    pseudo_peak_max: int = 2,
    warp_prob: float = 0.25,
    warp_strength: float = 0.6,
    corr_noise_prob: float = 0.25,
    corr_strength: float = 0.6,
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
        center_sigma_oracle=center_sigma_oracle,
        center_sigma_min=center_sigma_min,
        center_sigma_max=center_sigma_max,
        pseudo_peak_prob=pseudo_peak_prob,
        pseudo_peak_max=pseudo_peak_max,
        warp_prob=warp_prob,
        warp_strength=warp_strength,
        corr_noise_prob=corr_noise_prob,
        corr_strength=corr_strength,
        enable_aug=enable_aug,
        height=height,
        width=width,
    )
