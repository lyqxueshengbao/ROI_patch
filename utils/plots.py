from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: str,
    normalize: bool,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm.astype(np.int64)
        fmt = "d"

    fig = plt.figure(figsize=(6.2, 5.6), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Pred",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_plot.max() * 0.6 if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(
                j,
                i,
                format(cm_plot[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
