# ROI patch 多类结构形态识别（Toy 仿真数据A路线）

本项目用 **Python + PyTorch** 从零搭建一个可运行的“ROI patch 多类结构形态识别”基线工程。数据完全由 toy generator 在 **41×41** 的 2D patch 上合成多个高斯峰（不依赖任何真实雷达数据/公开数据集）。

## 任务

- 输入：二维 ROI patch，形状 `[C, H, W]`，默认 `H=W=41`，默认 `C=2`
  - `X0 = log(X + eps)`，`X` 为强度图（多高斯峰叠加 + 噪声）
  - `Xhf`：`Laplacian(X0)` 或 `Sobel` 梯度幅值（通过参数控制）
- 输出：`K=4` 类分类
  - class0：2 peaks close（双峰近距）
  - class1：2 peaks far（双峰远距）
  - class2：3 peaks aligned roughly in a line（三峰线状）
  - class3：3 peaks clustered（三峰簇状）
- 评测：Accuracy、Macro-F1、混淆矩阵（原始计数 + 按行归一化），支持 repeat 统计 mean±std

## 安装

```bash
pip install -r requirements.txt
```

Windows 如果没有 `python` 命令，可用 `py` 代替（例如 `py train.py ...`）。

## 训练（toy 数据）

1) Baseline CNN：

```bash
py train.py --model baseline --epochs 30 --batch_size 256 --lr 3e-4 ^
  --snr_list -5 0 5 10 --L_list 1 4 16 --repeat 3 --out_dir runs/baseline_toy
```

2) 高频引导双分支 + gated fusion：

```bash
py train.py --model hf_gated --hf_mode laplacian --epochs 30 --batch_size 256 --lr 3e-4 ^
  --snr_list -5 0 5 10 --L_list 1 4 16 --repeat 3 --out_dir runs/hf_gated_toy
```

训练输出（每个 repeat 一个子目录 `runs/.../repeat_000/`）：
- `best.pt`、`last.pt`
- `train_log.csv`（每 epoch 的 train/val：loss、acc、macro-f1）
- `test_metrics_by_condition.csv`：按 `(SNR,L)` 分组的测试指标（后续画曲线/表格用这个）
- `test_metrics.csv`：同时给出
  - `*_all_conditions`：把所有 `(SNR,L)` 条件的 test 拼接后统一计算（各条件等权）
  - `*_mixed_sampler`：test 数据集内部混合采样 `snr_list/L_list` 得到的总体分数（不推荐作为主要口径）
- 混淆矩阵 PNG（计数版 + 行归一化版）：整体 + 各 `(SNR,L)` 条件
- `summary_mean_std.csv`：对 repeats 的 mean±std（基于 `*_all_conditions`）

## 评测（按条件）

对某次训练的 best checkpoint，在不同 SNR/L 条件下评测并导出图表：

```bash
py eval.py --ckpt runs/hf_gated_toy/repeat_000/best.pt ^
  --snr_list -5 0 5 10 --L_list 1 4 16 --out_dir runs/hf_gated_toy/repeat_000/eval
```

对一个 `out_dir` 下所有 repeat 的 `best.pt` 进行批量评测并汇总 mean±std：

```bash
py eval.py --ckpt_dir runs/hf_gated_toy --snr_list -5 0 5 10 --L_list 1 4 16
```

## Diagnostics: shortcut checks (border-jam / border-cut)

If you add border-based augmentation (e.g., border-jam) and performance behaves unexpectedly, use these tools to detect whether the augmentation injects a *boundary shortcut*:

1) **Handcrafted feature ablation (peaks vs stats)**
   - Run `eval_traditional.py` with `--methods handfeat_svm` and select `--handfeat_mode {full,peaks_only,stats_only}`.
   - If `stats_only` stays strong (or improves) under border-jam, it suggests the boundary artifact is being exploited as a shortcut.

2) **Edge-shortcut baseline (boundary-only classifier)**
   - Run `diagnose_edge_shortcut.py` to train a linear classifier using only boundary-band statistics.
   - If Macro-F1 is high, your border augmentation likely encodes class information at the boundary (shortcut), and should be revised/removed.
