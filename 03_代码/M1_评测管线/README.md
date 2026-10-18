# M1 评测管线

> 目标：**先通后优**。不接任何果蝇先验，先用 BNCI2014-001 把「LOSO × 多种子 × 两个 arm」的完整流程跑通，产出符合 IF-2 的结果表。
>
> 当前状态：骨架可运行；synthetic 数据可完整跑通；真实 BNCI2014-001 需要安装 MOABB。M1 第一版只做 CSP+LDA 与随机稀疏对照，EEGNet/torch 留给后续/M5。

## 1. 实现了什么

| 模块 | 内容 |
|---|---|
| `data.py` | `EpochDataset`；synthetic 数据生成；MOABB `BNCI2014_001` 加载；IF-1 `.fif` 读取；数据指纹 |
| `preprocess.py` | 训练折内 `ChannelStandardScaler` + `mne.decoding.CSP`；`RandomFeatureMask` |
| `arms.py` | `baseline`（CSP+LDA）与 `equiv_dropout`（CSP + 随机特征 mask + LDA） |
| `evaluate.py` | LOSO 循环、逐被试/逐 session balanced accuracy、逐试次预测、latency/memory、标签置换 canary |
| `run_m1.py` | 参数化 CLI：`--arm / --seed / --subject / --dataset`，`--dry-run`、`--check-only`、`--canary` |
| `configs/` | synthetic 与 BNCI2014-001 两份 YAML |

## 2. 两个 arm 的含义

### `baseline`
- `ChannelStandardScaler` → `CSP(n_components=6)` → `LDA(solver="lsqr", shrinkage="auto")`
- 无任何生物结构、无稀疏 mask。

### `equiv_dropout`（M1 的等效稀疏对照）
- 同样 `Scaler → CSP`，但在 CSP 特征后加一个**固定随机特征 mask**，保留 `density` 比例的特征，再进 LDA。
- `density=0.5` 只是 M1 的占位值；M2 之后应换成与结构 mask 相同的稀疏度。
- **注意**：这不是训练时的 literal dropout，而是"同等稀疏程度、但没有任何生物结构"的随机对照。它的作用是排除"只要稀疏/正则化就涨点"这一解释。等 M5 接入 EEGNet/torch 后，同一 arm 接口可以替换成真正的 dropout / 随机权重 mask。
- mask 的 seed 由 `derive_seed(seed, fold_id, "equiv_dropout")` 生成：每折不同、可复现、不使用标签。

## 3. 数据

### synthetic（默认，无需下载）
- `configs/m1_synthetic.yaml`：9 被试 × 2 session × 6 run × 4 类 × 4 trial；22 通道；250 Hz；0.5 s。
- 生成方式：类模板 + 被试偏移 + 噪声。类模板让分类器"有东西可学"，被试偏移模拟身份信息。
- **synthetic 的准确率没有科学意义**，只用于验证 split、Pipeline、结果表、日志。CLI 会打印警告；结果表的 `note` 里也带 `"dataset": "synthetic"`。

### BNCI2014-001（真实）
- 9 被试 / 22 EEG 导联 / 250 Hz / 2 session × 6 run / 4 类（左手、右手、双脚、舌头）/ 每类每 session 72 trial / 每被试 576 trial。
- 通过 MOABB `MotorImagery` 加载；固定 band-pass 8–32 Hz、resample 250 Hz（可在 YAML 改）。
- 安装：`pip install -r ../requirements-m1-moabb.txt`。Python 3.13 装不上时用 3.11/3.12 环境（本机已在 `anaconda3/envs/eeg` 装好 moabb 1.5.0）。
- **MOABB 1.5 的 Windows 路径坑**：绝对路径里的 `:` 会被 `_sanitize_path` 换成 `-`，下载会多套一层 `D-...` 假目录。`data.load_bnci2014_001` 已内置绕过（先 `os.chdir(data_path)` 再设 `MNE_DATASETS_BNCI_PATH="."`）。

### IF-1 `.fif`
- `data.load_epochs_fif(path)` 可读取 C 交付的 `mne.Epochs`；要求 `epochs.metadata` 至少包含 `label`，并尽量带 `subject/session/run/trial`。
- `EpochDataset.to_mne()` / `save_epochs_fif()` 可反向导出，便于和 C 对齐样例。

## 4. 运行

```bash
cd "03_代码/M1_评测管线"

# 只看切分
python run_m1.py --config configs/m1_synthetic.yaml --dataset synthetic --dry-run

# 只跑泄露检查，不训练
python run_m1.py --config configs/m1_synthetic.yaml --dataset synthetic --check-only

# synthetic 单折冒烟（两个 arm）
python run_m1.py --config configs/m1_synthetic.yaml \
    --dataset synthetic --arm baseline equiv_dropout --seed 0 --max-folds 1

# 真实数据：先跑一个被试/一个 seed 验证
python run_m1.py --config configs/m1_bnci2014_001.yaml \
    --dataset bnci2014_001 --arm baseline --seed 0 --subject 1

# 全量：两个 arm × 10 seeds × 9 folds（耗时较长）
python run_m1.py --config configs/m1_bnci2014_001.yaml --dataset bnci2014_001

# 主结果跑完后加做标签置换 canary
python run_m1.py --config configs/m1_bnci2014_001.yaml --dataset bnci2014_001 --canary
```

C 的 CLI 契约参数：`--arm`、`--seed`、`--subject`、`--dataset`。`--max-folds` 只用于调试，正式结果不要用它截断。

## 5. 输出

每次运行生成一个 `run_id = m1_YYYYmmdd_HHMMSS_<pid>`。

```
04_实验结果/
├── 结果表/
│   ├── results_<run_id>.csv       # IF-2 固定表头 + 每被试/每 session/整体
│   ├── results_<run_id>.json
│   └── predictions_<run_id>.csv   # 逐试次原始预测，给 B 做统计
└── 运行日志/<run_id>/
    ├── run.log
    ├── config.json / config.yaml
    ├── env.json
    ├── dataset_summary.json
    ├── dataset_fingerprint.json
    ├── splits.json                # 每折 train/test 的 subject 与位置索引
    ├── summary.json
    └── traceback.txt              # 仅出错时
```

### 结果表字段
`arm, model, subject, session, seed, balanced_acc, latency_ms, memory_MB, note, ablation_target, mask_id, activation_map_path`

- 每个被试产生：每个 session 一行 + `session="all"` 一行（整体）。
- `note` 是 JSON 字符串，含 `dataset/fold_id/n_train/n_test/fit_time_s/train_subjects/test_subjects`。
- `latency_ms` = 测试集单 trial 平均推理时间；`memory_MB` = `tracemalloc` 峰值（Python 分配，不是 GPU 显存）。
- `mask_id` 仅 `equiv_dropout` 非空，记录 density 与 mask seed。

### 逐试次预测字段
`arm, model, subject, session, run, trial, epoch_id, seed, fold_id, y_true, y_pred, y_score, label_true, label_pred`

B 可用这份文件自行做配对检验、多重比较校正、效应量；A 不改统计口径。

## 6. 泄露防线（与清单的对应）

| 防线 | 代码 | 清单项 |
|---|---|---|
| 被试不相交硬断言 | `common_utils.leakage_checks.assert_subject_disjoint` | A2/A5 |
| 测试被试全部 session 在测试侧 | `assert_all_subject_sessions_in_test` | A4/A8 |
| trial 不跨 split | `assert_trial_grouped` | A10 |
| 无重复 epoch | `assert_no_duplicate_epochs` | A1/A11 |
| 所有可训练变换只在训练折 fit | sklearn `Pipeline` + `build_preprocessor` | B1–B9 |
| 无测试折调参/早停 | M1 第一版固定超参；无 early stopping | C1–C3 |
| 所有 seed 报告，不选 best | `run_all` 遍历 `evaluation.seeds` | C4/D3 |
| 标签置换 canary | `evaluate.run_canary` + `--canary` | E3 |
| 每折 split 落盘 | `save_folds` → `splits.json` | A7 |
| 结果表/预测/日志 | `common_utils.results` / `RunLogger` | D4–D6 |

**没有实现/需要后续补的**：
- nested LOSO 调参（M1 第一版不调参；要调参时用 `make_nested_loso_splits`，并确认内层验证集与测试被试不相交）。
- 真正的训练时 dropout / EEGNet 臂（M5 或 M1 扩展）。
- 测试时自适应/在线归一化（M4/IF-5 再处理；当前主结果只用训练折统计量）。
- 严格的 95% CI / 统计检验（B 负责；A 只导出逐试次预测）。

## 7. 已知限制

1. **synthetic 数据不是科学结果**：类模板太容易分，准确率会接近 100%。它只证明管线通。
2. **CSP+LDA 只是 M1 基线**：M5 的移植验证要换 EEGNet/DeepConvNet；M1 的结论只说明"流程可复现"。
3. **`equiv_dropout` 是代理**：固定随机特征 mask，不是 literal dropout；M2/M5 接入神经网络后应替换。
4. **MOABB 已装于 `anaconda3/envs/eeg`**（Python 3.11.15 / moabb 1.5.0）。跑真实数据用 `.../envs/eeg/python.exe`；base 的 Python 3.13 未装 MOABB。
5. **内存**：当前 scaler 已避免全量 float64 拷贝；但 BNCI 全量 8 被试训练折仍然较大，建议在正常终端跑，不要在受限沙箱里跑全量。

## 8. 下一步

1. 安装 MOABB，跑 `--subject 1 --arm baseline --seed 0`，确认真实数据的形状/metadata。
2. 用 `--check-only` + `--dry-run` 核对 9 折的 subject/session 分布。
3. 跑 baseline 全量 9 折 × 1 seed，检查结果表与预测行数。
4. 接 `equiv_dropout` 全量；确认两臂 split/seed 完全一致。
5. 按 [M1 LOSO 泄露自查清单](../../06_会议与笔记/M1_LOSO_泄露自查清单.md) 做 A/B/C 类签核，再开正式多种子批跑。
