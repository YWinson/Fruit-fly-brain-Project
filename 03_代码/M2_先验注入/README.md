# M2 先验注入臂

> 目标：把果蝇全脑连接组的**类型簇级接线规律**变成解码器的一条固定 64×32 层间掩码，并用三条零模型（度保持重连 / ER 随机 / 等效 dropout）把"生物结构本身"与"稀疏、容量、正则化"分开。
>
> 设计依据：`M2_先验注入1页设计草案_初稿.pdf`（2026-10-21 导师例会稿）。
> 当前状态（2026-10-22/24）：M2 首轮完成。CSP Stage 1 fine 3 seed × 9 折 × 5 arm、medium/coarse 各 1 seed、1,000-null δ_topo、EEGNet Stage 1 seed 0 × 9 折 × 5 arm、mask 密度（5/10/20%）与选择（balanced/global）消融（matched seed 0）均已跑完。首轮结论是**弱/阴性且对 Stage 1 强弱稳健**：CSP 下 bio 相对 deg_rewired / er_random 仅 +0.82pp / +1.08pp；EEGNet 下整体准确率升到 ~0.44，但 bio 仍不优于 deg/er（ER 反而显著更高）；密度/选择消融没有发现能达到 δ_acc=2.5pp 的组合。详见文末第 10 节与 `05_报告与交付/阶段报告/M2_实现与结果.md`。

## 1. 数据：FlyWire v783 类型级连接矩阵

### 原始数据

| 文件 | 来源 | 说明 |
|---|---|---|
| `Connectivity_783.parquet` | `github.com/philshiu/Drosophila_brain_model`（Shiu et al. 2023 模型仓库；FlyWire v783） | 15,091,983 条 proofread 神经元对边；`Connectivity` 为突触数，`Excitatory` 为 +1/−1 |
| `Supplemental_file1_neuron_annotations.tsv` | `github.com/flyconnectome/flywire_annotations`（Schlegel et al. 2024） | 139,248 个神经元、8,846 个 cell type；含 superclass / cell class / neurotransmitter |

数据放在 `02_数据/fly_connectome/raw/`（D 盘，`.gitignore` 已排除原始数据）。

### 构建

```bash
cd "03_代码"
python "M2_先验注入/connectome.py" --force
```

输出 `02_数据/fly_connectome/processed/`：

- `W_syn.npz` / `W_exc.npz` / `W_inh.npz` / `W_edge.npz`：8,846×8,846 稀疏类型矩阵；
- `type_meta.csv`：每个 cell type 的 superclass、cell class、neurotransmitter、神经元数、总突触数；
- `summary.json`：2,688,204 条边（≥5 突触阈值）、34.0M 突触、404,919 个非零类型对；12,309 条边未匹配到注释。

**过滤规则**：按草案 `Connectivity >= 5`；去掉无 `cell_type` 注释的神经元（137,720/139,248 有注释）。

## 2. 类型聚类与三级粒度

`sensorimotor/MB` 子集（草案范围）：superclass ∈ `{central, sensory, ascending, descending, motor, sensory_ascending, endocrine}`，排除 optic / visual_projection / visual_centrifugal。实测 8,155 个类型、52,267 个神经元、34.65M 突触。

| 粒度 | K_out | K_in | 扩展方式 | 用途 |
|---|---:|---:|---|---|
| fine | 64 | 32 | 1×1 | 主实验（终端类型簇） |
| medium | 32 | 16 | 行/列各重复 2 次 → 64×32 | 中粒度探针（功能群/半谱系） |
| coarse | 2 | 2 | 行重复 32、列重复 16 → 64×32 | 纯 E/I 阴性边界 |

- `fine` / `medium` 的聚类特征：每个类型对 sensorimotor 子集内 10 个 superclass 的出入突触分布（行归一化）+ 6 类神经递质 one-hot + log 神经元数/总突触数；Ward 层次聚类。特征只来自连接组，不使用 EEG。
- `coarse` 按 `top_nt` 合并：ACh/Glu → 兴奋，GABA → 抑制；其他默认兴奋（草案的 E/I 简化）。
- **64×32 映射**：fine 直接是 64 个输出簇 × 32 个输入簇；medium/coarse 通过行/列重复扩展到固定 64×32，保证所有 arm 的层维度与参数预算相同。这是对草案"将果蝇图谱排序对齐到 64×32"的显式操作化，设计草案没有规定唯一映射。

## 3. 掩码与五条对照臂

主实验密度 `p = 10%`（64×32 中保留 205 条边），辅测 5%/20% 可在 YAML 改。

| arm | 掩码 | 保持的图论不变量 | 排除的解释 |
|---|---|---|---|
| `bio_sparse` | 真实类型簇矩阵的 10% | 总边数 E、真实出入度、簇级 motif | 核心自变量 |
| `deg_rewired` | 对 `bio_sparse` 做 bipartite curveball 重连 | E + 每行/列度 100% 不变 | 仅靠度分布/Hub 异质性 |
| `er_random` | 64×32 中均匀随机 E 条边 | 仅 E | Lottery-ticket / 随便稀疏一张图 |
| `equiv_dropout` | 全连接 + 训练态连接 dropout（keep=10%） | E[active edges]=E | 稀疏只是免费 dropout 正则化 |
| `dense_baseline` | 全 1（100%） | 完整容量 | 无结构先验锚点 |

### 两个必须记录的设计选择

1. **balanced top-k，而不是全局 top-k**：全局 top-k 会产生近似 Ferrers/嵌套图，度序列几乎强制整张图；实测 curveball 重连的 Hamming 距离最多 184 < E=205，草案的危险点①断言无法通过。改成"按行轮转取最强边"后，行度更分散，重连 Hamming 达到 288–340 ≥ E，且仍保留"每行最强"的生物含义。`mask_selection: global` 可复现全局版本。
2. **coarse 不做 deg_rewired**：纯 E/I 2×2 块扩展后是块图，度序列强制块结构，任何度保持零模型都与原图同构；对 coarse 只跑 ER 对照，并在报告中标注 N/A。这不是代码失败，而是该粒度下零模型不可识别。

### 度保持重连的实现

- 使用 **curveball** 二部图随机化：取两行 i,j，令 A=仅 i 的列、B=仅 j 的列，重排 A∪B，使 i 保留 |A| 列。逐行/列度严格不变、无重边。
- 每次生成后执行：
  - `np.array_equal(mask.sum(1), rewired.sum(1))`
  - `np.array_equal(mask.sum(0), rewired.sum(0))`
  - `hamming >= E`
  - NetworkX 二部图度校验
- 违例抛 `MaskError`，不写 warning 后继续。

## 4. 解码器：三段式

```
Stage 1  X (22×T) -> 2 bands × 16 CSP = 32 维 z
Stage 2  z (32) -> W_eff = (W * Σ * M) -> ReLU (64)
Stage 3  h (64) -> 线性 4 类
```

- Stage 1（默认 `preprocess.stage1: csp`）：8–13 / 13–30 Hz 两个频带，各 16 个 CSP 分量；固定 Butterworth 滤波 + 训练折内 CSP + StandardScaler。Stage 1 **每折只 fit 一次**，所有 arm 共享，避免 5× 重复计算。
- Stage 1 可选 `eegnet`（`configs/m2_bnci2014_001_eegnet.yaml`）：compact EEGNet-8,2 trunk（kernel 32、F1=8/D=2/F2=16、40 epochs、batch 128、8 threads），每折在训练被试上训练一次，取 32 维 penultimate 特征后冻结；临时 4 类头丢弃，五条 arm 共享这 32 维特征。实测约 2.5–3 分钟/折。
- Stage 2：`M ∈ {0,1}^{64×32}`、`Σ ∈ {−1,+1}^{64×32}` 为固定 buffer；`W` 可训练，被 mask 的位置梯度为 0。所有 arm 使用同一初始化 seed，保证配对。
- Stage 3：4 类线性头。
- `equiv_dropout`：训练时对 64×32 连接做 inverted dropout，keep=0.10；推理时用全连接。
- 训练：Adam，100 epochs，batch 256，weight decay 1e-4，固定 epoch 数，无测试折早停。

## 5. δ_topo 与 δ_acc

- `motifs.py` 实现 FFL 与 bi-fan 计数；零模型用 `networkx.directed_edge_swap` 做度保持随机化（2×E 次三边交换），默认 1,000 个零模型，99% 分位数即 δ_topo。
- 结果包含 observed / null mean / null std / q99 / Z / empirical p，保存 `motif_calibration.json` 与 `motif_null.png`。
- δ_acc 按草案取 **2.0–3.0 pp**（`δ_acc = 0.025`）；正式配对 Wilcoxon 由 B 执行，A 只给逐试次预测与描述统计。

## 6. 运行

```bash
cd "03_代码"

# 构建/检查连接组与掩码
python "M2_先验注入/connectome.py"
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml" --dry-run
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml" --check-only

# 单折冒烟（5 arms）
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml" \
    --arm bio_sparse deg_rewired er_random equiv_dropout dense_baseline \
    --seed 0 --subject 1 --max-folds 1

# 全量（3 seeds × 9 folds × 5 arms）
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml"

# 粒度探针
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml" --granularity medium
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml" --granularity coarse

# δ_topo 校准（1,000 nulls，默认 joblib 全核）
python "M2_先验注入/run_m2.py" --config "M2_先验注入/configs/m2_bnci2014_001.yaml" --calibrate-motifs --dry-run
```

## 7. 输出

- 结果表：`04_实验结果/结果表/results_m2_<run_id>.csv|json`
- 逐试次预测：`04_实验结果/结果表/predictions_m2_<run_id>.csv`
- 运行日志：`04_实验结果/运行日志/m2_<run_id>/`
  - `config.yaml|json`、`env.json`、`splits.json`、`connectome_summary.json`、`granularity_stats.json`、`bio_mask_summary.json`
  - `masks/<arm>_fold<id>_seed<seed>.npz`（每条 arm 的实际 mask/sign）
  - `motifs/`（若运行校准）
- 结果表符合 IF-2 表头；`note` 里带 mask 的 `n_edges/density/hamming/granularity/selection` 与 Stage 1 / classifier 的 fit 时间。

## 8. 已知限制

1. **不是逐神经元连接组**：我们做的是 cell-type 级聚合 + 层次聚类，不是把 139k 神经元逐一对齐到 64×32；这是草案要求的"类型簇级"降维。
2. **64×32 映射是操作化选择**：fine 用 64 输出簇×32 输入簇；medium/coarse 用行/列重复扩展。草案没有规定唯一映射，报告中必须写清楚。
3. **coarse 的 deg_rewired N/A**：块图度序列强制结构；coarse 只与 ER 对照。
4. **Stage 1 是 2 bands × 16 CSP**：为让全量实验可跑，未用 4×8 或时空卷积；Stage 1 不是科学自变量，但仍需在报告中说明。
5. **synthetic 结果不代表科学结论**：仅用于管线冒烟。
6. **δ_topo 的零模型是 edge-swap 近似**：2×E 次三边交换，1,000 个样本；不是解析零模型。

## 9. 测试

```bash
cd "03_代码"
"../anaconda3/envs/eeg/python.exe" -m pytest tests/test_m2.py -q
```

覆盖：balanced mask 密度、curveball 度不变量与 Hamming、ER 边数、Masked MLP 输出/概率、MultiBandCSP 32 维、toy motif 计数、M2 config、synthetic 端到端单折。

## 10. 首轮 fine 结果（2026-10-22）

结果文件：`04_实验结果/结果表/results_m2_20261022_124200_0468.csv`（405 行；逐试次预测 77,760 行）。

| arm | balanced_acc（27 个 subject×seed） |
|---|---:|
| bio_sparse | **0.3807 ± 0.1063** |
| deg_rewired | 0.3724 ± 0.0988 |
| er_random | 0.3699 ± 0.0997 |
| equiv_dropout | 0.3601 ± 0.0987 |
| dense_baseline | 0.3625 ± 0.0932 |

配对差（按 subject 平均 3 seeds）：

| 比较 | mean diff | std | 初步 Wilcoxon p | ≥ δ_acc 2.5pp |
|---|---:|---:|---:|---|
| bio − deg_rewired | +0.0082 | 0.0109 | 0.0547 | 否 |
| bio − er_random | +0.0108 | 0.0282 | 0.3594 | 否 |
| bio − equiv_dropout | +0.0205 | 0.0425 | 0.2500 | 否 |
| bio − dense_baseline | +0.0182 | 0.0356 | 0.1289 | 否 |

**结论**：bio 在三个 seed 中都是最高，但相对两个零模型的优势只有 0.8–1.1pp，**未达预设 SESOI 2.5pp**；逐被试效应不稳定（S8/S9 驱动优势，S3/S7 的 deg_rewired 更高）。本轮**不支持**"果蝇类型级结构先验带来 ≥2.5pp 跨被试提升"；这是弱/阴性结果，不是正向证据。

仍待补：medium/coarse 粒度探针（1 seed）、1,000-null δ_topo 校准、正式统计（B）。

### 粒度探针（seed 0；medium/coarse 各 1 seed）

| 粒度 | bio | deg | er | equiv | dense | bio − er |
|---|---:|---:|---:|---:|---:|---:|
| fine | 0.3785 | 0.3735 | 0.3644 | 0.3565 | 0.3586 | +1.41pp |
| medium | 0.3627 | 0.3686 | 0.3644 | 0.3565 | 0.3586 | −0.17pp |
| coarse | 0.3414 | N/A | 0.3644 | 0.3565 | 0.3586 | −2.30pp |

- `er_random` / `equiv_dropout` / `dense_baseline` 在三档粒度下完全相同（0.3644 / 0.3565 / 0.3586），因为它们不依赖聚类；这是管线 sanity check。
- 只有 fine 的 bio 高于 ER（+1.41pp）；medium 持平；coarse 明显低于 ER（−2.30pp）。方向符合"细粒度有一点信号、粗化会丢失、纯 E/I 有害"，但 fine 的增益仍 **低于 δ_acc=2.5pp**。

### δ_topo（1,000 个度保持零模型）

细粒度 square graph：64×64、2,583 条边、63 个非孤立节点；`networkx.directed_edge_swap`，2×E 次三边交换，1,000 个 null。

| motif | observed | null mean | null 99% (δ_topo) | Z | empirical p |
|---|---:|---:|---:|---:|---:|
| FFL | 100,052 | 99,058.3 | 99,245.0 | 13.00 | 0.0 |
| bi-fan | 1,173,997 | 1,163,228.8 | 1,165,262.4 | 12.85 | 0.0 |

- 真实类型簇图的 motif 结构**显著高于度保持零模型**；但这**没有转化为解码增益**（bio 相对 deg/er 只有 0.8–1.1pp，低于 δ_acc）。
- 结论要分开写：**结构显著性 ≠ 任务收益**。M2 的负结果更可能是"类型级结构在 64×32 掩码/当前 Stage 1 上没有转化为跨被试优势"，而不是"连接组没有结构"。

### Stage 1 对照：EEGNet（seed 0）

Stage 1 换成 compact EEGNet-8,2 trunk（kernel 32、F1=8/D=2/F2=16、40 epochs、batch 128、8 threads），每折训练一次并冻结 32 维特征；五条 arm 共享特征。`stage1_fit_time_s ≈ 143 s/折`。

| arm | balanced_acc（9 subjects, seed 0） |
|---|---:|
| bio_sparse | 0.4406 ± 0.1494 |
| deg_rewired | 0.4414 ± 0.1534 |
| er_random | 0.4479 ± 0.1512 |
| equiv_dropout | 0.4377 ± 0.1524 |
| dense_baseline | 0.4450 ± 0.1533 |

- bio − deg：−0.08pp（p=0.94）；bio − er：**−0.73pp（p=0.035）**；bio − equiv：+0.29pp；bio − dense：−0.44pp。
- EEGNet 把整体准确率从 CSP 的 ~0.38 提到 ~0.44，但 **bio 仍不优于零模型**；ER 反而显著更高。负结果不是 CSP Stage 1 太弱造成的。
- 限制：EEGNet 只跑了 seed 0；跨 Stage 1 比较是描述性的。

### Mask 密度消融（CSP Stage 1，matched seed 0）

| 密度 | bio | deg_rewired | er_random | equiv_dropout | dense_baseline | bio−deg | bio−er |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.3663 | 0.3598 | 0.3650 | 0.3372 | 0.3586 | +0.66pp | +0.14pp |
| 10% | 0.3785 | 0.3735 | 0.3644 | 0.3565 | 0.3586 | +0.50pp | +1.41pp |
| 20% | 0.3785 | 0.3756 | 0.3665 | 0.3628 | 0.3586 | +0.29pp | +1.20pp |

- 5% 太稀疏；10% 与 20% 是平台。
- bio−deg 在所有密度下 <0.7pp；bio−er 在 10–20% 为 1.2–1.4pp，仍 < δ_acc=2.5pp。
- 5%/20% 的 `deg_rewired` 均通过 Hamming ≥ E；10% global top-k 的 deg N/A（Hamming <205），只能比较 bio vs ER。

### 映射选择（10%，balanced vs global top-k，matched seed 0）

| 选择方式 | bio | deg_rewired | er_random | equiv_dropout | dense_baseline | bio−deg | bio−er |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | **0.3785** | 0.3735 | 0.3644 | 0.3565 | 0.3586 | +0.50pp | +1.41pp |
| global | 0.3754 | N/A | 0.3644 | 0.3565 | 0.3586 | N/A | +1.10pp |

- global top-k 的 9 折 `deg_rewired` 全部 N/A（Hamming 172–192 < E=205）；ER/equiv/dense 与 balanced 相同。
- balanced 比 global 的 bio 高 0.31pp；global 没有额外收益，bio−er 1.10pp 仍低于 δ_acc。
- 结论：balanced top-k 是更可用的选择方式；global 不会让生物结构产生 δ_acc 级优势，反而让度保持零模型不可识别。
