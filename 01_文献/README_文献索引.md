# 01_文献 — 索引说明

> 建立时间：2026-10-06 · 来源：Zotero「脑电EEG」集合（D:/Zotero/storage，已复制副本，Zotero原件未动）

## 目录结构

| 子目录 | 用途 | 对应任务书章节 |
|---|---|---|
| `01_果蝇全脑连接组/` | 果蝇全脑接线图原始论文与基于连接组的建模工作 | 论文1 的先验来源 |
| `02_EEG脑电解码与BCI/` | 脑电解码基准与主干网络 | M1 评测管线 / M5 移植验证 |
| `03_脉冲神经网络与学习规则/` | 三因子可塑性、资格迹、SNN 理论基础 | M4 可塑性概念验证 |
| `04_网络科学与图论/` | 网络 motif / 度序列等图论工具参考 | M2 先验注入（δ 阈值粒度） |
| `05_网页快照/` | 期刊网页快照（无 PDF 时备查） | 随时 |

## 文件 ↔ 任务书参考文献编号对照

### 01_果蝇全脑连接组
| 文件 | 任务书编号 | 说明 |
|---|---|---|
| Dorkenwald 等 2024 - Neuronal wiring diagram of an adult brain | **[1]** | 核心：成年果蝇全脑接线图（约14万神经元） |
| Schlegel 等 2024 - Whole-brain annotation and multi-connectome cell typing | **[2]** | 核心：全脑注释与多连接组细胞类型 |
| Lappalainen 等 2024 - Connectome-constrained networks（flyvis） | **[3]** | 前人工作：连接组约束网络预测果蝇视觉反应 |
| Shiu 等 2023 - A leaky integrate-and-fire computational model | **[4]** | 前人工作：全脑连接组 LIF 模型 |
| Vaxenburg 等 2025 - Whole-body physics simulation of fruit fly locomotion | **[9]** | 前人工作：flybody 深度强化学习身体仿真 |

### 02_EEG脑电解码与BCI
| 文件 | 任务书编号 | 说明 |
|---|---|---|
| Chevallier 等 2024 - The largest EEG-based BCI reproducibility study (MOABB benchmark) | **[6] 基准扩版** | M1 评测基准；30 管线 × 36 数据集；BNCI2014-001 各管线数字见精读笔记 |
| Jayaram 和 Barachant 2018 - MOABB: trustworthy algorithm benchmarking for BCIs | **[6] 原文献** | MOABB 原始论文，J. Neural Eng. 15 066011；2026-10-17 补齐 |
| Lawhern 等 2018 - EEGNet | **[12]** | M5 移植验证的主干网络；注意其 dropout 跨被试设 0.25、被试内 0.5 |
| Brookshire 等 2024 - Data leakage in deep learning studies of translational EEG | **[7]** | ⚠️ 数据泄露；任务书"约一半已发表研究虚高成绩"的出处。**M1 跑 LOSO 前必读**——跨被试场景的泄露形式与被试内不同 |

### 03_脉冲神经网络与学习规则
| 文件 | 任务书编号 | 说明 |
|---|---|---|
| Schmidgall & Hays 2022 - Learning to learn online with neuromodulated synaptic plasticity | **[10]** | ⚠️危险点② 资格迹的主要参考；须能讲清与动量的区别 |
| Bellec 等 2020 - A solution to the learning dilemma for recurrent networks of SNN | **[15]** | e-prop 原始论文（标题勿写成 "eligibility placeholder gradients"） |
| Maass 等 1997 - Networks of spiking neurons（网页快照） | **[13]** | SNN 理论基础 |

### 04_网络科学与图论
| 文件 | 任务书编号 | 说明 |
|---|---|---|
| Milo 等 2002 - Network Motifs（Science 网页快照） | **[14]** | M2 模式粒度、δ 阈值的方法学参考 |

### 05_网页快照
| 文件 | 对应 |
|---|---|
| Cross-subject generalization for EEG decoding: a survey of deep learning methods | **[8]** 跨被试综述；"83%→62%" 数据的出处 |

## 尚缺（任务书已列但Zotero 未下载）

- **[5]** FlyConnectome MaleCNS 全中枢连接组（male-cns.janelia.org，配套 Cell 2026-09）
- **[11]** networkx 度序列保持随机化函数文档（建议直接查官网最新版，注明版本号）
- 课题组已答复的三则补充文献：Klepl et al. IEEE TNSRE 2024;32:493-503（GNN 对照族综述）、GCNs-Net IEEE TNNLS 2023、Honegger et al. J Neurosci 2011;31(33):11772-11785 与 Campbell et al. 2013;33(25):10568-10581（KC 稀疏度5-10% 区间）

> **2026-10-17 更新**：[7] Brookshire、[6] 原文献 Jayaram & Barachant 已补齐入库；Chevallier 2024 的重复副本（MD5 一致）已清除。索引当前共 12 个 PDF + 3 个网页快照。
>
> **2026-10-18 更新**：[7] Brookshire 已完成逐节精读，笔记见 [Data_leakage_in_deep_learning_studies_of_translational_EEG.md](../20_Research/Papers/EEG脑电解码与BCI/Data_leakage_in_deep_learning_studies_of_translational_EEG.md)；据此生成的 [M1 LOSO 泄露自查清单](../06_会议与笔记/M1_LOSO_泄露自查清单.md) 已入库，M1 开跑前逐项签核。

## 纪律提醒（来自任务书§五）

- 理论问题禁止让 AI 自答（图论/SNN 理论会产出"错得很像对"的内容）→ 疑问记入 `../06_会议与笔记/问题清单/问题清单_A.md`
- 文献检索统一走「文献需求单」给导师，不要自行下载替代
- 新下载的文献：先存 Zotero（保持条目管理），再复制一份到对应子目录并在上表登记