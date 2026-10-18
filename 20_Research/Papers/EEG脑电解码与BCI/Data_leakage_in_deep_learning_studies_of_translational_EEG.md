---
document_id: "file-fd0f48f1d1ff52a6"
source_type: "local_pdf"
source_url: ""
title: "Data leakage in deep learning studies of translational EEG"
authors: "Geoffrey Brookshire; Jake Kasper; Nicholas M. Blauch; Yunan Charles Wu; Ryan Glatt; David A. Merrill; Spencer Gerrol; Keith J. Yoder; Colin Quirk; Ché Lucero"
published: "2024-05-03"
domain: "EEG脑电解码与BCI"
tags:
  - "论文笔记"
  - "EEG脑电解码与BCI"
  - "local_pdf"
quality_score: "8.5"
related_papers: []
created: "2026-10-18"
updated: "2026-10-18"
status: "analyzed"
---

# Data leakage in deep learning studies of translational EEG

## 核心信息
- **文档 ID**：file-fd0f48f1d1ff52a6
- **来源类型**：local_pdf
- **作者**：Geoffrey Brookshire; Jake Kasper; Nicholas M. Blauch; Yunan Charles Wu; Ryan Glatt; David A. Merrill; Spencer Gerrol; Keith J. Yoder; Colin Quirk; Ché Lucero
- **机构**：SPARK Neuro Inc.（New York）；Carnegie Mellon University；Pacific Brain Health Center / Pacific Neuroscience Institute；Saint John's Cancer Institute；UCLA
- **发布时间**：2024-05-03（收稿 2024-01-19；接收 2024-04-04）
- **会议/期刊/发布方**：Frontiers in Neuroscience 18:1373515（Brief Research Report）
- **链接**：[DOI](https://doi.org/10.3389/fnins.2024.1373515)；[medRxiv 预印本](https://www.medrxiv.org/content/10.1101/2024.01.16.24301366v1)
- **引用**：被引次数未核（--）

## 摘要与核心要点

### 原文摘要
A growing number of studies apply deep neural networks (DNNs) to recordings of human electroencephalography (EEG) to identify a range of disorders. In many studies, EEG recordings are split into segments, and each segment is randomly assigned to the training or test set. As a consequence, data from individual subjects appears in both the training and the test set. Could high test-set accuracy reflect data leakage from subject-specific patterns in the data, rather than patterns that identify a disease? We address this question by testing the performance of DNN classifiers using segment-based holdout (in which segments from one subject can appear in both the training and test set), and comparing this to their performance using subject-based holdout (where all segments from one subject appear exclusively in either the training set or the test set). In two datasets (one classifying Alzheimer's disease, and the other classifying epileptic seizures), we find that performance on previously-unseen subjects is strongly overestimated when models are trained using segment-based holdout. Finally, we survey the literature and find that the majority of translational DNN-EEG studies use segment-based holdout. Most published DNN-EEG studies may dramatically overestimate their classification performance on new subjects.

**关键词**：electroencephalography, deep neural networks, data leakage, cross-validation, Alzheimer's disease, epilepsy

### 中文概述
这篇论文用一个很干净的对照实验说明：当 EEG 被切成很多短片段、再按片段随机分训练/测试集时，同一个被试的片段会同时出现在两边；模型可以靠"认出这个人"而不是"识别疾病"拿到高分。作者固定已发表的两个 CNN 架构与数据处理流程，只改变切分方式：在阿尔茨海默病（AD vs SCI）数据集上，片段级切分的测试准确率是 99.8%，改成被试级切分后跌到 53.0%（二分类 chance = 50%）；在癫痫片段检测数据集上，79.1% 跌到 65.1%。随后作者检索 63 篇 2018 年以来的平移性 DNN-EEG 研究，只有 17 篇（27.0%）明确采用被试级切分。结论很直接：**受试者必须是切分的最小单位；同一受试者的所有片段只能出现在训练、验证或测试中的一个分区。**

### 核心要点
- **问题**：EEG 样本少，研究者常把长记录切成短片段来扩样本；若按片段随机分训练/测试集，同一被试会跨分区出现，造成 subject-specific data leakage（也称 identity confounding）。
- **方法**：复用两篇已发表研究的 CNN 架构与预处理，不调参、不做模型选择，只比较 segment-based holdout 与 subject-based holdout。
- **主要结果**：AD 99.8% → 53.0%（Wilcoxon T=0.0, p=0.002）；癫痫 79.1% → 65.1%（T=0.0, p=0.0001）。
- **文献现状**：63 篇中仅 17 篇（27.0%）明确按被试切分；其余 46 篇（73.0%）为 segments、both 或 unclear。
- **研究意义**：跨被试泛化的真实性能可能被严重高估；被试级切分是报告"新受试者性能"的最低要求。

## 研究背景与问题

### 背景
- DNN 在 EEG 疾病分类中被大量使用；EEG 记录通常只有几十到几百名被试，但每条记录长达数分钟到数小时。
- 为了增加训练样本，常见做法是把连续 EEG 切成 1–2 秒的片段，每个片段作为一个独立观测。
- 同一被试的片段之间比不同被试的片段更相似（EEG "fingerprinting"；Demuru & Fraschini 2020）。因此，如果同一被试的片段同时出现在训练集和测试集，模型可以学习"这个人的特异模式 → 标签"的捷径，而不是疾病本身的通用模式。
- 这种泄露在 MRI、OCT、手机传感器、语音等多个领域已被记录；EEG 因为片段化最普遍，风险尤其高。

### 问题定义
论文把 data leakage 定义为：**测试集的信息出现在训练集中，导致模型性能被正向偏置**（Kaufman et al. 2012）。本文聚焦其中一种：subject-specific leakage，即同一受试者的数据跨训练/测试分区出现。

### 适用边界
- 作者在 Discussion 中明确：同样的原则适用于 task-based EEG，不只 resting-state；也不只 AD/癫痫。
- 但泄露的**具体幅度**会随被试数、模型架构、任务细节变化；被试越少，偏置可能越大。
- 本文只处理"切分方式"造成的泄露；**没有**系统覆盖预处理、特征选择、超参数选择、归一化、数据增强等其他泄露形式。这是我们在 M1 中必须自己补的部分。

## 方法概述

### 核心方法
1. **Experiment 1（AD vs SCI，between-subject 任务）**
   - 数据：Pacific Neuroscience Institute 提供的静息态 EEG；49 名 AD 患者 + 49 名 SCI 对照；19 导联 10-20 系统；250 Hz；仅用 eyes-open resting-state；低通 <125 Hz；切成 2 s（500 样本）非重叠片段。
   - 架构：复用 Oh et al. (2020) 的 1D CNN；4 个卷积层（每层 ReLU + max pooling + batch norm）→ 20/10 两个全连接层 → 2 类输出。
   - 训练：70 epochs，无 early stopping、无超参调优；batch 32；Adam，lr=0.0001。
2. **Experiment 2（seizure detection，within-subject 事件任务）**
   - 数据：Siena Scalp EEG Database（PhysioNet）；14 名癫痫患者；47 次发作；约 128 h 记录；512 Hz；10-20 系统。
   - 预处理：256 样本（0.5 s）窗、128 样本（0.25 s）重叠、Hann taper 计算频谱图；切成 1.5 s 片段；用 viridis 色图转成 224×224×3 RGB 图像。
   - 架构：复用 Rashed-Al-Mahfuz et al. (2021) 的 2D CNN；4 个卷积层（ReLU + pooling + batch norm）→ 256/512 全连接层 → dropout → 2 类输出。
   - 训练：70 epochs，无 early stopping；batch 32；RMSProp，lr=0.00001。
3. **两种切分**
   - **Segment-based holdout**：把所有片段视为等价，不考虑 subject ID，直接按片段索引随机分成 k 折。
   - **Subject-based holdout**：先取唯一 subject 向量 \(s_u\)，把 \(s_u\) 分成 k 个不重叠的块；每个块作为验证折，其余块训练。这样同一被试的所有片段只出现在一个分区。
4. **文献回顾**：Google Scholar 检索 AD、Parkinson's、ADHD、depression、schizophrenia、seizures 六类疾病；纳入 2018 年以来使用 DNN + EEG 的 63 篇；按论文描述把切分方式编码为 segments / subjects / both / unclear。

### 关键公式
设共有 \(n\) 个片段、每个片段 \(m\) 个时间点，构造 \(X \in \mathbb{R}^{n \times m}\) 与标签向量 \(y \in \mathbb{R}^n\)。

**Segment-based holdout**：把索引向量 \(\alpha = \{1,\dots,n\}\) 分成不相交的 \(\alpha_{train}\) 与 \(\alpha_{test}\)：

\[
X_{train} = \{X_i\}_{i \in \alpha_{train}}, \quad
X_{test} = \{X_i\}_{i \in \alpha_{test}}, \quad
y_{train} = \{y_i\}_{i \in \alpha_{train}}, \quad
y_{test} = \{y_i\}_{i \in \alpha_{test}}
\]

**Subject-based holdout**：引入 subject 向量 \(s\)，只对唯一 subject 集合 \(s_u\) 做划分，并把每个 subject 的全部片段收集到同一分区，强制：

\[
s_i \neq s_j \quad \forall i \in \alpha_{train},\ j \in \alpha_{test}
\]

也就是说，**subject 是分组变量（group），不是普通样本**。

### 方法架构与图片
本文核心图在 PDF P5–P6：Figure 1 是 segment-based 与 subject-based 切分的示意（三个被试、每人三个片段、3 折 CV）；Figure 2 是两种切分下的测试准确率箱线图；Figure 3 是逐 epoch 的训练/测试曲线；Figure 4 是 63 篇文献的切分方式统计。本笔记未提取图片，建议直接看 PDF 对应页。

## 实验与证据

- **数据集/任务**：
  - AD vs SCI：98 名被试（49 AD + 49 SCI），19 导联静息态 EEG，2 s 片段，二分类。
  - Seizure vs non-seizure：14 名癫痫患者，47 次发作，1.5 s 频谱图片段，二分类。
- **基线**：不是"另一个模型"，而是**另一种切分方式**。两实验都复用已发表架构，不做模型选择/调参，以确保差异只来自 split。
- **指标**：test-set accuracy（含 95% CI）；跨折比较用 Wilcoxon 符号秩检验。
- **主要结果**：

| 实验 | Segment-based | Subject-based | 变化 | 统计检验 |
|---|---:|---:|---:|---|
| AD vs SCI（between-subject） | 99.8%（99.1–100.0） | 53.0%（43.1–64.8） | −46.8 pp | Wilcoxon T=0.0, p=0.002 |
| Seizure（within-subject 事件） | 79.1%（78.8–79.4） | 65.1%（61.3–69.1） | −14.0 pp | Wilcoxon T=0.0, p=0.0001 |

  - AD 的 subject-based 95% CI 包含 50% chance；也就是说，**在新被试上几乎等于随机**。
  - 癫痫任务的下降幅度较小，但依然显著；作者解释：within-subject 任务里模型不能简单"给每个人贴标签"，但仍可能学到被试特异的发作模式，或同一事件相邻片段之间的相关性。
- **文献回顾**：63 篇中 17 篇（27.0%）明确使用 subject-based holdout；其余 46 篇（73.0%）为 segments、both 或 unclear。作者结论："Most published DNN-EEG studies may dramatically overestimate their classification performance on new subjects."
- **消融与稳健性**：
  - 作者没有做传统消融；他们的"消融"就是改变 split。
  - 两个数据集、两种任务（between-subject 疾病分类 vs within-subject 事件检测）都复现了偏置，说明现象不限于单一任务。
  - 作者明确列出 caveats：泄露幅度随被试数、架构、实验细节变化；被试越少偏置可能越强；能更好学习 subject-specific 表示的架构可能偏置更大。

## 深度分析

### 核心贡献与创新
1. **把"切分泄露"做成了干净的因果对照**：固定数据、架构、预处理、训练轮数，只改 split；因此性能差异可以归因于 split，而不是模型容量或调参。
2. **同时覆盖 between-subject 与 within-subject 事件任务**：说明即使任务本身是"在被试内找事件"，只要同一被试跨分区，仍然会泄露；只是幅度小一些。
3. **给出了文献层面的 prevalence**：63 篇中只有 27% 明确安全；这把"个别研究的不严谨"升级为"领域性问题"。
4. **给出了极简、可执行的建议**：同一受试者的所有片段只能进入 train / validation / test 中的一个分区。

### 优势
- 实验设计非常干净：复用已发表架构、不调参、不做模型选择，排除了"模型更好"或"调参更好"的替代解释。
- 效应量巨大且直观：AD 从 99.8% 跌到 53.0%，几乎是从"完美"跌到"随机"。
- 两个数据集、两种任务互相印证；文献回顾补充了外部效度。
- 作者主动报告 caveats，没有把结论外推到所有 EEG 场景。
- 数据/代码可获取（Siena 公开；AD 数据需协议；代码在 Supplementary material）。

### 局限与风险
**作者明确承认的：**
- 泄露的具体幅度随实验细节变化；被试越少，偏置可能越大。
- 模型架构会影响泄露程度；不同架构学习 subject-specific 表示的能力不同。
- within-subject 事件的下降幅度小于 between-subject 疾病分类。
- 文献回顾是 non-exhaustive；编码依赖论文对切分方式的描述，可能存在"unclear"或漏报。

**我们的分析（明确标记为推断）：**
- 本文只覆盖 **split-level leakage**。一个完整的 LOSO 管线还可能在以下地方泄露：全局标准化、全局 CSP/PCA/ICA、用全部被试选通道、用测试折调超参/早停、数据增强跨折、用测试被试的统计量做归一化、用测试标签做阈值/类别权重。Brookshire 的"subject-based holdout"是**必要条件，不是充分条件**。
- 本文的任务是 AD/癫痫，不是运动想象（MI）。机制（subject-specific patterns）对 task-based EEG 同样成立（作者明确说 "the same principles of data leakage will apply to task-based EEG"），但 MI 的具体泄露幅度、以及 MI 中常见的 CSP/ERD 特征是否更/更少受身份信息影响，需要 M1 自己的对照来确认。
- 文献回顾把"每个被试单独训练一个模型"也标为 segment-based；这在临床落地角度是合理的（不能泛化到新被试），但如果研究目标本来就是 subject-specific 解码，标签需要更细。M1 的目标是跨被试 LOSO，因此必须用 subject-based。
- 论文没有给出"泄露检测"的统计检验或代码工具；我们只能把它的原则转成工程检查（见 M1 清单的代码级防呆）。

### 适用场景
- **适用**：任何"每个被试贡献多个片段/试次"的 EEG 研究；尤其是声称"跨被试泛化""新患者诊断""临床可用"的 DNN 研究。
- **不适用/需补充**：只做被试内解码且明确不声称跨被试泛化的研究（仍需注意 trial/window 级泄露）；只做离线可视化/探索的研究（但报告性能时仍应说明 split）；以及需要审计预处理/模型选择泄露的场景——本文不覆盖这些，需另建清单。
- **对 M1 的直接含义**：LOSO 是 subject-based holdout 的一个特例；M1 的主结果必须是"测试被试完全未参与任何 fit"的 LOSO，且所有可训练预处理都只能在训练折内 fit。

## 相关工作与技术路线
- [[flyvis_Lappalainen2024_逐页精读笔记|flyvis 精读笔记]] — 同样是"用严格对照把方法起作用的真正原因分离出来"的研究范式；flyvis 用 random DMN / 类型粗化等对照，Brookshire 用 split 对照。两者都提醒：**主结果好看不等于机制成立**。
- [[README_文献索引|文献索引]] — 本文是索引中的 [7]，被标注为 "M1 跑 LOSO 前必读"。
- [[M1_LOSO_泄露自查清单|M1 LOSO 泄露自查清单]] — 把本文原则转成 M1 开工前的逐项检查。

## 综合评价

| 维度 | 分数 | 理由 |
|---|---:|---|
| 创新性 | 7/10 | subject-specific leakage 在 MRI/OCT/传感器领域已有先例；本文的新意在于 EEG 领域的干净复现 + 文献 prevalence，而非提出全新概念。 |
| 技术质量 | 8/10 | 固定架构、不调参、只改 split，因果解释强；但只覆盖 split-level leakage，未系统审计预处理/模型选择泄露。 |
| 证据充分性 | 8/10 | 两个数据集、两种任务、大效应量、显著检验；文献回顾非穷尽，且任务不是 MI。 |
| 写作质量 | 9/10 | 结构清晰、结论克制、caveats 明确；一页就能读懂核心。 |
| 实用性 | 10/10 | 直接给出可执行规则；对 M1 的 LOSO 设计是必读的最低标准。 |

**综合：8.5/10。** 这不是一篇方法创新论文，而是一篇"领域纠错"论文；它的价值在于把一条容易被忽视的工程纪律变成了有数字、有对照、有文献证据的硬结论。

## 我的笔记

### 对 M1 的直接影响（5 条）
1. **LOSO 的切分单位必须是被试，不是 trial/segment。** 测试被试的两个 session、全部 run、全部 trial 都只能进测试折。
2. **Brookshire 只保证 split 正确，不保证管线正确。** M1 还要单独审计：全局标准化、CSP/PCA/ICA、通道选择、超参/早停、数据增强、测试时归一化。
3. **BNCI2014-001 有 2 个 session。** 主 LOSO 中测试被试的两个 session 都要留在测试侧；不能出现"同一被试 session 1 训练、session 2 测试"却把它叫作跨被试。
4. **MOABB 的 `CrossSubject` 只负责 split。** 所有可训练预处理必须放进 sklearn `Pipeline`，在每折内 fit；不要在外面先对全部被试 `fit_transform`。
5. **证据边界要写清楚。** 本文是 AD/癫痫，不是 MI；引用时用它支持"被试级切分是必要条件"，不要声称它直接证明了 MI 的泄露幅度。

> 逐项核查见 [[M1_LOSO_泄露自查清单|M1 LOSO 泄露自查清单]]。

## 外部资源
- [DOI / 原文](https://doi.org/10.3389/fnins.2024.1373515)
- [medRxiv 预印本](https://www.medrxiv.org/content/10.1101/2024.01.16.24301366v1)
- [Siena Scalp EEG Database (PhysioNet)](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- [Frontiers Supplementary Material](https://www.frontiersin.org/articles/10.3389/fnins.2024.1373515/full#supplementary-material)
- 代码：作者称复现代码在 Supplementary material；未发现独立 GitHub 仓库（按原文）。
