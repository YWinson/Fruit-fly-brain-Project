# M1 LOSO 数据泄露自查清单

> **适用**：M1 评测管线（BNCI2014-001，LOSO，无结构基线臂 + 等效 dropout 臂）
> **依据**：Brookshire et al. 2024, *Data leakage in deep learning studies of translational EEG*, Front. Neurosci. 18:1373515（[论文笔记](../20_Research/Papers/EEG脑电解码与BCI/Data_leakage_in_deep_learning_studies_of_translational_EEG.md)）
> **补充依据**：MOABB / BNCI2014-001 数据规格；任务书§三 M1、§四 IF-1/IF-2；工作区铁律
> **使用方式**：M1 开跑前逐项核查。每项填 `✅ 通过 / ❌ 不通过 / N/A`，并附证据路径（代码行、日志、命令输出）。**A/B/C 类任一 ❌，不得开跑主实验。**
> **版本**：v1.0 · 2026-10-18 · 自查人：____ · 复核人（B）：____
>
> **代码对应（2026-10-18）**：`assert_subject_disjoint` / `assert_all_subject_sessions_in_test` / `assert_trial_grouped` / `assert_no_duplicate_epochs` 已实现于 `03_代码/common_utils/leakage_checks.py`；toy 单测与标签置换 canary 见 `03_代码/tests/`。E1/E2/E3/E4 已可用；真实数据上仍需逐项签核。

---

## 0. 先看证据：Brookshire 到底证明了什么

### 0.1 核心数字

| 实验 | Segment-based holdout | Subject-based holdout | 变化 | 检验 |
|---|---:|---:|---:|---|
| AD vs SCI（between-subject 疾病分类） | 99.8%（99.1–100.0） | 53.0%（43.1–64.8） | −46.8 pp | Wilcoxon T=0.0, p=0.002 |
| 癫痫片段检测（within-subject 事件） | 79.1%（78.8–79.4） | 65.1%（61.3–69.1） | −14.0 pp | Wilcoxon T=0.0, p=0.0001 |

- AD 的 subject-based 95% CI 包含 50% chance：在新被试上几乎等于随机。
- 文献回顾：63 篇 2018 年以来的平移性 DNN-EEG 研究中，**仅 17 篇（27.0%）明确使用 subject-based holdout**；其余 46 篇（73.0%）为 segments / both / unclear。

### 0.2 一句话原则

> **同一受试者的所有片段/试次，只能出现在训练、验证或测试中的一个分区。**

论文原文："all segments from a given subject must be assigned to only a single partition of the data (i.e., train or validation or test)."

### 0.3 证据边界（必须写进报告，防止过度引用）

- 本文任务是 **AD/癫痫**，不是运动想象（MI）。作者明确说同样的原则适用于 task-based EEG，但 **MI 的具体泄露幅度没有在这篇论文里测过**。
- 本文只覆盖 **split-level leakage**（切分造成的泄露）。它**不覆盖**：全局标准化、全局 CSP/PCA/ICA、用全部被试选通道、用测试折调超参/早停、数据增强跨折、用测试被试统计量做归一化、用测试标签做阈值/类别权重。
- 因此：**subject-based LOSO 是必要条件，不是充分条件。** 本清单在 Brookshire 之外补齐了预处理、调参、增强、报告四类检查。

---

## 1. M1 的切分规格（先把规则写死，再写代码）

| 项目 | M1 规定 |
|---|---|
| 数据集 | BNCI2014-001（MOABB 下载；版本/哈希记录在案） |
| 被试数 | 9 |
| 数据规模 | 22 EEG 导联（以 MOABB 实际加载为准）；250 Hz；2 session × 6 run；4 类（左手/右手/双脚/舌头）；每类每 session 72 trial；每被试 576 trial |
| 主评估 | **LOSO（Leave-One-Subject-Out）**，9 折；每折训练 8 被试、测试 1 被试 |
| 测试折 | 测试被试的 **2 个 session、全部 run、全部 trial** 都进测试；训练折不含该被试任何 trial |
| 验证集 | 优先 nested LOSO（在 8 个训练被试内再留一）；若 M1 第一版不调参，则使用固定超参，不设验证折 |
| 随机种子 | 固定列表（建议 0–9），所有 arm 共用同一组 seed；报告全部 seed，不报 best seed |
| 主指标 | balanced accuracy（4 类 chance = 25%）；每被试一值 + 均值 ± 标准差 + 95% CI |
| 对照臂 | 无结构基线臂 vs 等效 dropout 臂；**除 arm 特定机制外，split、seed、预处理、训练预算完全一致** |
| 结果表 | 固定表头 `arm/model/subject/session/seed/balanced_acc/latency_ms/memory_MB/note` + 消融字段；CSV + JSON |
| 原始预测 | 每条 trial 的预测值单独导出，供 B 做统计；A 不改统计口径 |

> **红线**：M1 主结果中，任何"按 trial / 按 segment / 按 window 随机划分"的做法都不允许。被试是唯一分组变量。

---

## 2. A 类：切分与分组（最高优先级）

### A1. 锁定数据版本与元数据
- [ ] 记录数据集名称、MOABB 版本、下载日期、文件哈希（SHA256）。
- [ ] 建立元数据表：`subject_id / session_id / run_id / trial_id / class / split`，每个 trial 一行。
- [ ] 核对：9 被试 × 2 session × 6 run × 4 类 × 72 trial = 576 trial/被试；总数与 MOABB 返回一致。
- **怎么查**：`data_card.md` + 哈希命令输出 + `metadata.csv` 行数/唯一值统计。
- **通过标准**：元数据表可独立重建每个 fold；没有缺失 subject/session/trial ID。

### A2. 被试是唯一分组变量
- [ ] 代码中 split 函数只接受 `subject_id` 作为 group，不接受 trial/segment/window 作为 group。
- [ ] 禁止用 `train_test_split(..., random_state=...)` 直接切 trial；必须用 `GroupKFold` / `LeaveOneGroupOut` / MOABB `CrossSubject`。
- **怎么查**：`grep -n "train_test_split\|KFold\|GroupKFold\|LeaveOneGroupOut"`；逐处确认 group 参数。
- **通过标准**：主实验路径中不存在无 group 的随机切分。

### A3. 主实验严格 LOSO，9 折
- [ ] 9 个 fold，每折恰好 1 个测试被试、8 个训练被试。
- [ ] 每个被试恰好被测试 1 次；9 折并集覆盖全部被试。
- **怎么查**：打印每折 `train_subjects` / `test_subjects`；用集合运算验证。
- **通过标准**：`len(folds)==9`；每个 subject 在 test 中出现且仅出现 1 次。

### A4. 测试被试的两个 session 全部在测试侧
- [ ] 测试折包含该被试全部 session（BNCI2014-001 共 2 个；以数据卡的实际 session 编码为准）的全部 trial。
- [ ] 训练折不含该被试任何 session 的 trial。
- **怎么查**：按 `subject_id` 分组后检查 `set(test.session_id) == set(all_sessions_of_that_subject)`；检查 `train` 中无该 subject。
- **通过标准**：测试被试的 session 集合完整；训练侧无该 subject。

### A5. 训练/测试被试集合不相交（硬断言）
- [ ] 每个 fold 开跑前执行：
  ```python
  assert set(train_subjects).isdisjoint(set(test_subjects))
  ```
- [ ] 断言失败时直接抛异常，不写"warning 后继续"。
- **怎么查**：代码审查 + 单测。
- **通过标准**：任何 fold 都不可能带着交集进入训练。

### A6. 同一被试的全部 trial 在同一分区
- [ ] 不允许"同一被试的一部分 trial 训练、另一部分测试"。
- [ ] 如果做被试内 sanity check，单独命名（如 `within_subject_sanity`），**不得混入主 LOSO 结果表**。
- **怎么查**：检查每个 `trial_id` 的 subject 是否只映射到一个 split。
- **通过标准**：`subject_id -> split` 是单值映射。

### A7. 保存每折 split index
- [ ] 每折保存 `train_trial_ids.npy` / `test_trial_ids.npy`（或 CSV）到 `04_实验结果/运行日志/<run_id>/splits/`。
- [ ] 结果表能通过 `run_id + fold` 反查到具体 trial。
- **怎么查**：重新加载 split 文件，重算 balanced accuracy，与结果表一致。
- **通过标准**：任何人可用 split 文件复现同一折的评估。

### A8. 主 LOSO 中没有 session 级泄露
- [ ] 不出现"同一被试 session 1 训练、session 2 测试"却把它算作跨被试结果。
- [ ] 如果要做 cross-session 实验，单独命名、单独报告，不与 LOSO 混表。
- **怎么查**：按 subject 汇总 session 分布；检查 `CrossSubject` 与 `CrossSession` 是否被混用。
- **通过标准**：主结果表的 `split` 字段只标 `LOSO`。

### A9. 验证集与训练/测试都 subject-disjoint
- [ ] 如果调参/早停：验证集来自训练被试，且与测试被试不相交；推荐 nested LOSO（外层 9 折，内层在 8 个训练被试中留一）。
- [ ] 如果 M1 第一版不调参：明确写"固定超参，无验证折"，并记录超参来源（文献/MOABB 默认）。
- **怎么查**：打印内层/外层 subject 集合；检查验证集 subject 是否出现在测试折。
- **通过标准**：`set(val_subjects) ∩ set(test_subjects) == ∅`。

### A10. 重叠窗口/滑动窗必须按 trial 分组
- [ ] 如果一个 trial 被切成多个重叠 window，所有 window 必须同属一个 split。
- [ ] 禁止按 window 随机分 train/test；这会制造同一 trial 的"近重复样本"跨分区。
- **怎么查**：检查 window 索引是否带 `trial_id`；用 `GroupKFold` 以 `trial_id` 为 group 做窗口级 CV（如果必须做窗口级实验）。
- **通过标准**：同一 `trial_id` 的所有 window 只出现在一个分区。

### A11. 检查重复被试/重复 trial
- [ ] 检查 `subject_id` 是否有重复/别名（如 `sub-01` vs `01`）。
- [ ] 检查 trial 是否有完全重复的行（哈希/相关性近 1）。
- [ ] 如果使用多个数据集拼接，确认没有同一被试跨数据集重复。
- **怎么查**：`duplicated()` 统计；对 trial 做精确哈希 + 近重复抽样检查。
- **通过标准**：无未解释的重复。

### A12. MOABB `CrossSubject` 的正确用法
- [ ] 主实验使用 MOABB `CrossSubject`（或等价的 `LeaveOneGroupOut`），不要用 `WithinSession` 冒充跨被试。
- [ ] 确认 MOABB 只负责 split；**外挂的预处理没有在 `CrossSubject` 之外对全部被试 fit**。
- [ ] 如果使用 MOABB 的 `paradigm.get_data()`，确认没有在返回的全量数据上做全局 `fit_transform`。
- **怎么查**：阅读 MOABB evaluation 代码路径 + 检查自定义 preprocessing 的调用位置。
- **通过标准**：split 由 `CrossSubject` 产生；所有可训练预处理在 fold 内 fit。

---

## 3. B 类：预处理与特征（最容易被忽视的泄露）

### B1. 所有可训练预处理只在训练折 fit
- [ ] 清单：`StandardScaler / RobustScaler / MinMaxScaler / PCA / ICA / CSP / TS / SelectKBest / 通道选择 / 伪迹阈值 / 空间滤波`。
- [ ] 对每一项确认：`fit` / `fit_transform` 只在训练折调用；测试折只 `transform`。
- **怎么查**：逐个类搜索 `fit` 调用；在 fold 循环内打印 fit 数据的 subject 集合。
- **通过标准**：测试被试的 trial 从未进入任何 `fit`。

### B2. 禁止"先全量 fit_transform，再切分"
- [ ] 不允许先在完整 `X` 上做标准化/CSP/PCA，再交给 split。
- [ ] 正确顺序：`split -> fit(train) -> transform(train) -> transform(test)`。
- **怎么查**：`grep -n "fit_transform"`；检查其作用域是否在 split 之前、是否覆盖全部被试。
- **通过标准**：主路径中没有跨 fold 的 `fit_transform`。

### B3. 用 sklearn `Pipeline` 封装预处理 + 分类器
- [ ] 把 scaler、CSP、特征选择、分类器串成 `Pipeline`，在每折 `fit` 一次。
- [ ] 不要在 Pipeline 外对数据做任何依赖标签/统计量的变换。
- **怎么查**：检查 `Pipeline(...)` 构造；确认 `cross_val_score` / 自定义 fold 循环传入的是原始 epoch。
- **通过标准**：每折的 fit 只看到训练数据。

### B4. 固定滤波可以全局做，但参数必须记录
- [ ] 带通/陷波/重参考等固定参数滤波可以预先做（不依赖标签），但要写进 config。
- [ ] 如果滤波参数是数据驱动选的（如按被试选频带），必须在训练折内选。
- **怎么查**：config 中的 `filter` 字段；检查是否有 `for subject in all_subjects: choose_band(subject)` 然后用于测试。
- **通过标准**：滤波参数要么固定，要么在训练折内确定。

### B5. 归一化：主结果用训练被试统计量
- [ ] M1 主结果：per-channel 均值/标准差从训练被试计算，测试被试只 `transform`。
- [ ] 如果使用测试被试自身统计量，必须标注为 `transductive` / `online adaptation`，且只能使用**因果的过去数据**（不能使用未来 trial）。
- [ ] 不允许用整个测试 session 的均值/方差做归一化后再算离线准确率（这是非因果的）。
- **怎么查**：检查 `scaler.fit` 的数据索引；确认没有 `test_subject_all_trials` 进入 fit。
- **通过标准**：主结果不依赖测试被试未来数据。

### B6. 基线校正不能跨 trial/session
- [ ] 每个 epoch 用自己的 pre-cue 基线可以。
- [ ] 禁止用整个 session 或全数据均值做基线。
- **怎么查**：检查 `mne.Epochs` 的 `baseline` 参数或自定义 baseline 代码。
- **通过标准**：baseline 只使用该 trial 自身的时间窗。

### B7. 伪迹拒绝/ICA/坏道插值必须在训练折内 fit
- [ ] 阈值、ICA 分量选择、坏道检测规则不能看着测试集调。
- [ ] 如果使用自动伪迹检测，用固定阈值或只在训练折拟合阈值。
- **怎么查**：检查 `ICA().fit`、`find_bad_channels`、`threshold=` 的调用位置。
- **通过标准**：测试折不参与任何伪迹模型的 fit。

### B8. 通道选择不能使用全量数据
- [ ] 不能按全量方差/互信息选通道后再切分。
- [ ] 如果要选通道：训练折内做，或直接使用固定 10-20 全通道（M1 推荐后者）。
- **怎么查**：`SelectKBest`、`mutual_info`、`variance` 排序的调用位置。
- **通过标准**：主结果使用固定通道集，或在训练折内选择。

### B9. CSP/TS 等监督特征只在训练折 fit
- [ ] CSP 使用标签，必须只在训练折 fit；测试折只 `transform`。
- [ ] 不要把全部被试的 epoch 拼起来 fit 一个 CSP。
- **怎么查**：检查 `CSP().fit(X_train, y_train)` 的 `X_train` subject 集合。
- **通过标准**：测试被试的标签从未进入 CSP。

### B10. 数据增强：先切分，后增强，只增强训练折
- [ ] 增强（加噪、时间平移、通道 dropout、mixup 等）只在训练折做。
- [ ] 同一 trial 的增强副本不得跨折；不要在 split 前把增强副本写回数据集。
- [ ] 不要用测试集统计量（均值/方差/频谱）来设计增强。
- **怎么查**：检查 augmentation 调用位置；确认增强后的样本仍带原 `trial_id` 且只在 train。
- **通过标准**：测试折的 trial 从未被增强，也没有增强副本进入训练。

### B11. 任何预处理都不得使用测试标签
- [ ] 检查是否有 `y_test` 进入 scaler/CSP/特征选择/阈值/类别权重。
- [ ] 半监督/伪标签如果使用测试数据，必须单独作为 `transductive` 臂，不与主结果混。
- **怎么查**：`grep -n "y_test"`；检查所有使用 `y_test` 的位置是否只用于最终评估。
- **通过标准**：`y_test` 只出现在评估函数中。

### B12. 离线/在线一致性（部署有效性，非严格泄露）
- [ ] 离线评估如果使用零相位滤波（非因果），要记录；IF-5 上线前用因果滤波复测。
- [ ] 在线接口的归一化/特征提取只能使用当前及过去数据。
- **怎么查**：对比 `M1` 预处理与 `IF-5` 预处理代码；记录差异。
- **通过标准**：报告里明确说明离线/在线差异；不把离线非因果结果当作在线性能。

---

## 4. C 类：训练、调参与模型选择

### C1. M1 第一版不调参
- [ ] 使用文献/MOABB 的固定超参（学习率、batch size、epochs、dropout、CSP 分量数等）。
- [ ] 在 config 中记录每个超参的来源。
- **怎么查**：config diff；确认没有 `GridSearchCV` / `RandomSearch` 在主 LOSO 外层运行。
- **通过标准**：主结果可在一组固定超参下复现。

### C2. 如果调参，只在训练折内部调
- [ ] 使用 nested LOSO：外层 9 折评估；内层在 8 个训练被试中再留一选超参。
- [ ] 测试折的 balanced accuracy 不参与任何超参选择。
- **怎么查**：打印每次超参搜索的 subject 集合；确认无测试被试。
- **通过标准**：`set(search_subjects) ∩ set(outer_test_subjects) == ∅`。

### C3. 早停只看内层验证集
- [ ] 早停指标来自训练折内的验证集，不来自测试折。
- [ ] 可以记录测试折逐 epoch 曲线用于事后分析，但**不得据此选 epoch**。
- **怎么查**：检查 `EarlyStopping(monitor=...)` 的数据来源；确认 `val_*` 不是 outer test。
- **通过标准**：测试折不参与停止决策。

### C4. 固定种子，所有 arm 共用，报告全部
- [ ] 预注册 seed 列表（如 0–9）；baseline 与等效 dropout 臂使用同一组 seed。
- [ ] 报告每个 seed × 每个 fold 的结果；不报 best seed、不剔除"不好看"的 seed。
- **怎么查**：结果表按 `(arm, subject, seed)` 分组计数；确认没有缺失组合。
- **通过标准**：结果表是完整的笛卡尔积（arm × subject × seed）。

### C5. 模型选择预先注册
- [ ] 在 M1 开跑前写死主模型（如 CSP+LDA 与 EEGNet-8,2 各一条），不能看完测试结果再换。
- [ ] 如果增加模型，作为探索性分析单独报告，不替换主结果。
- **怎么查**：M1 设计文档/周报中的预注册记录。
- **通过标准**：主模型列表在跑测试前已固定。

### C6. 类别权重/重采样/阈值只在训练折计算
- [ ] 类别权重、过采样/欠采样、决策阈值不能使用测试折标签。
- [ ] 如果做阈值校准，必须在训练折内的验证集做。
- **怎么查**：检查 `compute_class_weight`、`RandomOverSampler`、阈值搜索的输入。
- **通过标准**：测试标签不参与这些计算。

### C7. 预训练/外部数据无重叠
- [ ] 如果使用预训练模型或外部 EEG 数据，确认与 BNCI2014-001 无被试/试次重叠。
- [ ] 记录预训练数据来源、许可、预处理。
- **怎么查**：数据来源清单 + 被试 ID 对照。
- **通过标准**：无重叠；来源可追溯。

### C8. 半监督/伪标签/自训练单独成臂
- [ ] 如果使用测试被试的无标签数据进行自适应，明确标注 `transductive`，并与主 LOSO 结果分表。
- [ ] 伪标签不能用测试标签；自训练只能使用测试输入的预测，不能使用真值。
- **怎么查**：检查训练循环中 `test_loader` 是否被用于更新模型。
- **通过标准**：主结果不使用测试数据更新模型；transductive 臂单独报告。

### C9. 测试折不做特征标准化/样本筛选
- [ ] 不允许"先看测试集分布，再决定删哪些 trial/通道"。
- [ ] 不允许在测试折上做 per-batch 标准化后再评估（除非明确是 online 方法）。
- **怎么查**：检查测试循环中的 `normalize` / `drop` / `filter` 调用。
- **通过标准**：测试折只做固定的、训练时确定的变换。

### C10. 推理模式正确
- [ ] 评估时 `model.eval()`；BatchNorm 使用训练折统计量，不用测试 batch 更新。
- [ ] Dropout 在推理时关闭；如果"等效 dropout 臂"是结构性 mask，明确它与训练态 dropout 的区别。
- **怎么查**：检查 `model.train()` / `model.eval()` 切换；检查 BN `track_running_stats`。
- **通过标准**：推理不更新任何可训练统计量。

---

## 5. D 类：评价与报告

### D1. 主指标与不确定性
- [ ] 主指标：balanced accuracy；4 类 chance = 25%。
- [ ] 报告每被试一值、均值 ± 标准差、95% CI（bootstrap 或 t 分布，口径写死）。
- **怎么查**：结果表 + 统计脚本输出。
- **通过标准**：9 个被试的值都在；CI 可复现。

### D2. 不剔除"表现差"的被试/折
- [ ] 所有 9 个被试、所有预注册 fold 都报告。
- [ ] 如果必须排除（如数据损坏），预先定义规则并在结果中列出被排除项及原因。
- **怎么查**：结果表 subject 计数 = 9；排除清单为空或有书面理由。
- **通过标准**：没有"只报好被试"。

### D3. 不报 best seed / best fold
- [ ] 报告全部 seed 的分布，不用 max 或 best-case 作为主结果。
- [ ] 如果展示单条曲线，标明它是哪个 seed，并同时给分布。
- **怎么查**：结果表 seed 计数完整；图表注明聚合方式。
- **通过标准**：主结果是预注册聚合，不是挑选后的最优值。

### D4. 统计检验与原始预测
- [ ] A 导出每条 trial 的 `y_true / y_pred / y_score / subject / session / seed / arm`。
- [ ] B 做配对检验（如 Wilcoxon）、多重比较校正、效应量；A 不替 B 选统计口径。
- **怎么查**：`predictions.csv` 行数 = 9 被试 × 576 trial/被试 × seeds × arms；字段齐全。
- **通过标准**：B 可仅凭导出文件复现统计。

### D5. 结果表固定表头
- [ ] `arm/model/subject/session/seed/balanced_acc/latency_ms/memory_MB/note` + `ablation_target/mask_id/activation_map_path`（如适用）。
- [ ] CSV + JSON 各一份；字段类型/单位写进 data dictionary。
- **怎么查**：表头 diff；JSON schema 校验。
- **通过标准**：C/B 可直接读取，无需问 A 字段含义。

### D6. 运行日志完整
- [ ] 每次批跑保存：config 快照、split index、环境版本（Python/包/GPU）、数据哈希、stdout/stderr。
- [ ] 日志目录与 `run_id` 一一对应。
- **怎么查**：随机抽一个结果行，按 `run_id` 找回全部日志。
- **通过标准**：三个月后可按日志复现。

### D7. 区分正式结果与示意演示
- [ ] 未跑完/未通过本清单的结果，demo/汇报中标注"示意演示"。
- [ ] 不用绝对词（"证明/首次/必然"），用"结果表明/与预期一致"。
- **怎么查**：检查报告/PPT 文案。
- **通过标准**：没有把 pilot 当正式结果。

### D8. 只和同协议数字比较
- [ ] 与文献比较时确认：被试内 vs 跨被试、2 类 vs 4 类、是否使用两个 session、预处理是否一致。
- [ ] 不拿 MOABB 被试内数字与 M1 的 LOSO 数字直接比。
- **怎么查**：文献对照表标注协议列。
- **通过标准**：比较表里协议一致或明确标注差异。

### D9. 两个 arm 的对照是干净的
- [ ] baseline 与等效 dropout 臂使用同一 split、同一 seed 列表、同一训练预算、同一评价代码。
- [ ] 唯一差异是 arm 特定机制（mask/dropout 结构）。
- **怎么查**：config diff；确认除 arm 参数外无其他差异。
- **通过标准**：性能差异可归因于 arm 机制，而不是数据/训练差异。

### D10. 报告泄露自查结果
- [ ] 在 M1 阶段报告/周报中附本清单的签核页。
- [ ] 如果发现过泄露并修复，记录"发现-修复-复测"过程。
- **怎么查**：报告附录。
- **通过标准**：读者能确认主结果通过了哪些检查。

---

## 6. E 类：代码级防呆与可复现

### E1. 硬断言函数
- [ ] 在 `common_utils` 实现：
  ```python
  def assert_subject_disjoint(train_idx, test_idx, metadata):
      train_s = set(metadata.loc[train_idx, "subject_id"])
      test_s = set(metadata.loc[test_idx, "subject_id"])
      assert train_s.isdisjoint(test_s), f"LEAKAGE: {train_s & test_s}"
  ```
- [ ] 每个 fold 的 DataLoader 构造前调用；失败即终止。
- **怎么查**：代码审查 + 单测。
- **通过标准**：无法绕过断言生成训练/测试集。

### E2. Toy 数据单测
- [ ] 构造 3 个被试 × 每个 10 trial 的 toy 数据，运行 split；断言无交集、每个被试只出现一次。
- [ ] 构造重叠窗口 toy 数据，断言同一 trial 的窗口同折。
- **怎么查**：`pytest tests/test_split.py -q`。
- **通过标准**：单测全绿。

### E3. 标签置换 canary
- [ ] 把所有标签随机打乱，重跑完整管线（包括预处理/调参/评估）；balanced accuracy 应回到 chance 附近。
- [ ] 如果仍显著高于 chance，排查泄露。
- **怎么查**：`pytest` 或独立脚本 + 结果日志。
- **通过标准**：置换后性能 ≈ 25%（4 类）。

### E4. 重复 trial canary
- [ ] 故意把一个测试 trial 复制进训练折，运行管线；观察性能是否异常升高。
- [ ] 这是对"数据索引/split 实现"的测试，不是对模型的测试；测试完删除注入。
- **怎么查**：独立分支/临时脚本 + 记录。
- **通过标准**：注入被 split 逻辑拒绝，或能被检测到。

### E5. Dry-run 打印 fold 摘要
- [ ] `--dry-run` 打印每折：`train_subjects`、`test_subjects`、样本数、类别分布、session 分布。
- [ ] 人工核对 9 折摘要后才允许正式跑。
- **怎么查**：把 dry-run 输出贴进 run log。
- **通过标准**：9 折摘要无异常。

### E6. `fit` 调用点审计
- [ ] `grep -n "\.fit(\|\.fit_transform(\|partial_fit("` 列出全部调用点；逐处标注作用域（fold 内/外）。
- [ ] 重点检查 `StandardScaler`、`CSP`、`PCA`、`ICA`、`SelectKBest`、`compute_class_weight`。
- **怎么查**：审计表 + code review 记录。
- **通过标准**：所有可训练变换都在 fold 内。

### E7. 预处理缓存带 fold 标识
- [ ] `02_数据/预处理缓存/` 中的缓存文件必须带 `subject/session/fold` 标识，或只缓存未监督的原始 epoch。
- [ ] 不允许缓存"全量 fit 后的标准化数据"给所有 fold 复用。
- **怎么查**：缓存文件命名 + 内容检查。
- **通过标准**：缓存不会把训练折统计量带进测试折。

### E8. CI / pre-commit 检查
- [ ] 把 A5/E1/E2 的单测放进 CI 或至少放进批跑入口；失败阻止运行。
- [ ] 在 `common_utils` 记录检查版本；清单更新时同步。
- **怎么查**：CI 配置 / pre-commit hook。
- **通过标准**：泄露检查不是"靠人记得"。

---

## 7. F 类：签核与变更管理

### F1. A 自查
- [ ] A 逐项填写本清单；不通过项写修复计划。
- [ ] 自查完成日期：____。

### F2. B 互审
- [ ] B 抽查至少 10 项（必须包含 A2/A4/A5/B1/B9/C2/C4）。
- [ ] B 签字/日期：____。

### F3. 证据留档
- [ ] 每项"通过"附证据路径：代码行、日志、命令输出、split 文件。
- [ ] 不写"应该没问题"；写可验证的证据。

### F4. 归档与周报
- [ ] 本清单归档到 `06_会议与笔记/`，并在周报中引用签核结果。
- [ ] 相关 run_id 与结果表一起归档。

### F5. 变更后重跑
- [ ] 任何改动（预处理、split、模型、增强、评价代码）后，重新过一遍受影响类别。
- [ ] 变更记录写入 `run_log`；旧结果不覆盖，保留版本。

---

## 8. 红黄绿判定

| 状态 | 条件 | 动作 |
|---|---|---|
| 🟢 绿 | A/B/C/D/E 全部通过（或 N/A 有书面理由） | 允许跑主实验并报告 |
| 🟡 黄 | 仅 D/E 的文档类项缺失，A/B/C 全通过 | 允许跑 pilot；不得作为正式结果报告 |
| 🔴 红 | A/B/C 任一 ❌ | **不得开跑主实验**；先修复并复测 |

> **M1 的底线**：A2、A4、A5、B1、B9、C2、C4 是红线中的红线。任何一条不通过，主结果都不可信。

---

## 9. 附录 1：泄露类型 → M1 检查点

| 泄露类型 | Brookshire 是否覆盖 | M1 检查点 | 严重度 |
|---|---|---|---|
| 同一被试跨 train/test（segment-based split） | ✅ 核心 | A2–A6 | 🔴 致命 |
| 测试被试的另一个 session 进入训练 | 部分（subject-level 覆盖） | A4、A8 | 🔴 致命 |
| 重叠窗口/同一 trial 的副本跨分区 | ❌ | A10、B10 | 🔴 致命 |
| 全局标准化/归一化使用测试数据 | ❌ | B1、B5 | 🔴 致命 |
| 全局 CSP/PCA/ICA/特征选择 | ❌ | B1、B7、B9 | 🔴 致命 |
| 用测试折调超参/早停/选 epoch | ❌ | C2、C3、C5 | 🔴 致命 |
| 用测试标签做类别权重/阈值/伪标签 | ❌ | B11、C6、C8 | 🔴 致命 |
| 数据增强在 split 前做/跨折复制 | ❌ | B10 | 🟠 高 |
| 通道级独立样本（channel-as-sample） | ✅ 提及 | A2、B8 | 🟠 高 |
| 只报 best seed/best fold | ❌ | C4、D3 | 🟠 高 |
| 文献比较协议不一致 | ❌ | D8 | 🟡 中 |
| 离线非因果滤波 vs 在线因果滤波 | ❌ | B12 | 🟡 中 |

## 10. 附录 2：开跑前 10 分钟快查（一页版）

- [ ] 1. 数据哈希、元数据表、9 被试 × 2 session；每被试 576 trial 核对完毕。
- [ ] 2. 主 split 是 LOSO；9 折；每折 1 测试被试，训练侧 8 人。
- [ ] 3. 测试被试的两个 session 全在测试侧；训练侧无该被试任何 trial。
- [ ] 4. `assert_subject_disjoint` 已在每个 fold 调用并通过。
- [ ] 5. 所有 scaler/CSP/PCA/ICA/特征选择只在训练折 `fit`。
- [ ] 6. 没有在 split 前做全局 `fit_transform`；没有缓存全量 fit 后的数据。
- [ ] 7. 没有用测试折调参/早停/选 epoch；seed 列表固定且所有 arm 一致。
- [ ] 8. 数据增强只在训练折，同一 trial 的窗口/副本不跨折。
- [ ] 9. 主指标按被试报告；9 个被试都在；不报 best seed/fold。
- [ ] 10. 原始 per-trial 预测已导出；结果表表头符合 IF-2；run log 完整。

## 11. 附录 3：证据摘录与引用

### 11.1 论文关键句
- "Data leakage, which arises when information about the test set is present in the training set, results in a positively-biased estimate of the model's performance."（P2）
- "To avoid this kind of data leakage, all segments from a given subject must be assigned to only a single partition of the data (i.e., train or validation or test)."（P2）
- "When training a translational DNN classifier, the model must be tested with subjects who were not included in the training set."（P7）
- "The same principles of data leakage will apply to task-based EEG; providing a classifier with person-specific information enables it to artificially inflate performance."（P7–P8）
- "Only 17 (27.0%) unambiguously avoided this type of data leakage."（P6）

### 11.2 论文自述 caveats
- 泄露幅度随实验细节变化；被试越少，偏置可能越强。
- 模型架构会影响泄露程度。
- within-subject 事件任务的下降幅度小于 between-subject 疾病分类。
- 文献回顾是 non-exhaustive；编码依赖论文描述。

### 11.3 本文不覆盖、但 M1 必须查的
- 预处理 fit 作用域（B1–B9）
- 超参/早停/模型选择（C1–C5）
- 数据增强（B10）
- 测试时归一化/自适应（B5、C8）
- 统计与报告（D1–D10）
- 代码级防呆（E1–E8）

### 11.4 关联文件
- 论文笔记：[[Data_leakage_in_deep_learning_studies_of_translational_EEG|Brookshire 2024]]
- 文献索引：[[README_文献索引]]
- 工作区导航：[[README_工作区导航]]
- 任务书：`20260925_任务书_学生A.docx`
