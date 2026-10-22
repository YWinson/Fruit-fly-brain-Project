# 03_代码

项目代码目录。当前已搭建 **M1 评测管线 + M2 先验注入臂 + common_utils**；M3–M5 目录保留为空壳，按里程碑逐步填充。

```
03_代码/
├── common_utils/           # 共用底座：路径、元数据、LOSO split、泄露硬断言、结果表、日志、配置
├── M1_评测管线/            # 无结构基线 + 等效稀疏臂的 LOSO 流程
├── M2_先验注入/            # 已实现：FlyWire 类型级先验 + 5 条对照臂 + masked MLP + δ_topo
├── M3_消融工具/            # 待 M2 结果稳定后动工
├── M4_可塑性验证/          # 可在 M1 后并行启动
├── M5_移植验证/            # 待 M3 定位有效回路后动工
├── IF-5在线接口/            # batch=1 在线接口，随 M1–M4 带出
├── tests/                  # common_utils + M1 + M2 的单元测试/冒烟测试
├── requirements-m1.txt     # 核心依赖（numpy/pandas/scipy/sklearn/mne/pytest）
└── requirements-m1-moabb.txt  # 加上 moabb（真实 BNCI2014-001 需要）
```

## 环境说明

**当前跑 M1/M2 用 `anaconda3/envs/eeg`**（Python 3.11.15）：已装 `moabb 1.5.0`、`mne 1.12.1`、`scikit-learn 1.8.0`、`torch 2.5.0`、`pandas 3.0.2`、`networkx 3.6.1`。

- base 的 Python 3.13 只有 M1 的基础依赖（无 moabb、无 torch），不要用它跑 M2。
- **先跑 synthetic**：不需要 BNCI 下载，验证 split、Pipeline、结果表、日志。
- **真实 BNCI2014-001**：数据已在 `02_数据/BNCI2014-001_原始/`（D 盘）；用 `envs/eeg/python.exe`。
- **FlyWire 数据**：`02_数据/fly_connectome/raw/`（D 盘，约 130 MB）；`.gitignore` 已排除。

## 快速开始

```bash
cd "03_代码"

# 1) 单元测试 + 冒烟测试
python -m pytest tests -q

# 2) 只看 LOSO 切分，不训练
python "M1_评测管线/run_m1.py" --config "M1_评测管线/configs/m1_synthetic.yaml" \
    --dataset synthetic --dry-run

# 3) 一个 fold 的 synthetic 冒烟运行（结果不是科学结果）
python "M1_评测管线/run_m1.py" --config "M1_评测管线/configs/m1_synthetic.yaml" \
    --dataset synthetic --arm baseline equiv_dropout --seed 0 --max-folds 1

# 4) 真实数据（需 MOABB）
"C:/Users/Winson/anaconda3/envs/eeg/python.exe" "M1_评测管线/run_m1.py" \
    --config "M1_评测管线/configs/m1_bnci2014_001.yaml" \
    --dataset bnci2014_001 --arm baseline --seed 0 --subject 1

# 5) M2：构建 FlyWire 类型级矩阵（只需一次）
"C:/Users/Winson/anaconda3/envs/eeg/python.exe" "M2_先验注入/connectome.py" --force

# 6) M2：单折 5 arms 冒烟
"C:/Users/Winson/anaconda3/envs/eeg/python.exe" "M2_先验注入/run_m2.py" \
    --config "M2_先验注入/configs/m2_bnci2014_001.yaml" \
    --arm bio_sparse deg_rewired er_random equiv_dropout dense_baseline \
    --seed 0 --subject 1 --max-folds 1

# 7) M2：全量 3 seeds × 9 folds × 5 arms
"C:/Users/Winson/anaconda3/envs/eeg/python.exe" "M2_先验注入/run_m2.py" \
    --config "M2_先验注入/configs/m2_bnci2014_001.yaml" --seed 0 1 2

# 8) M2 with EEGNet Stage 1（更强特征提取器；每折约 2.5–3 分钟）
"C:/Users/Winson/anaconda3/envs/eeg/python.exe" "M2_先验注入/run_m2.py" \
    --config "M2_先验注入/configs/m2_bnci2014_001_eegnet.yaml" --seed 0
```

## 关键约定

- **被试是唯一分组变量**：主实验只允许 LOSO；任何按 trial/window 的随机切分只能作为单独命名的 sanity check。
- **所有可训练变换只在训练折 fit**：`StandardScaler`、`CSP`、`PCA`、`ICA`、特征选择全部封装在 sklearn `Pipeline` 里，每折 `fit` 一次。
- **失败即停**：`common_utils.leakage_checks` 抛 `LeakageError`，不写 warning 后继续。
- **结果表遵守 IF-2**：`arm/model/subject/session/seed/balanced_acc/latency_ms/memory_MB/note` + 消融字段；CSV + JSON；逐试次预测单独导出。
- **synthetic 结果必须标注"示意演示"**：它只验证管线，不代表方法效果。

## 相关文档

- [M1 评测管线 README](M1_评测管线/README.md)
- [M2 先验注入 README](M2_先验注入/README.md)
- [M1 LOSO 泄露自查清单](../06_会议与笔记/M1_LOSO_泄露自查清单.md)
- [工作区导航](../README_工作区导航.md)
- [任务书](../20260925_任务书_学生A.docx)
