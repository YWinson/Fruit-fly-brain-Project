# 03_代码

项目代码目录。当前已搭建 **M1 评测管线 + common_utils**；M2–M5 目录保留为空壳，按里程碑逐步填充。

```
03_代码/
├── common_utils/           # 共用底座：路径、元数据、LOSO split、泄露硬断言、结果表、日志、配置
├── M1_评测管线/            # 当前实现：无结构基线 + 等效 dropout 臂的 LOSO 流程
├── M2_先验注入/            # 待 M2 设计草案评审后动工
├── M3_消融工具/            # 待 M1/M2 稳定后动工
├── M4_可塑性验证/          # 可在 M1 后并行启动
├── M5_移植验证/            # 待 M3 定位有效回路后动工
├── IF-5在线接口/            # batch=1 在线接口，随 M1–M4 带出
├── tests/                  # common_utils + M1 的单元测试/冒烟测试
├── requirements-m1.txt     # 核心依赖（numpy/pandas/scipy/sklearn/mne/pytest）
└── requirements-m1-moabb.txt  # 加上 moabb（真实 BNCI2014-001 需要）
```

## 环境说明

当前机器的 Anaconda Python 3.13 已安装 `numpy / pandas / scipy / scikit-learn / mne / PyYAML / joblib / pytest`，但 **未安装 `moabb`**。

- **先跑 synthetic**：不需要 MOABB，直接验证 split、Pipeline、结果表、日志。
- **跑真实 BNCI2014-001**：安装 MOABB。如果 Python 3.13 装不上，建一个 Python 3.11/3.12 环境：
  ```bash
  conda create -n flyeeg python=3.11 -y
  conda activate flyeeg
  pip install -r requirements-m1-moabb.txt
  ```
- **torch / EEGNet**：M1 当前不依赖 torch；M5 或后续神经网络臂再装。

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
python "M1_评测管线/run_m1.py" --config "M1_评测管线/configs/m1_bnci2014_001.yaml" \
    --dataset bnci2014_001 --arm baseline --seed 0 --subject 1
```

## 关键约定

- **被试是唯一分组变量**：主实验只允许 LOSO；任何按 trial/window 的随机切分只能作为单独命名的 sanity check。
- **所有可训练变换只在训练折 fit**：`StandardScaler`、`CSP`、`PCA`、`ICA`、特征选择全部封装在 sklearn `Pipeline` 里，每折 `fit` 一次。
- **失败即停**：`common_utils.leakage_checks` 抛 `LeakageError`，不写 warning 后继续。
- **结果表遵守 IF-2**：`arm/model/subject/session/seed/balanced_acc/latency_ms/memory_MB/note` + 消融字段；CSV + JSON；逐试次预测单独导出。
- **synthetic 结果必须标注"示意演示"**：它只验证管线，不代表方法效果。

## 相关文档

- [M1 评测管线 README](M1_评测管线/README.md)
- [M1 LOSO 泄露自查清单](../06_会议与笔记/M1_LOSO_泄露自查清单.md)
- [工作区导航](../README_工作区导航.md)
- [任务书](../20260925_任务书_学生A.docx)
