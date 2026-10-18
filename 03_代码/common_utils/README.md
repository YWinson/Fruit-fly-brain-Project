# common_utils

M1–M5 共用的工程底座。当前已实现 M1 需要的部分：

| 文件 | 作用 | 对应清单项 |
|---|---|---|
| `paths.py` | 项目路径常量（`04_实验结果`、`运行日志`、数据缓存等） | — |
| `metadata.py` | 统一 `epoch_id/subject_id/session_id/run_id/trial_id/label` 表；检查唯一性 | A1、A6 |
| `splits.py` | 按被试分组的 LOSO / nested LOSO；每折自动跑泄露检查 | A2–A10 |
| `leakage_checks.py` | 硬断言：被试不相交、测试被试全部 session 在测试侧、trial 不跨 split、无重复 epoch、标签置换 canary | A2/A4/A5/A10、E1–E4 |
| `results.py` | IF-2 结果表头、逐试次预测导出、聚合 | D4–D6 |
| `logging_utils.py` | 每个 run 的日志目录、config 快照、环境版本、文件哈希 | D6 |
| `config.py` | YAML 加载、深度合并、校验、config hash | — |
| `seeding.py` | 固定种子、由 seed + fold 派生 mask seed | C4 |

## 约定

- `Fold.train_idx` / `Fold.test_idx` 是**位置索引**，不是 pandas 标签。
- 主实验只允许按 `subject_id` 分组；任何按 trial/window 随机切分都必须命名为单独的 sanity check。
- 所有可训练变换（`StandardScaler`、`CSP`、`PCA`、`ICA`、特征选择）只能在一个 fold 的训练集上 `fit`。`M1_评测管线` 用 sklearn `Pipeline` 保证这一点。
- 失败即抛 `LeakageError`，不写 warning 后继续。

## 自测

```bash
cd "03_代码"
python -m pytest tests -q
```
