"""Shared project paths.

The project root is two levels above this file:
``03_代码/common_utils/paths.py`` -> ``03_代码/common_utils`` -> ``03_代码`` -> project root.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "03_代码"

DATA_DIR = PROJECT_ROOT / "02_数据"
BNCI_RAW_DIR = DATA_DIR / "BNCI2014-001_原始"
CACHE_DIR = DATA_DIR / "预处理缓存"
DATA_CARDS_DIR = DATA_DIR / "数据卡data_cards"

FLY_DIR = DATA_DIR / "fly_connectome"
FLY_RAW_DIR = FLY_DIR / "raw"
FLY_PROCESSED_DIR = FLY_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "04_实验结果"
RESULTS_TABLE_DIR = RESULTS_DIR / "结果表"
RESULTS_LOG_DIR = RESULTS_DIR / "运行日志"
RESULTS_FIGURE_DIR = RESULTS_DIR / "图表"

REPORTS_DIR = PROJECT_ROOT / "05_报告与交付"
NOTES_DIR = PROJECT_ROOT / "06_会议与笔记"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (including parents) and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
