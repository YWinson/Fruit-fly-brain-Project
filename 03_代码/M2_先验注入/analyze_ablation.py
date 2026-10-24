"""Summarise the mask-density / mapping-selection ablation."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd
from scipy.stats import wilcoxon

from common_utils.paths import RESULTS_TABLE_DIR

CONDITIONS = [
    ("density_05", "m2_ablation/density_05/results_m2_*.csv"),
    ("density_10", "results_m2_20261022_124200_0468.csv"),
    ("density_20", "m2_ablation/density_20/results_m2_*.csv"),
    ("selection_global_10", "m2_ablation/selection_global_10/results_m2_*.csv"),
]
COMPARISONS = [
    ("bio_sparse", "deg_rewired"),
    ("bio_sparse", "er_random"),
    ("bio_sparse", "equiv_dropout"),
    ("bio_sparse", "dense_baseline"),
]


def _load(pattern: str) -> pd.DataFrame:
    path = Path(pattern)
    files = [pattern] if path.is_file() else sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern!r}")
    df = pd.concat([pd.read_csv(item) for item in files], ignore_index=True)
    return df.drop_duplicates(
        subset=["arm", "subject", "session", "seed"], keep="last"
    )


def _paired(all_df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    subset = all_df[all_df["arm"].isin([left, right])]
    pivot = subset.pivot_table(
        index=["subject", "seed"], columns="arm", values="balanced_acc"
    )
    if left not in pivot.columns or right not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=[left, right])
    pivot["diff"] = pivot[left] - pivot[right]
    return pivot.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M2 mask density/selection ablation")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="use one matched seed across all conditions (default: 0)",
    )
    args = parser.parse_args(argv)

    out_dir = RESULTS_TABLE_DIR / "m2_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_rows = []
    pair_rows = []
    details = {}
    for condition, pattern in CONDITIONS:
        full_pattern = str(RESULTS_TABLE_DIR / pattern)
        matches = (
            [full_pattern]
            if Path(full_pattern).is_file()
            else glob.glob(full_pattern)
        )
        if not matches:
            print(f"skip {condition}: no result files yet")
            continue
        df = _load(full_pattern)
        all_df = df[
            (df["session"].astype(str) == "all") & (df["seed"] == args.seed)
        ].copy()
        if all_df.empty:
            print(f"skip {condition}: no session=all rows for seed={args.seed}")
            continue

        summary = (
            all_df.groupby("arm")["balanced_acc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for row in summary.to_dict(orient="records"):
            row["condition"] = condition
            arm_rows.append(row)

        condition_details = {}
        for left, right in COMPARISONS:
            paired = _paired(all_df, left, right)
            if paired.empty:
                continue
            per_subject = paired.groupby("subject")["diff"].mean()
            try:
                test = wilcoxon(per_subject.values)
                pvalue = float(test.pvalue)
            except ValueError:
                pvalue = float("nan")
            row = {
                "condition": condition,
                "comparison": f"{left} - {right}",
                "n_subjects": int(len(per_subject)),
                "mean_diff": float(per_subject.mean()),
                "std_diff": float(per_subject.std(ddof=1)),
                "min_diff": float(per_subject.min()),
                "max_diff": float(per_subject.max()),
                "wilcoxon_p": pvalue,
            }
            pair_rows.append(row)
            condition_details[row["comparison"]] = paired.to_dict(orient="records")
        details[condition] = condition_details

    arms = pd.DataFrame(arm_rows)
    pairs = pd.DataFrame(pair_rows)
    arms.to_csv(out_dir / "ablation_arms.csv", index=False, encoding="utf-8")
    pairs.to_csv(out_dir / "ablation_pairs.csv", index=False, encoding="utf-8")
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(
            {
                "arms": arm_rows,
                "pairs": pair_rows,
                "paired_details": details,
                "note": "descriptive; B owns the formal statistical protocol",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== per condition / arm ===")
    print(arms.round(4).to_string(index=False))
    print()
    print("=== paired bio - null ===")
    print(pairs.round(4).to_string(index=False))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
