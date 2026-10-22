"""Aggregate M2 result files and produce the draft's comparison plots.

This script only computes descriptive statistics and a preliminary paired
Wilcoxon sanity check; B owns the formal statistical protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common_utils.paths import RESULTS_FIGURE_DIR, RESULTS_TABLE_DIR, ensure_dir

ARM_ORDER = [
    "dense_baseline",
    "equiv_dropout",
    "er_random",
    "deg_rewired",
    "bio_sparse",
]
COMPARISONS = [
    ("bio_sparse", "deg_rewired"),
    ("bio_sparse", "er_random"),
    ("bio_sparse", "equiv_dropout"),
    ("bio_sparse", "dense_baseline"),
]


def _load_results(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no result files matched {pattern!r}")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    # Keep the latest row if the same arm/subject/session/seed was rerun.
    df = df.drop_duplicates(
        subset=["arm", "subject", "session", "seed"], keep="last"
    )
    return df


def _paired_table(all_df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    subset = all_df[all_df["arm"].isin([left, right])]
    pivot = subset.pivot_table(
        index=["subject", "seed"], columns="arm", values="balanced_acc"
    )
    if left not in pivot.columns or right not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=[left, right]).copy()
    pivot["diff"] = pivot[left] - pivot[right]
    return pivot.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate M2 results")
    parser.add_argument(
        "--pattern",
        default=str(RESULTS_TABLE_DIR / "results_m2_*.csv"),
    )
    parser.add_argument("--out-dir", default=str(RESULTS_FIGURE_DIR / "m2_analysis"))
    parser.add_argument("--delta-acc", type=float, default=0.025)
    args = parser.parse_args(argv)

    out_dir = ensure_dir(args.out_dir)
    df = _load_results(args.pattern)
    all_df = df[df["session"].astype(str) == "all"].copy()
    if all_df.empty:
        raise SystemExit("no session='all' rows in the M2 results")
    all_df["arm"] = pd.Categorical(all_df["arm"], categories=ARM_ORDER, ordered=True)

    summary = (
        all_df.groupby("arm", observed=True)["balanced_acc"]
        .agg(["mean", "std", "count"])
        .reindex(ARM_ORDER)
        .dropna(how="all")
    )
    summary.to_csv(out_dir / "summary_by_arm.csv", encoding="utf-8")

    per_subject = (
        all_df.groupby(["arm", "subject"], observed=True)["balanced_acc"]
        .mean()
        .unstack("arm")
        .reindex(columns=ARM_ORDER)
    )
    per_subject.to_csv(out_dir / "per_subject_balanced_acc.csv", encoding="utf-8")

    comparison_rows = []
    comparison_details = {}
    for left, right in COMPARISONS:
        paired = _paired_table(all_df, left, right)
        if paired.empty:
            continue
        per_subject_diff = paired.groupby("subject")["diff"].mean()
        try:
            test = wilcoxon(per_subject_diff.values)
            statistic, pvalue = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, pvalue = float("nan"), float("nan")
        row = {
            "comparison": f"{left} - {right}",
            "left": left,
            "right": right,
            "n_subjects": int(len(per_subject_diff)),
            "mean_diff": float(per_subject_diff.mean()),
            "std_diff": float(per_subject_diff.std(ddof=1)),
            "min_diff": float(per_subject_diff.min()),
            "max_diff": float(per_subject_diff.max()),
            "delta_acc": float(args.delta_acc),
            "meets_delta_acc": bool(per_subject_diff.mean() >= args.delta_acc),
            "wilcoxon_stat": statistic,
            "wilcoxon_p": pvalue,
        }
        comparison_rows.append(row)
        comparison_details[f"{left} - {right}"] = paired.to_dict(orient="records")

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out_dir / "paired_differences.csv", index=False, encoding="utf-8")
    (out_dir / "analysis.json").write_text(
        json.dumps(
            {
                "n_files": len(glob.glob(args.pattern)),
                "n_rows": int(len(df)),
                "n_all_rows": int(len(all_df)),
                "arms": [arm for arm in ARM_ORDER if arm in set(all_df["arm"])],
                "seeds": sorted(int(seed) for seed in all_df["seed"].unique()),
                "subjects": sorted(str(subject) for subject in all_df["subject"].unique()),
                "delta_acc": float(args.delta_acc),
                "summary": summary.reset_index().to_dict(orient="records"),
                "comparisons": comparison_rows,
                "paired_details": comparison_details,
                "note": "preliminary descriptive statistics; B owns the formal statistical test",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Figure 1: per-arm balanced accuracy.
    present = [arm for arm in ARM_ORDER if arm in summary.index]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(present))
    values = summary.loc[present, "mean"].to_numpy()
    errors = summary.loc[present, "std"].fillna(0).to_numpy()
    colors = ["#7f8c8d" if arm != "bio_sparse" else "#c0392b" for arm in present]
    ax.bar(positions, values, yerr=errors, capsize=3, color=colors, alpha=0.85)
    ax.axhline(0.25, color="#2c3e50", ls="--", lw=1, label="chance (4 classes)")
    ax.set_xticks(positions)
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel("balanced accuracy (mean ± sd, subject × seed)")
    ax.set_title("M2: fly-topology prior vs null masks")
    ax.set_ylim(0, max(0.65, float(np.nanmax(values + errors)) + 0.05))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "m2_arms_balanced_acc.png", dpi=200)
    plt.close(fig)

    # Figure 2: paired differences against the δ_acc threshold.
    if not comparisons.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        positions = np.arange(len(comparisons))
        means = comparisons["mean_diff"].to_numpy()
        stds = comparisons["std_diff"].fillna(0).to_numpy()
        ax.bar(positions, means, yerr=stds, capsize=3, color="#2980b9", alpha=0.85)
        ax.axhline(0, color="#2c3e50", lw=1)
        ax.axhline(
            args.delta_acc,
            color="#c0392b",
            ls="--",
            lw=1.2,
            label=f"δ_acc = {args.delta_acc:.3f}",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(comparisons["comparison"], rotation=20, ha="right")
        ax.set_ylabel("paired difference in balanced accuracy")
        ax.set_title("M2: bio_sparse minus each null")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "m2_paired_differences.png", dpi=200)
        plt.close(fig)

    # Figure 3: per-subject paired means.
    if not per_subject.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for arm in present:
            if arm in per_subject.columns:
                ax.plot(
                    per_subject.index.astype(str),
                    per_subject[arm].to_numpy(),
                    marker="o",
                    ms=3,
                    lw=1,
                    label=arm,
                )
        ax.set_ylabel("balanced accuracy (mean over seeds)")
        ax.set_xlabel("held-out subject")
        ax.set_title("M2: per-subject comparison")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "m2_per_subject.png", dpi=200)
        plt.close(fig)

    print(summary.round(4).to_string())
    if not comparisons.empty:
        print()
        print(comparisons.round(4).to_string(index=False))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
