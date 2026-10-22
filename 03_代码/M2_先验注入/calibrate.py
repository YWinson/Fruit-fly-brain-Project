"""δ_topo calibration CLI (FFL / bi-fan vs degree-preserving nulls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np

from common_utils.paths import FLY_PROCESSED_DIR, RESULTS_FIGURE_DIR, ensure_dir
from M2_先验注入.clusters import build_granularity
from M2_先验注入.config import load_m2_config
from M2_先验注入.connectome import load_type_connectome
from M2_先验注入.motifs import calibrate_motifs, plot_motif_null, save_calibration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 δ_topo motif calibration")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("configs") / "m2_bnci2014_001.yaml"),
    )
    parser.add_argument("--granularity", choices=["fine", "medium", "coarse"], default="fine")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_FIGURE_DIR / "m2_motifs"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_m2_config(args.config)
    fly = config["fly"]
    n_samples = int(args.n_samples or fly.get("motif_null_samples", 1000))
    n_jobs = int(args.n_jobs if args.n_jobs is not None else fly.get("motif_n_jobs", -1))

    tc = load_type_connectome(FLY_PROCESSED_DIR)
    result = build_granularity(
        tc,
        args.granularity,
        superclasses=tuple(fly["superclasses"]),
        min_neurons=int(fly.get("min_neurons", 1)),
        seed=0,
    )
    adjacency = (result.W_syn_square > 0)
    np.fill_diagonal(adjacency, False)
    print(
        f"granularity={result.name} square_graph={adjacency.shape} "
        f"edges={int(adjacency.sum())} nonzero_nodes="
        f"{int(((adjacency.sum(0) > 0) | (adjacency.sum(1) > 0)).sum())}"
    )
    calibration = calibrate_motifs(
        adjacency,
        n_samples=n_samples,
        seed=0,
        method="edge_swap",
        n_jobs=n_jobs,
    )
    out_dir = ensure_dir(args.out_dir)
    summary_path = save_calibration(calibration, out_dir)
    plot_motif_null(calibration, out_dir / f"motif_null_{result.name}.png")
    print(
        json.dumps(
            {
                "n_valid": calibration["n_valid"],
                "ffl": calibration["ffl"],
                "bifan": calibration["bifan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
