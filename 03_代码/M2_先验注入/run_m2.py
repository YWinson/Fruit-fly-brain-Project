"""M2 prior-injection CLI.

Example
-------
Dry run (build masks and print fold summaries, no training)::

    python "03_代码/M2_先验注入/run_m2.py" --dry-run

One fine-granularity fold::

    python "03_代码/M2_先验注入/run_m2.py" \
        --granularity fine --arm bio_sparse --seed 0 --subject 1 --max-folds 1

Full run::

    python "03_代码/M2_先验注入/run_m2.py" --granularity fine
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import mne
import numpy as np

mne.set_log_level("WARNING")

from common_utils.logging_utils import (
    RunLogger,
    collect_env,
    configure_utf8_stdio,
    make_run_id,
)
from common_utils.paths import (
    FLY_PROCESSED_DIR,
    FLY_RAW_DIR,
    ensure_dir,
    resolve_project_path,
)
from common_utils.results import write_predictions, write_results
from common_utils.splits import make_loso_splits, save_folds
from M1_评测管线.data import load_dataset
from M2_先验注入.arms import ARM_NAMES
from M2_先验注入.clusters import build_granularity, save_granularity
from M2_先验注入.config import ALLOWED_GRANULARITIES, load_m2_config
from M2_先验注入.connectome import build_from_defaults, load_type_connectome
from M2_先验注入.evaluate import fold_report, run_all, run_canary, summary_table
from M2_先验注入.masks import (
    bipartite_degree_preserving_rewire,
    build_bio_mask,
    er_random_mask,
)
from M2_先验注入.motifs import calibrate_motifs, plot_motif_null, save_calibration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = Path(__file__).with_name("configs") / "m2_bnci2014_001.yaml"
    parser = argparse.ArgumentParser(description="M2 fly-topology prior injection")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--dataset", choices=["bnci2014_001", "synthetic"], default=None)
    parser.add_argument(
        "--granularity", choices=list(ALLOWED_GRANULARITIES), default=None
    )
    parser.add_argument(
        "--arm", nargs="+", choices=list(ARM_NAMES) + ["all"], default=None
    )
    parser.add_argument("--seed", nargs="+", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--calibrate-motifs", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--list-arms", action="store_true")
    return parser.parse_args(argv)


def _select_arms(args, config: dict) -> list[str]:
    if not args.arm or "all" in args.arm:
        return list(config["arms"])
    return list(dict.fromkeys(args.arm))


def _select_seeds(args, config: dict) -> list[int]:
    if not args.seed or "all" in args.seed:
        return [int(seed) for seed in config["evaluation"]["seeds"]]
    try:
        return [int(seed) for seed in args.seed]
    except ValueError as exc:
        raise ValueError(f"--seed values must be integers or 'all', got {args.seed}") from exc


def _load_connectome(config: dict, logger: RunLogger):
    processed_dir = resolve_project_path(config["fly"]["processed_dir"])
    raw_dir = resolve_project_path(config["fly"]["raw_dir"])
    ensure_dir(processed_dir)
    try:
        tc = load_type_connectome(processed_dir)
        logger.log(f"loaded type connectome from {processed_dir}")
    except FileNotFoundError:
        logger.log(f"type connectome missing; building from {raw_dir}")
        tc = build_from_defaults(raw_dir=raw_dir, out_dir=processed_dir)
    logger.save_json("connectome_summary.json", tc.summary)
    return tc, processed_dir


def _build_granularity(config: dict, tc, logger: RunLogger):
    fly = config["fly"]
    result = build_granularity(
        tc,
        fly["granularity"],
        superclasses=tuple(fly["superclasses"]),
        min_neurons=int(fly.get("min_neurons", 1)),
        seed=0,
    )
    save_granularity(result, logger.log_dir / "granularity")
    logger.save_json("granularity_stats.json", result.stats)
    logger.log(
        f"granularity={result.name} k_out={result.k_out} k_in={result.k_in} "
        f"types={result.stats['n_types_selected']} "
        f"positive_expanded={result.stats['n_nonzero_expanded_pairs']}"
    )
    return result


def _build_bio_summary(result, config: dict, logger: RunLogger):
    fly = config["fly"]
    mask, sign, info = build_bio_mask(
        result,
        density=float(fly["mask_density"]),
        selection=str(fly["mask_selection"]),
    )
    mask_dir = ensure_dir(logger.log_dir / "masks")
    mask_path = mask_dir / "bio_mask.npy"
    sign_path = mask_dir / "bio_sign.npy"
    np.save(mask_path, mask)
    np.save(sign_path, sign)
    info["mask_path"] = str(mask_path)
    info["sign_path"] = str(sign_path)
    logger.save_json("bio_mask_summary.json", info)
    logger.log(
        f"bio mask edges={info['n_edges']} density={info['density']:.4f} "
        f"granularity={info['granularity']} selection={info['selection']}"
    )
    return mask, sign, info


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = parse_args(argv)
    if args.list_arms:
        print("available arms:", ", ".join(ARM_NAMES))
        return 0

    config = load_m2_config(args.config)
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.granularity:
        config["fly"]["granularity"] = args.granularity
    arms = _select_arms(args, config)
    seeds = _select_seeds(args, config)
    results_dir = resolve_project_path(args.results_dir or config["output"]["results_dir"])
    log_dir = resolve_project_path(args.log_dir or config["output"]["log_dir"])

    run_id = make_run_id("m2")
    logger = RunLogger(run_id, log_dir)
    try:
        logger.log(f"run_id={run_id}")
        logger.save_config(config)
        logger.save_json("env.json", collect_env())

        dataset = load_dataset(config["dataset"]["name"], config)
        logger.log(
            f"dataset={config['dataset']['name']} shape={tuple(dataset.X.shape)} "
            f"subjects={dataset.metadata['subject_id'].nunique()}"
        )
        tc, _ = _load_connectome(config, logger)
        result = _build_granularity(config, tc, logger)
        bio_mask, bio_sign, bio_info = _build_bio_summary(result, config, logger)

        folds = make_loso_splits(dataset.metadata)
        save_folds(folds, logger.log_dir / "splits.json")
        logger.log(f"folds={len(folds)} split_file={logger.log_dir / 'splits.json'}")

        if args.calibrate_motifs:
            adjacency = (result.W_syn_square > 0)
            np.fill_diagonal(adjacency, False)
            logger.log(
                f"calibrating motifs on {result.name} square graph "
                f"({int(adjacency.sum())} edges, n={config['fly']['motif_null_samples']}) ..."
            )
            calibration = calibrate_motifs(
                adjacency,
                n_samples=int(config["fly"]["motif_null_samples"]),
                seed=0,
                method="edge_swap",
                n_jobs=int(config["fly"].get("motif_n_jobs", -1)),
            )
            save_calibration(calibration, logger.log_dir / "motifs")
            plot_motif_null(calibration, logger.log_dir / "motifs" / "motif_null.png")
            logger.log(
                "motif δ_topo: FFL q99="
                f"{calibration['ffl']['null_q99_delta_topo']:.1f}, "
                f"observed={calibration['ffl']['observed']}, "
                f"Z={calibration['ffl']['z']:.2f}; "
                f"bi-fan q99={calibration['bifan']['null_q99_delta_topo']:.1f}, "
                f"observed={calibration['bifan']['observed']}"
            )

        if args.dry_run:
            print(fold_report(dataset, max_folds=args.max_folds).to_string(index=False))
            print()
            print(json.dumps(_public_mask_info(bio_info), ensure_ascii=False, indent=2))
            return 0

        if args.check_only:
            _run_mask_checks(bio_mask, result, logger)
            logger.log("all split and mask checks passed")
            return 0

        results, predictions = run_all(
            dataset=dataset,
            config=config,
            arms=arms,
            seeds=seeds,
            granularity_result=result,
            subject=args.subject,
            max_folds=args.max_folds,
            mask_dir=logger.log_dir / "masks",
            logger=logger,
        )
        results_csv, results_json = write_results(results, results_dir, run_id)
        predictions_csv = write_predictions(predictions, results_dir, run_id)
        summary = summary_table(results)
        print()
        print(summary.to_string(index=False))
        logger.save_json("summary.json", summary.to_dict(orient="records"))
        logger.log(f"results_csv={results_csv}")
        logger.log(f"results_json={results_json}")
        logger.log(f"predictions_csv={predictions_csv}")

        if args.canary:
            failed = False
            for arm in arms:
                for seed in seeds:
                    canary = run_canary(
                        dataset=dataset,
                        arm=arm,
                        seed=seed,
                        config=config,
                        granularity_result=result,
                        subject=args.subject,
                        max_folds=args.max_folds,
                    )
                    logger.log(f"canary arm={arm} seed={seed}: {canary.message}")
                    failed = failed or (not canary.passed)
            if failed:
                logger.log("canary FAILED — inspect leakage before reporting")
                return 3

        logger.log("done")
        return 0
    except Exception as exc:
        logger.log(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        logger.save_text("traceback.txt", traceback.format_exc())
        return 1
    finally:
        logger.close()


def _public_mask_info(info: dict) -> dict:
    return {
        key: value
        for key, value in info.items()
        if key not in {"out_degree", "in_degree", "out_degree_sequence", "in_degree_sequence"}
    }


def _run_mask_checks(bio_mask, result, logger: RunLogger) -> None:
    n_edges = int(bio_mask.sum())
    logger.log(f"bio mask: edges={n_edges} density={bio_mask.mean():.4f}")
    if result.name != "coarse":
        rewired, info = bipartite_degree_preserving_rewire(bio_mask, seed=0)
        logger.log(
            "deg_rewired: edges="
            f"{int(rewired.sum())} hamming={info['hamming']} "
            f"(E={n_edges})"
        )
        if info["hamming"] < n_edges:
            raise RuntimeError("degree-preserving null Hamming distance is too small")
    else:
        logger.log("deg_rewired: N/A for coarse E/I block mask")
    random_mask = er_random_mask(64, 32, n_edges=n_edges, seed=0)
    logger.log(f"er_random: edges={int(random_mask.sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
