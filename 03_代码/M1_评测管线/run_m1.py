"""M1 command-line entry point.

Examples
--------
Smoke test on synthetic data (no MOABB needed)::

    python "03_代码/M1_评测管线/run_m1.py" --config "03_代码/M1_评测管线/configs/m1_synthetic.yaml" \
        --dataset synthetic --arm baseline --seed 0 --dry-run

Run one real LOSO fold on BNCI2014-001::

    python "03_代码/M1_评测管线/run_m1.py" \
        --config "03_代码/M1_评测管线/configs/m1_bnci2014_001.yaml" \
        --arm baseline --seed 0 --subject 1

The CLI accepts the task book's IF-2 parameters: ``--arm``, ``--seed``,
``--subject``, and ``--dataset``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Make both ``common_utils`` and the ``M1_评测管线`` package importable when
# this file is executed as a script (rather than via ``python -m``).
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common_utils.config import config_hash, load_config, validate_config
from common_utils.leakage_checks import LeakageError
from common_utils.logging_utils import (
    RunLogger,
    collect_env,
    configure_utf8_stdio,
    make_run_id,
)
from common_utils.paths import resolve_project_path
from common_utils.results import write_predictions, write_results
from common_utils.splits import make_loso_splits, save_folds
from M1_评测管线.arms import ARM_NAMES
from M1_评测管线.data import dataset_fingerprint, dataset_summary, load_dataset
from M1_评测管线.evaluate import (
    dry_run_report,
    run_all,
    run_canary,
    summary_table,
)

import mne

mne.set_log_level("WARNING")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = Path(__file__).with_name("configs") / "m1_synthetic.yaml"
    parser = argparse.ArgumentParser(
        description="M1 LOSO evaluation: baseline vs equivalent-dropout arm"
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="YAML config (default: configs/m1_synthetic.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "bnci2014_001"],
        default=None,
        help="override dataset.name from the config",
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        choices=list(ARM_NAMES) + ["all"],
        default=None,
        help="arm(s) to run; default: config arms",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        default=None,
        help="seed(s) to run, or 'all'; default: config seeds",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="run only the fold whose test subject matches this id (debug)",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="limit the number of folds (debug/smoke test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print fold summaries and exit without fitting",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run all split/leakage checks and exit without fitting",
    )
    parser.add_argument(
        "--canary",
        action="store_true",
        help="run the label-permutation canary after the main evaluation",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="override output.results_dir",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="override output.log_dir",
    )
    parser.add_argument(
        "--list-arms",
        action="store_true",
        help="print available arms and exit",
    )
    return parser.parse_args(argv)


def _select_arms(args: argparse.Namespace, config: dict) -> list[str]:
    if not args.arm or "all" in args.arm:
        return list(config["arms"])
    return list(dict.fromkeys(args.arm))


def _select_seeds(args: argparse.Namespace, config: dict) -> list[int]:
    if not args.seed or "all" in args.seed:
        return [int(seed) for seed in config["evaluation"]["seeds"]]
    try:
        return [int(seed) for seed in args.seed]
    except ValueError as exc:
        raise ValueError(f"--seed values must be integers or 'all', got {args.seed}") from exc


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = parse_args(argv)

    if args.list_arms:
        print("available arms:", ", ".join(ARM_NAMES))
        return 0

    config = load_config(args.config)
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    validate_config(config)

    arms = _select_arms(args, config)
    seeds = _select_seeds(args, config)
    results_dir = resolve_project_path(
        args.results_dir or config["output"]["results_dir"]
    )
    log_dir = resolve_project_path(args.log_dir or config["output"]["log_dir"])

    run_id = make_run_id("m1")
    logger = RunLogger(run_id, log_dir)
    try:
        logger.log(f"run_id={run_id}")
        logger.log(f"config_hash={config_hash(config)}")
        logger.save_config(config)
        logger.save_json("env.json", collect_env())

        dataset = load_dataset(config["dataset"]["name"], config)
        logger.log(
            f"dataset={config['dataset']['name']} "
            f"shape={tuple(dataset.X.shape)} "
            f"subjects={dataset.metadata['subject_id'].nunique()} "
            f"classes={dataset.class_names}"
        )
        logger.save_json("dataset_summary.json", dataset_summary(dataset))
        logger.save_json(
            "dataset_fingerprint.json",
            {"fingerprint": dataset_fingerprint(dataset)},
        )
        if config["dataset"]["name"] == "synthetic":
            logger.log(
                "WARNING: synthetic data is for pipeline validation only; "
                "the resulting accuracy is not a scientific result."
            )

        folds = make_loso_splits(dataset.metadata)
        split_path = save_folds(folds, logger.log_dir / "splits.json")
        logger.log(f"folds={len(folds)} split_file={split_path}")

        if args.dry_run:
            report = dry_run_report(dataset, max_folds=args.max_folds)
            print(report.to_string(index=False))
            logger.save_json("dry_run_folds.json", report.to_dict(orient="records"))
            return 0

        if args.check_only:
            logger.log(f"all split/leakage checks passed for {len(folds)} fold(s)")
            return 0

        results, predictions = run_all(
            dataset=dataset,
            config=config,
            arms=arms,
            seeds=seeds,
            subject=args.subject,
            max_folds=args.max_folds,
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
            canary_failed = False
            for arm in arms:
                for seed in seeds:
                    result = run_canary(
                        dataset=dataset,
                        arm=arm,
                        seed=seed,
                        config=config,
                        subject=args.subject,
                        max_folds=args.max_folds,
                    )
                    logger.log(f"canary arm={arm} seed={seed}: {result.message}")
                    canary_failed = canary_failed or (not result.passed)
            if canary_failed:
                logger.log("canary FAILED — inspect leakage before reporting results")
                return 3

        logger.log("done")
        return 0
    except LeakageError as exc:
        logger.log(f"LEAKAGE CHECK FAILED: {exc}")
        return 2
    except RuntimeError as exc:
        logger.log(f"ERROR: {exc}")
        return 2
    except Exception as exc:  # keep the CLI human-readable, but preserve traceback
        logger.log(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        logger.save_text("traceback.txt", traceback.format_exc())
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
