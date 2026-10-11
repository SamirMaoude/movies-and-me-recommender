"""Run the preparation or ALS training stage declared in dvc.yaml."""

import argparse
import json
import os
from math import isfinite
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.evaluation.data import utc_cutoff
from src.evaluation.prepare import build_report


def load_params(path: Path) -> dict:
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(params, dict) or set(params) != {"evaluation", "als", "tracking"}:
        raise ValueError("params.yaml must contain evaluation, als and tracking.")
    return params


def _evaluation_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != {
        "positive_threshold", "train_end", "validation_end",
    }:
        raise ValueError("evaluation needs positive_threshold, train_end and validation_end.")
    threshold = config["positive_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not isfinite(threshold)
        or not 0.5 <= threshold <= 5.0
    ):
        raise ValueError("positive_threshold must be a finite number between 0.5 and 5.")
    if utc_cutoff(config["train_end"]) >= utc_cutoff(config["validation_end"]):
        raise ValueError("train_end must be strictly before validation_end.")
    return {**config, "positive_threshold": float(threshold)}


def _require_new_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Use dvc repro to manage stage outputs.")


def _write_json(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def prepare_stage(
    params_path: Path = Path("params.yaml"),
    data_dir: Path = Path("data/raw"),
    evaluation_path: Path = Path("artifacts/dvc/evaluation.json"),
    report_path: Path = Path("reports/dvc/data.json"),
) -> dict:
    """Audit the frozen splits and export the evaluation JSON used by training."""
    _require_new_outputs(evaluation_path, report_path)
    evaluation = _evaluation_config(load_params(params_path)["evaluation"])
    with TemporaryDirectory(prefix="dvc-prepare-") as temporary:
        temporary_config = Path(temporary) / "evaluation.json"
        _write_json(temporary_config, evaluation)
        report = build_report(data_dir, temporary_config)
    _write_json(evaluation_path, evaluation)
    _write_json(report_path, report)
    return report


def _validation_metrics(report: dict, k: int) -> dict:
    metrics = {}
    for name, result in report["models"].items():
        metrics[name] = {}
        for metric in (f"recall_at_{k}", f"ndcg_at_{k}"):
            value = result["metrics"][metric]
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"Expected a finite validation metric: {name}.{metric}.")
            metrics[name][metric] = float(value)
    return metrics


def train_stage(
    params_path: Path = Path("params.yaml"),
    data_dir: Path = Path("data/raw"),
    evaluation_path: Path = Path("artifacts/dvc/evaluation.json"),
    model_dir: Path = Path("artifacts/dvc/model"),
    report_path: Path = Path("reports/dvc/comparison.json"),
    metrics_path: Path = Path("reports/dvc/metrics.json"),
) -> dict:
    """Run the existing train-only selection; leave all registry aliases alone."""
    import mlflow

    from src.training.train import load_experiment_config, run_experiment

    _require_new_outputs(model_dir, report_path, metrics_path)
    params = load_params(params_path)
    tracking = params["tracking"]
    if not isinstance(tracking, dict) or set(tracking) != {"uri"}:
        raise ValueError("tracking must contain uri.")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", tracking["uri"])
    if not isinstance(tracking_uri, str) or not tracking_uri.strip():
        raise ValueError("Set a nonempty MLflow tracking URI.")
    evaluation = _evaluation_config(json.loads(evaluation_path.read_text(encoding="utf-8")))
    declared_evaluation = _evaluation_config(params["evaluation"])
    if evaluation != declared_evaluation:
        raise ValueError("Prepared evaluation config differs from params.yaml. Run dvc repro prepare first.")
    if mlflow.active_run() is not None:
        raise ValueError("Finish the active MLflow run before starting the DVC training stage.")
    with TemporaryDirectory(prefix="dvc-train-") as temporary:
        experiments_path = Path(temporary) / "als.json"
        _write_json(experiments_path, params["als"])
        experiments = load_experiment_config(experiments_path)
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            report = run_experiment(data_dir, evaluation_path, experiments_path, model_dir)
        finally:
            mlflow.set_tracking_uri(previous_uri)
    _write_json(report_path, report)
    _write_json(metrics_path, _validation_metrics(report, experiments["k"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare = subparsers.add_parser("prepare", help="Audit data and export the frozen evaluation config.")
    train = subparsers.add_parser("train", help="Train ALS trials and compare on validation.")
    for command in (prepare, train):
        command.add_argument("--params", type=Path, default=Path("params.yaml"))
        command.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        command.add_argument("--evaluation-config", type=Path, default=Path("artifacts/dvc/evaluation.json"))
    prepare.add_argument("--output", type=Path, default=Path("reports/dvc/data.json"))
    train.add_argument("--artifact-dir", type=Path, default=Path("artifacts/dvc/model"))
    train.add_argument("--output", type=Path, default=Path("reports/dvc/comparison.json"))
    train.add_argument("--metrics", type=Path, default=Path("reports/dvc/metrics.json"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            report = prepare_stage(args.params, args.data_dir, args.evaluation_config, args.output)
            print(f"Data audit: {args.output}")
        else:
            report = train_stage(
                args.params, args.data_dir, args.evaluation_config,
                args.artifact_dir, args.output, args.metrics,
            )
            print(f"Selected ALS trial: {report['selection']['selected_trial']}")
            print(f"Selected MLflow run: {report['selection']['selected_run_id']}")
            print(f"Model: {args.artifact_dir}")
            print(f"Comparison: {args.output}")
            print(f"Validation metrics: {args.metrics}")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
