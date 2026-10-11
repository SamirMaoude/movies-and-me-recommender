"""Run the preparation or ALS training stage declared in dvc.yaml."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.evaluation.prepare import build_report
from src.files import write_json


def load_params(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require_new_outputs(*paths: Path) -> None:
    # dvc repro removes stage outputs first; a manual run must not replace them.
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")


def prepare_stage(
    params_path: Path = Path("params.yaml"),
    data_dir: Path = Path("data/raw"),
    evaluation_path: Path = Path("artifacts/dvc/evaluation.json"),
    report_path: Path = Path("reports/dvc/data.json"),
) -> dict:
    """Audit the splits and export the evaluation settings used by training."""
    _require_new_outputs(evaluation_path, report_path)
    evaluation = load_params(params_path)["evaluation"]
    # Write the stage outputs only once the audit has succeeded.
    with TemporaryDirectory(prefix="dvc-prepare-") as temporary:
        config_path = Path(temporary) / "evaluation.json"
        write_json(config_path, evaluation)
        report = build_report(data_dir, config_path)
    write_json(evaluation_path, evaluation)
    write_json(report_path, report)
    return report


def train_stage(
    params_path: Path = Path("params.yaml"),
    data_dir: Path = Path("data/raw"),
    evaluation_path: Path = Path("artifacts/dvc/evaluation.json"),
    model_dir: Path = Path("artifacts/dvc/model"),
    report_path: Path = Path("reports/dvc/comparison.json"),
    metrics_path: Path = Path("reports/dvc/metrics.json"),
) -> dict:
    """Train and compare the ALS trials; registry aliases are left unchanged."""
    # Imported here so that the prepare stage needs neither MLflow nor implicit.
    import mlflow

    from src.training.train import run_experiment

    _require_new_outputs(model_dir, report_path, metrics_path)
    params = load_params(params_path)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", params["tracking"]["uri"])
    with TemporaryDirectory(prefix="dvc-train-") as temporary:
        experiments_path = Path(temporary) / "als.json"
        write_json(experiments_path, params["als"])
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            report = run_experiment(
                data_dir, evaluation_path, experiments_path, model_dir
            )
        finally:
            mlflow.set_tracking_uri(previous_uri)
    write_json(report_path, report)
    metrics = {name: model["metrics"] for name, model in report["models"].items()}
    write_json(metrics_path, metrics)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="Audit data and export the frozen evaluation config."
    )
    train = subparsers.add_parser(
        "train", help="Train ALS trials and compare on validation."
    )
    for command in (prepare, train):
        command.add_argument("--params", type=Path, default=Path("params.yaml"))
        command.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        command.add_argument(
            "--evaluation-config",
            type=Path,
            default=Path("artifacts/dvc/evaluation.json"),
        )
    prepare.add_argument("--output", type=Path, default=Path("reports/dvc/data.json"))
    train.add_argument("--artifact-dir", type=Path, default=Path("artifacts/dvc/model"))
    train.add_argument(
        "--output", type=Path, default=Path("reports/dvc/comparison.json")
    )
    train.add_argument("--metrics", type=Path, default=Path("reports/dvc/metrics.json"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            prepare_stage(
                args.params, args.data_dir, args.evaluation_config, args.output
            )
            print(f"Data audit: {args.output}")
        else:
            report = train_stage(
                args.params,
                args.data_dir,
                args.evaluation_config,
                args.artifact_dir,
                args.output,
                args.metrics,
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
