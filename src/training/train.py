"""Fit the ALS trials on train, compare them on validation and save the best."""

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mlflow

from src.evaluation.compare import (
    build_validation_cases,
    compare_models,
    evaluate_recommender,
)
from src.evaluation.prepare import load_splits, normalized_config, split_summary
from src.files import read_json, sha256, write_json
from src.models.collaborative import ALSConfig, ImplicitRecommender


def _selection_score(metrics: dict, k: int) -> tuple[float, float]:
    return metrics[f"ndcg_at_{k}"], metrics[f"recall_at_{k}"]


def select_trial(trials: list[dict], k: int = 10) -> dict:
    """Prefer NDCG, then Recall; the first declared trial wins exact ties."""
    return max(
        trials, key=lambda trial: _selection_score(trial["evaluation"]["metrics"], k)
    )


def _collect_provenance(data_dir: Path) -> dict:
    """Record the Git state, the input checksums and the package versions."""
    repo_dir = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, encoding="utf-8"
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_dir, text=True, encoding="utf-8"
    )
    packages = (
        "implicit",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "threadpoolctl",
        "mlflow",
    )
    return {
        "git": {"commit": commit, "dirty": bool(status.strip())},
        "sources": {
            name: {"sha256": sha256(data_dir / name)}
            for name in ("ratings.csv", "links.csv", "movies.csv")
        },
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in packages},
        },
    }


def run_experiment(
    data_dir: Path,
    evaluation_config_path: Path,
    experiment_config_path: Path,
    artifact_dir: Path | None = None,
) -> dict:
    """Fit each ALS trial on train, select one on validation and save it."""
    if artifact_dir is not None and artifact_dir.exists():
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")
    experiment_config = read_json(experiment_config_path)
    evaluation_config = read_json(evaluation_config_path)
    provenance = _collect_provenance(data_dir)
    tables, preparation, splits = load_splits(data_dir, evaluation_config)
    train, validation = splits["train"], splits["validation"]
    candidate_ids, cases = build_validation_cases(train, validation)
    if not cases:
        raise ValueError("No eligible validation users to select an ALS trial.")
    k = experiment_config["k"]
    report = compare_models(train, validation, tables["movies"], tables["links"], k)
    provenance.update(
        {
            "evaluation_config": evaluation_config,
            "experiment_config": experiment_config,
            "preparation": preparation,
            "splits": {
                name: split_summary(splits[name]) for name in ("train", "validation")
            },
        }
    )

    mlflow.set_experiment("movies-and-me-recommender")
    trials, fitted = [], []
    for declared in experiment_config["trials"]:
        name = declared["name"]
        config = ALSConfig(
            **{key: value for key, value in declared.items() if key != "name"}
        )
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id
            mlflow.set_tags(
                {
                    "git.commit": provenance["git"]["commit"],
                    "git.dirty": str(provenance["git"]["dirty"]).lower(),
                }
            )
            mlflow.log_dict(provenance, "provenance.json")
            mlflow.log_params({**asdict(config), **evaluation_config, "k": k})
            start = perf_counter()
            recommender = ImplicitRecommender.fit(train, tables["links"], config)
            training_seconds = perf_counter() - start
            evaluation = evaluate_recommender(recommender, cases, candidate_ids, k)
            mlflow.log_metrics(
                {
                    f"validation_{metric}": value
                    for metric, value in evaluation["metrics"].items()
                }
            )
            mlflow.log_metric("training_seconds", training_seconds)
            mlflow.log_metric("validation_users", evaluation["users_evaluated"])
            mlflow.log_dict(evaluation, "evaluation.json")
            with TemporaryDirectory(prefix="als-") as temporary:
                model_dir = Path(temporary) / "model"
                recommender.save(
                    model_dir,
                    metadata={
                        **provenance,
                        "run_id": run_id,
                        "trial": name,
                        "fit_split": "train",
                        "evaluation_config": evaluation_config,
                        "validation_metrics": evaluation["metrics"],
                    },
                )
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
        trials.append(
            {
                "name": name,
                "run_id": run_id,
                "config": asdict(config),
                "evaluation": evaluation,
                "training_seconds": training_seconds,
            }
        )
        fitted.append(recommender)

    selected = select_trial(trials, k)
    # Keep the train-only fit that produced the validation scores: validation
    # chooses the trial without training it, and test data stays unused.
    recommender = fitted[trials.index(selected)]
    report["models"]["als"] = selected["evaluation"]
    validation_winner = max(
        report["models"],
        key=lambda model: _selection_score(report["models"][model]["metrics"], k),
    )
    config = normalized_config(evaluation_config)
    selection = {
        "metric": f"ndcg_at_{k}",
        "tie_breaker": f"recall_at_{k}, then declared trial order",
        "selected_trial": selected["name"],
        "selected_run_id": selected["run_id"],
        "validation_winner": validation_winner,
        "fit_split": "train",
        "final_test_evaluated": False,
    }
    report["protocol"]["ties"]["als"] = "ascending TMDB ID"
    report["protocol"].update(
        {
            "empty_predictions": "keep eligible users with zero scores; do not pad partial or empty rankings",
            "als_fit": "binary positive train matrix; implicit applies alpha once at fit and profile recalculation",
            "als_inference": "request favorites only, with recalculate_user=True; no stored user factor is selected",
            "als_fallback": "train popularity when no supplied favorite belongs to the trained catalog",
            "als_fallback_scope": "not exercised by the eligible validation cohort; checked separately with request tests",
            "test_usage": "not used to fit, select or score any candidate",
        }
    )
    report.update(
        {
            "config": config,
            "experiment_config": experiment_config,
            "sources": provenance["sources"],
            "environment": provenance["environment"],
            "git": provenance["git"],
            "preparation": preparation,
            "splits": provenance["splits"],
            "training_matrix": {
                "users": len(recommender.training_data.user_ids),
                "movies": len(recommender.training_data.tmdb_ids),
                "positive_pairs": recommender.training_data.matrix.nnz,
                "values": "binary; rating magnitude is not a weight",
            },
            "trials": trials,
            "selection": selection,
        }
    )
    if artifact_dir is not None:
        recommender.save(
            artifact_dir,
            metadata={
                **provenance,
                "run_id": selected["run_id"],
                "config": config,
                "selection": selection,
                "validation_metrics": selected["evaluation"]["metrics"],
            },
        )
    mlflow.log_dict(report, "comparison.json", run_id=selected["run_id"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--evaluation-config", type=Path, default=Path("config/evaluation.json")
    )
    parser.add_argument("--config", type=Path, default=Path("config/als.json"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/implicit-als")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/als_comparison.json")
    )
    args = parser.parse_args()
    report = run_experiment(
        args.data_dir, args.evaluation_config, args.config, args.artifact_dir
    )
    write_json(args.output, report)
    for trial in report["trials"]:
        metrics = json.dumps(trial["evaluation"]["metrics"], allow_nan=False)
        print(f"{trial['name']}: {metrics}")
    print(f"Selected ALS trial: {report['selection']['selected_trial']}")
    print(f"Selected MLflow run: {report['selection']['selected_run_id']}")
    print(f"Best validation model: {report['selection']['validation_winner']}")
    print(f"Model: {args.artifact_dir}")
    print(f"Comparison report: {args.output}")


if __name__ == "__main__":
    main()
