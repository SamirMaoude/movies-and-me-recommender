"""Train ALS candidates on train, compare on validation and save the best ALS."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import platform

from src.data.exploration import load_movielens
from src.evaluation.compare import build_validation_cases, compare_models, evaluate_recommender
from src.evaluation.data import prepare_interactions, temporal_split, utc_cutoff
from src.evaluation.prepare import split_summary
from src.models.collaborative import ALSConfig, ImplicitRecommender


def load_experiment_config(path: Path) -> dict:
    """Validate the declared trials before loading data or fitting a model."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"k", "trials"}:
        raise ValueError("ALS experiments must contain k and trials.")
    if isinstance(config["k"], bool) or not isinstance(config["k"], int) or config["k"] <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(config["trials"], list) or not config["trials"]:
        raise ValueError("Declare at least one ALS trial.")
    names = set()
    fields = set(asdict(ALSConfig()))
    for trial in config["trials"]:
        if not isinstance(trial, dict) or set(trial) != fields | {"name"}:
            raise ValueError("Each trial needs a name and all ALSConfig parameters.")
        name = trial["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Trial names must be nonempty and unique.")
        names.add(name)
        ALSConfig(**{key: value for key, value in trial.items() if key != "name"})
    return config


def _selection_score(metrics: dict, k: int) -> tuple[float, float]:
    values = (metrics[f"ndcg_at_{k}"], metrics[f"recall_at_{k}"])
    if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) for value in values):
        raise ValueError("Selection requires finite validation metrics on an eligible cohort.")
    return tuple(float(value) for value in values)


def select_trial(trials: list[dict], k: int = 10) -> dict:
    """Prefer NDCG, then Recall; retain the first declared trial on exact ties."""
    if not trials:
        raise ValueError("No ALS trial to select.")
    return max(trials, key=lambda trial: _selection_score(trial["evaluation"]["metrics"], k))


def run_experiment(
    data_dir: Path,
    evaluation_config_path: Path,
    experiment_config_path: Path,
    artifact_dir: Path | None = None,
) -> dict:
    """Fit each declared ALS once and preserve the chosen train-only candidate.

    Validation selects parameters, never supplies favorites or model weights.
    The saved ALS remains fitted on train so its scores can be reproduced.
    Final refitting and test evaluation belong to a later, frozen selection.
    """
    if artifact_dir is not None and artifact_dir.exists():
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}. Choose a new --artifact-dir.")
    experiment_config = load_experiment_config(experiment_config_path)
    evaluation_config = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
    required = {"positive_threshold", "train_end", "validation_end"}
    if not isinstance(evaluation_config, dict) or set(evaluation_config) != required:
        raise ValueError("Evaluation config must contain positive_threshold, train_end and validation_end.")
    tables = load_movielens(data_dir)
    interactions, preparation = prepare_interactions(
        tables["ratings"], tables["links"], evaluation_config["positive_threshold"],
    )
    splits = temporal_split(interactions, evaluation_config["train_end"], evaluation_config["validation_end"])
    train, validation = splits["train"], splits["validation"]
    candidate_ids, cases = build_validation_cases(train, validation)
    if not cases:
        raise ValueError("Cannot select ALS parameters without eligible validation users.")
    k = experiment_config["k"]
    report = compare_models(train, validation, tables["movies"], tables["links"], k)
    trials, fitted = [], []
    for declared in experiment_config["trials"]:
        config = ALSConfig(**{key: value for key, value in declared.items() if key != "name"})
        recommender = ImplicitRecommender.fit(train, tables["links"], config)
        if recommender.training_data.tmdb_ids != candidate_ids:
            raise ValueError("ALS and the baselines must use exactly the same candidate catalog.")
        evaluation = evaluate_recommender(recommender, cases, candidate_ids, k)
        trials.append({"name": declared["name"], "config": asdict(config), "evaluation": evaluation})
        fitted.append(recommender)
    selected = select_trial(trials, k)
    selected_index = next(index for index, trial in enumerate(trials) if trial is selected)
    recommender = fitted[selected_index]
    report["models"]["als"] = selected["evaluation"]
    validation_winner = max(
        report["models"], key=lambda name: _selection_score(report["models"][name]["metrics"], k),
    )
    config = {
        "positive_threshold": float(evaluation_config["positive_threshold"]),
        "train_end": utc_cutoff(evaluation_config["train_end"]).isoformat(),
        "validation_end": utc_cutoff(evaluation_config["validation_end"]).isoformat(),
    }
    sources = {
        name: {"sha256": hashlib.sha256((data_dir / name).read_bytes()).hexdigest()}
        for name in ("ratings.csv", "links.csv", "movies.csv")
    }
    selection = {
        "metric": f"ndcg_at_{k}",
        "tie_breaker": f"recall_at_{k}, then declared trial order",
        "selected_trial": selected["name"],
        "validation_winner": validation_winner,
        "fit_split": "train",
        "final_test_evaluated": False,
    }
    report["protocol"]["ties"]["als"] = "ascending TMDB ID"
    report["protocol"].update({
        "empty_predictions": "keep eligible users with zero scores; do not pad partial or empty rankings",
        "als_fit": "binary positive train matrix; implicit applies alpha once at fit and profile recalculation",
        "als_inference": "request favorites only, with recalculate_user=True; no stored user factor is selected",
        "als_fallback": "train popularity when no supplied favorite belongs to the trained catalog",
        "als_fallback_scope": "not exercised by the eligible validation cohort; checked separately with request tests",
        "test_usage": "not used to fit, select or score any candidate",
    })
    report.update({
        "config": config,
        "experiment_config": experiment_config,
        "sources": sources,
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in (
                "implicit", "numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl",
            )},
        },
        "preparation": preparation,
        "splits": {name: split_summary(splits[name]) for name in ("train", "validation")},
        "training_matrix": {
            "users": len(recommender.training_data.user_ids),
            "movies": len(recommender.training_data.tmdb_ids),
            "positive_pairs": int(recommender.training_data.matrix.nnz),
            "values": "binary; rating magnitude is not a weight",
        },
        "trials": trials,
        "selection": selection,
    })
    if artifact_dir is not None:
        recommender.save(artifact_dir, metadata={
            "config": config,
            "sources": sources,
            "selection": selection,
            "validation_metrics": selected["evaluation"]["metrics"],
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("config/evaluation.json"))
    parser.add_argument("--config", type=Path, default=Path("config/als.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/implicit-als"))
    parser.add_argument("--output", type=Path, default=Path("reports/als_comparison.json"))
    args = parser.parse_args()
    report = run_experiment(args.data_dir, args.evaluation_config, args.config, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for trial in report["trials"]:
        print(f"{trial['name']}: {json.dumps(trial['evaluation']['metrics'], allow_nan=False)}")
    print(f"Selected ALS trial: {report['selection']['selected_trial']}")
    print(f"Best validation model: {report['selection']['validation_winner']}")
    print(f"Model: {args.artifact_dir}")
    print(f"Comparison report: {args.output}")


if __name__ == "__main__":
    main()
