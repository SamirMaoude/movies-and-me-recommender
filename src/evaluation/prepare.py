"""Audit the evaluation data: preparation counts, splits and validation coverage."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.exploration import load_movielens
from src.evaluation.data import prepare_interactions, temporal_split, utc_cutoff
from src.files import read_json, sha256, write_json


def load_splits(
    data_dir: Path, config: dict
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, pd.DataFrame]]:
    """Read the CSV files, prepare the interactions and split them as configured."""
    tables = load_movielens(data_dir)
    interactions, preparation = prepare_interactions(
        tables["ratings"], tables["links"], config["positive_threshold"]
    )
    splits = temporal_split(interactions, config["train_end"], config["validation_end"])
    return tables, preparation, splits


def normalized_config(config: dict) -> dict:
    """Evaluation settings as written in reports, with cutoffs in UTC."""
    return {
        "positive_threshold": float(config["positive_threshold"]),
        "train_end": utc_cutoff(config["train_end"]).isoformat(),
        "validation_end": utc_cutoff(config["validation_end"]).isoformat(),
    }


def split_summary(interactions: pd.DataFrame) -> dict:
    """Count a split's interactions, users and films, and give its time range."""
    times = pd.to_datetime(interactions["timestamp"], unit="s", utc=True)
    positives = int(interactions["is_positive"].sum())
    return {
        "interactions": len(interactions),
        "positive_interactions": positives,
        "other_interactions": len(interactions) - positives,
        "users": int(interactions["userId"].nunique()),
        "movies": int(interactions["tmdbId"].nunique()),
        "first_interaction": None if interactions.empty else times.min().isoformat(),
        "last_interaction": None if interactions.empty else times.max().isoformat(),
    }


def validation_coverage(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Count validation targets by whether their user and film are known from train."""
    positive_train = train.loc[train["is_positive"]]
    targets = validation.loc[validation["is_positive"]]
    history_users = set(positive_train["userId"])
    candidate_movies = set(positive_train["tmdbId"])
    known_user = targets["userId"].isin(history_users)
    known_movie = targets["tmdbId"].isin(candidate_movies)
    eligible = known_user & known_movie
    target_users = int(targets["userId"].nunique())
    eligible_users = int(targets.loc[eligible, "userId"].nunique())
    return {
        "reference": "positive train interactions only",
        "candidate_movies": len(candidate_movies),
        "positive_targets": len(targets),
        "users_with_positive_targets": target_users,
        "users_with_positive_history": int(targets.loc[known_user, "userId"].nunique()),
        "eligible_users": eligible_users,
        "positive_targets_in_catalog": int(known_movie.sum()),
        "positive_targets_outside_catalog": int((~known_movie).sum()),
        "targets_known_user_known_movie": int(eligible.sum()),
        "targets_known_user_unknown_movie": int((known_user & ~known_movie).sum()),
        "targets_unknown_user_known_movie": int((~known_user & known_movie).sum()),
        "targets_unknown_user_unknown_movie": int((~known_user & ~known_movie).sum()),
        "catalog_coverage": float(known_movie.mean()) if len(targets) else None,
        "eligible_user_coverage": eligible_users / target_users
        if target_users
        else None,
    }


def build_report(data_dir: Path, config_path: Path) -> dict:
    config = read_json(config_path)
    _, preparation, splits = load_splits(data_dir, config)
    return {
        "config": normalized_config(config),
        "policies": {
            "positive_feedback": "rating >= positive_threshold",
            "missing_tmdb": "exclude and count",
            "canonical_movie_id": "smallest MovieLens ID declared in links for each TMDB ID",
            "duplicate_user_movie": "earliest timestamp, then smallest source movieId",
            "cutoff_boundaries": "equal timestamps belong to the next split",
            "request_history": "positive interactions simulate favorites",
            "recommendation_exclusions": "supplied favorites only; lower ratings are not exclusions",
            "test_usage": "descriptive data counts only; no model evaluation",
        },
        "sources": {
            name: {"sha256": sha256(data_dir / name)}
            for name in ("ratings.csv", "links.csv")
        },
        "preparation": preparation,
        "splits": {name: split_summary(frame) for name, frame in splits.items()},
        "validation_coverage": validation_coverage(
            splits["train"], splits["validation"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config/evaluation.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/evaluation_data.json")
    )
    args = parser.parse_args()
    write_json(args.output, build_report(args.data_dir, args.config))
    print(f"Evaluation data report: {args.output}")


if __name__ == "__main__":
    main()
