"""Compare the popularity and genre baselines on the frozen validation split."""

import argparse
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

import pandas as pd

from src.data.id_mapper import IdMapper
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.evaluation.prepare import (
    load_splits,
    normalized_config,
    split_summary,
    validation_coverage,
)
from src.files import read_json, sha256, write_json
from src.models.content_based import ContentBasedRecommender, tokenize_genres
from src.models.popularity import PopularityRecommender
from src.models.tmdb_recommender import TMDBRecommender


@dataclass(frozen=True)
class ValidationCase:
    user_id: int
    favorites: tuple[int, ...]
    targets: tuple[int, ...]


def build_validation_cases(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[tuple[int, ...], list[ValidationCase]]:
    """Cases for users with train favorites and a validation target in the catalog."""
    positive_train = train.loc[train["is_positive"]]
    targets = validation.loc[validation["is_positive"]]
    candidate_ids = tuple(
        sorted(int(value) for value in positive_train["tmdbId"].unique())
    )
    catalog = set(candidate_ids)
    histories = {
        int(user_id): tuple(sorted(int(value) for value in group["tmdbId"]))
        for user_id, group in positive_train.groupby("userId")
    }
    cases = []
    for user_id, group in targets.groupby("userId", sort=True):
        favorites = histories.get(int(user_id), ())
        # Targets outside the catalog stay: the all-targets metrics count them.
        user_targets = tuple(sorted(int(value) for value in group["tmdbId"]))
        if favorites and catalog.intersection(user_targets):
            cases.append(ValidationCase(int(user_id), favorites, user_targets))
    return candidate_ids, cases


def evaluate_recommender(
    recommender, cases: list[ValidationCase], candidate_ids, k: int = 10
) -> dict:
    """Macro-average Recall and NDCG at k over the cases; None without any case."""
    catalog = set(candidate_ids)
    recall_key, ndcg_key = f"recall_at_{k}", f"ndcg_at_{k}"
    per_user = []
    for case in cases:
        targets = set(case.targets)
        known_targets = targets & catalog
        # A missing model still counts: its users score zero instead of disappearing.
        recommendations = (
            [] if recommender is None else recommender.recommend(case.favorites, n=k)
        )
        ranked = [row["tmdb_id"] for row in recommendations]
        per_user.append(
            {
                "user_id": case.user_id,
                "favorite_count": len(set(case.favorites)),
                "target_count": len(targets),
                "target_count_in_catalog": len(known_targets),
                "target_count_outside_catalog": len(targets - catalog),
                "recommended_tmdb_ids": ranked,
                f"hits_at_{k}": len(set(ranked) & known_targets),
                recall_key: recall_at_k(ranked, known_targets, k),
                ndcg_key: ndcg_at_k(ranked, known_targets, k),
                f"{recall_key}_all_targets": recall_at_k(ranked, targets, k),
                f"{ndcg_key}_all_targets": ndcg_at_k(ranked, targets, k),
            }
        )

    def average(key):
        return fmean(row[key] for row in per_user) if per_user else None

    return {
        "users_evaluated": len(per_user),
        "metrics": {key: average(key) for key in (recall_key, ndcg_key)},
        "metrics_all_targets": {
            key: average(f"{key}_all_targets") for key in (recall_key, ndcg_key)
        },
        "empty_recommendation_users": sum(
            not row["recommended_tmdb_ids"] for row in per_user
        ),
        "mean_recommendations": fmean(
            len(row["recommended_tmdb_ids"]) for row in per_user
        )
        if per_user
        else None,
        "per_user": per_user,
    }


def _genre_catalog(train: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Metadata of the canonical movies with positive train interactions."""
    movie_ids = train.loc[train["is_positive"], ["movieId"]].drop_duplicates()
    catalog = movie_ids.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left",
        validate="one_to_one",
    )
    return catalog.sort_values("movieId").reset_index(drop=True)


def compare_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Fit both baselines on train and score them on the shared validation cohort."""
    candidate_ids, cases = build_validation_cases(train, validation)
    mapper = IdMapper(links)
    genre_catalog = _genre_catalog(train, movies)
    usable_genres = any(tokenize_genres(value) for value in genre_catalog["genres"])
    genre_model = (
        TMDBRecommender(ContentBasedRecommender(genre_catalog), mapper)
        if usable_genres
        else None
    )
    coverage = validation_coverage(train, validation)
    known_count = sum(len(set(case.targets) & set(candidate_ids)) for case in cases)
    outside_count = sum(len(set(case.targets) - set(candidate_ids)) for case in cases)
    coverage.update(
        {
            "eligible_positive_targets": known_count,
            "eligible_positive_targets_outside_catalog": outside_count,
            "eligible_target_coverage": known_count / coverage["positive_targets"]
            if coverage["positive_targets"]
            else None,
            "users_with_history_without_catalog_targets": coverage[
                "users_with_positive_history"
            ]
            - len(cases),
        }
    )
    popularity = evaluate_recommender(
        PopularityRecommender(train), cases, candidate_ids, k
    )
    genres = evaluate_recommender(genre_model, cases, candidate_ids, k)
    genres["catalog_has_usable_genres"] = usable_genres
    return {
        "protocol": {
            "split": "validation",
            "k": k,
            "catalog": "TMDB films with positive train interactions only",
            "profiles": "positive train interactions simulate favorites",
            "exclusions": "favorites only; lower ratings are not exclusions",
            "cohort": "users with train favorites and at least one validation target in the catalog",
            "targets": "positive validation interactions; binary relevance",
            "main_metrics": "catalog targets only; unweighted mean across the same eligible users",
            "metrics_all_targets": "all validation targets for the same eligible users, including unavailable films",
            "recall": "hits in the first k ranks divided by the number of distinct targets",
            "ndcg": "binary gain / log2(rank + 1); ideal ranking has min(k, target count) hits",
            "empty_predictions": "keep the user with zero scores; do not add a fallback",
            "empty_cohort": "null metrics",
            "genres_fit": "TF-IDF fitted on canonical metadata of the shared train catalog only; IMAX excluded",
            "ties": {
                "popularity": "ascending TMDB ID",
                "genres": "ascending canonical MovieLens ID",
            },
            "test_usage": "not used to fit, select or score either baseline",
        },
        "catalog": {"movies": len(candidate_ids)},
        "coverage": coverage,
        "models": {"popularity": popularity, "genres": genres},
    }


def build_report(data_dir: Path, config_path: Path, k: int = 10) -> dict:
    config = read_json(config_path)
    tables, preparation, splits = load_splits(data_dir, config)
    report = compare_models(
        splits["train"], splits["validation"], tables["movies"], tables["links"], k
    )
    report.update(
        {
            "config": normalized_config(config),
            "sources": {
                name: {"sha256": sha256(data_dir / name)}
                for name in ("ratings.csv", "links.csv", "movies.csv")
            },
            "environment": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ("numpy", "pandas", "scikit-learn")
                },
            },
            "preparation": preparation,
            "splits": {
                name: split_summary(splits[name]) for name in ("train", "validation")
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config/evaluation.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/baseline_comparison.json")
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    report = build_report(args.data_dir, args.config, args.k)
    write_json(args.output, report)
    print(f"Validation users: {report['coverage']['eligible_users']}")
    for name, result in report["models"].items():
        print(f"{name}: {json.dumps(result['metrics'], allow_nan=False)}")
    print(f"Baseline comparison report: {args.output}")


if __name__ == "__main__":
    main()
