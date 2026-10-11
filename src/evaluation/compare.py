"""Compare the popularity and genre baselines on the frozen validation split."""

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

from src.data.exploration import load_movielens
from src.data.id_mapper import IdMapper
from src.evaluation.data import prepare_interactions, temporal_split, utc_cutoff
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.evaluation.prepare import split_summary, validation_coverage
from src.models.content_based import ContentBasedRecommender, tokenize_genres
from src.models.popularity import PopularityRecommender
from src.models.tmdb_recommender import TMDBRecommender


@dataclass(frozen=True)
class ValidationCase:
    user_id: int
    favorites: tuple[int, ...]
    targets: tuple[int, ...]


def _positive_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept prepared interactions, not raw ratings or coerced feedback."""
    required = ("userId", "tmdbId", "is_positive")
    if not isinstance(frame, pd.DataFrame) or any(
        list(frame.columns).count(column) != 1 for column in required
    ):
        raise ValueError("Expected prepared userId, tmdbId and is_positive columns.")
    for column in ("userId", "tmdbId"):
        if (
            not pd.api.types.is_integer_dtype(frame[column].dtype)
            or frame[column].isna().any()
            or frame[column].le(0).any()
        ):
            raise ValueError(f"{column} must contain positive integer IDs.")
    if not pd.api.types.is_bool_dtype(frame["is_positive"].dtype) or frame["is_positive"].isna().any():
        raise ValueError("is_positive must contain booleans.")
    if frame.duplicated(["userId", "tmdbId"]).any():
        raise ValueError("Resolve repeated user/TMDB interactions before evaluation.")
    return frame.loc[frame["is_positive"]]


def build_validation_cases(
    train: pd.DataFrame, validation: pd.DataFrame,
) -> tuple[tuple[int, ...], list[ValidationCase]]:
    """Use train favorites and future positives; keep a common eligible cohort.

    Callers supply the prepared temporal splits. A case needs at least one
    train favorite and one validation target in the positive train catalog.
    All its validation targets are retained to expose out-of-catalog targets.
    Lower ratings neither build the profile nor exclude recommendations.
    """
    positive_train, targets = _positive_rows(train), _positive_rows(validation)
    train_pairs = pd.MultiIndex.from_frame(train[["userId", "tmdbId"]])
    validation_pairs = pd.MultiIndex.from_frame(validation[["userId", "tmdbId"]])
    if len(train_pairs.intersection(validation_pairs)):
        raise ValueError("Train and validation must not share a user/TMDB pair.")
    candidate_ids = tuple(sorted(int(value) for value in positive_train["tmdbId"].unique()))
    catalog = set(candidate_ids)
    histories = {
        int(user_id): tuple(sorted(int(value) for value in group["tmdbId"]))
        for user_id, group in positive_train.groupby("userId")
    }
    cases = []
    for user_id, group in targets.groupby("userId", sort=True):
        favorites = histories.get(int(user_id), ())
        user_targets = tuple(sorted(int(value) for value in group["tmdbId"]))
        if favorites and catalog.intersection(user_targets):
            cases.append(ValidationCase(int(user_id), favorites, user_targets))
    return candidate_ids, cases


def _validate_k(k: int) -> None:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _ranking_ids(recommendations: list[dict], catalog: set[int], favorites: set[int], k: int) -> list[int]:
    """Fail on invalid predictions instead of repairing rankings before scoring."""
    if not isinstance(recommendations, list) or len(recommendations) > k:
        raise ValueError("Recommendations must be a list containing at most k results.")
    ranked = []
    for recommendation in recommendations:
        if not isinstance(recommendation, dict) or not {"tmdb_id", "score"} <= recommendation.keys():
            raise ValueError("Each recommendation needs tmdb_id and score.")
        identifier, score = recommendation["tmdb_id"], recommendation["score"]
        if isinstance(identifier, (bool, np.bool_)) or not isinstance(identifier, Integral) or identifier <= 0:
            raise ValueError("Recommended TMDB IDs must be positive integers.")
        if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real) or not np.isfinite(score):
            raise ValueError("Recommendation scores must be finite numbers.")
        identifier = int(identifier)
        if identifier not in catalog or identifier in favorites or identifier in ranked:
            raise ValueError("Recommendations must be unique catalog films excluding favorites.")
        ranked.append(identifier)
    return ranked


def evaluate_recommender(recommender, cases: list[ValidationCase], candidate_ids, k: int = 10) -> dict:
    """Macro-average metrics on one cohort, without dropping empty predictions.

    Main metrics use targets inside the shared catalog. The second set uses
    all targets for these same users, including films unavailable to either
    model. None represents an empty cohort, never a measured zero score.
    """
    _validate_k(k)
    catalog = set(candidate_ids)
    recall_key, ndcg_key = f"recall_at_{k}", f"ndcg_at_{k}"
    per_user = []
    for case in cases:
        favorites, targets = set(case.favorites), set(case.targets)
        known_targets = targets & catalog
        if not favorites or not favorites <= catalog or not known_targets or favorites & targets:
            raise ValueError("Cases need train favorites and distinct, eligible validation targets.")
        recommendations = [] if recommender is None else recommender.recommend(case.favorites, n=k)
        ranked = _ranking_ids(recommendations, catalog, favorites, k)
        per_user.append({
            "user_id": int(case.user_id),
            "favorite_count": len(favorites),
            "target_count": len(targets),
            "target_count_in_catalog": len(known_targets),
            "target_count_outside_catalog": len(targets - catalog),
            "recommended_tmdb_ids": ranked,
            f"hits_at_{k}": len(set(ranked) & known_targets),
            recall_key: recall_at_k(ranked, known_targets, k),
            ndcg_key: ndcg_at_k(ranked, known_targets, k),
            f"{recall_key}_all_targets": recall_at_k(ranked, targets, k),
            f"{ndcg_key}_all_targets": ndcg_at_k(ranked, targets, k),
        })

    def average(key):
        return fmean(row[key] for row in per_user) if per_user else None

    return {
        "users_evaluated": len(per_user),
        "metrics": {key: average(key) for key in (recall_key, ndcg_key)},
        "metrics_all_targets": {
            key: average(f"{key}_all_targets") for key in (recall_key, ndcg_key)
        },
        "empty_recommendation_users": sum(not row["recommended_tmdb_ids"] for row in per_user),
        "mean_recommendations": fmean(len(row["recommended_tmdb_ids"]) for row in per_user) if per_user else None,
        "per_user": per_user,
    }


def _genre_catalog(train: pd.DataFrame, movies: pd.DataFrame, mapper: IdMapper) -> pd.DataFrame:
    positive_train = train.loc[train["is_positive"]]
    associations = positive_train[["movieId", "tmdbId"]].drop_duplicates()
    if associations["movieId"].duplicated().any() or associations["tmdbId"].duplicated().any():
        raise ValueError("Train films must have one canonical MovieLens/TMDB association.")
    for row in associations.itertuples(index=False):
        if mapper.to_movie_id(row.tmdbId) != row.movieId:
            raise ValueError("Train must use the canonical MovieLens ID declared in links.")
    required = ("movieId", "title", "genres")
    if not isinstance(movies, pd.DataFrame) or any(
        list(movies.columns).count(column) != 1 for column in required
    ):
        raise ValueError("Movies must contain movieId, title and genres columns.")
    catalog = associations[["movieId"]].merge(
        movies.loc[:, required], on="movieId", how="left", validate="one_to_one", indicator=True,
    )
    if catalog["_merge"].ne("both").any():
        raise ValueError("Missing metadata for a canonical train movie; cannot shrink the shared catalog.")
    return catalog.loc[:, required].sort_values("movieId").reset_index(drop=True)


def compare_models(
    train: pd.DataFrame, validation: pd.DataFrame,
    movies: pd.DataFrame, links: pd.DataFrame, k: int = 10,
) -> dict:
    """Fit both references on train and score only the shared validation cohort."""
    _validate_k(k)
    candidate_ids, cases = build_validation_cases(train, validation)
    mapper = IdMapper(links)
    genre_catalog = _genre_catalog(train, movies, mapper)
    usable_genres = any(tokenize_genres(value) for value in genre_catalog["genres"])
    genre_model = (
        TMDBRecommender(ContentBasedRecommender(genre_catalog), mapper)
        if usable_genres else None
    )
    coverage = validation_coverage(train, validation)
    known_count = sum(len(set(case.targets) & set(candidate_ids)) for case in cases)
    outside_count = sum(len(set(case.targets) - set(candidate_ids)) for case in cases)
    coverage.update({
        "eligible_positive_targets": known_count,
        "eligible_positive_targets_outside_catalog": outside_count,
        "eligible_target_coverage": known_count / coverage["positive_targets"] if coverage["positive_targets"] else None,
        "users_with_history_without_catalog_targets": coverage["users_with_positive_history"] - len(cases),
    })
    popularity = evaluate_recommender(PopularityRecommender(train), cases, candidate_ids, k)
    genres = evaluate_recommender(genre_model, cases, candidate_ids, k)
    genres["catalog_has_usable_genres"] = bool(usable_genres)
    return {
        "protocol": {
            "split": "validation",
            "k": int(k),
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
            "ties": {"popularity": "ascending TMDB ID", "genres": "ascending canonical MovieLens ID"},
            "test_usage": "not used to fit, select or score either baseline",
        },
        "catalog": {"movies": len(candidate_ids)},
        "coverage": coverage,
        "models": {"popularity": popularity, "genres": genres},
    }


def build_report(data_dir: Path, config_path: Path, k: int = 10) -> dict:
    """Rebuild a deterministic report from raw CSV files and frozen cutoffs."""
    _validate_k(k)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"positive_threshold", "train_end", "validation_end"}:
        raise ValueError("Evaluation config must contain positive_threshold, train_end and validation_end.")
    tables = load_movielens(data_dir)
    interactions, preparation = prepare_interactions(
        tables["ratings"], tables["links"], config["positive_threshold"],
    )
    splits = temporal_split(interactions, config["train_end"], config["validation_end"])
    report = compare_models(splits["train"], splits["validation"], tables["movies"], tables["links"], k)
    report.update({
        "config": {
            "positive_threshold": float(config["positive_threshold"]),
            "train_end": utc_cutoff(config["train_end"]).isoformat(),
            "validation_end": utc_cutoff(config["validation_end"]).isoformat(),
        },
        "sources": {
            name: {"sha256": hashlib.sha256((data_dir / name).read_bytes()).hexdigest()}
            for name in ("ratings.csv", "links.csv", "movies.csv")
        },
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "pandas", "scikit-learn")},
        },
        "preparation": preparation,
        "splits": {name: split_summary(splits[name]) for name in ("train", "validation")},
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config/evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_comparison.json"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    report = build_report(args.data_dir, args.config, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validation users: {report['coverage']['eligible_users']}")
    for name, result in report["models"].items():
        print(f"{name}: {json.dumps(result['metrics'], allow_nan=False)}")
    print(f"Baseline comparison report: {args.output}")


if __name__ == "__main__":
    main()
