"""Prepare TMDB-linked interactions and split them in time."""

import pandas as pd

from src.data.id_mapper import IdMapper

INTERACTION_COLUMNS = [
    "userId",
    "movieId",
    "tmdbId",
    "rating",
    "timestamp",
    "is_positive",
]


def utc_cutoff(value: str | pd.Timestamp) -> pd.Timestamp:
    """Express a timezone-aware cutoff in UTC (pandas rejects naive dates)."""
    return pd.Timestamp(value).tz_convert("UTC")


def prepare_interactions(
    ratings: pd.DataFrame,
    links: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Link ratings to TMDB films and keep one event per user and film."""
    mapper = IdMapper(links)
    associations = []
    for source_id in links["movieId"]:
        tmdb_id = mapper.to_tmdb_id(source_id)
        if tmdb_id is not None:
            # A TMDB film takes its smallest MovieLens ID, even one without ratings.
            associations.append((source_id, mapper.to_movie_id(tmdb_id), tmdb_id))
    linked_movies = pd.DataFrame(
        associations, columns=["source_movieId", "movieId", "tmdbId"], dtype="int64"
    )
    source = ratings[["userId", "movieId", "rating", "timestamp"]].rename(
        columns={"movieId": "source_movieId"}
    )
    linked = source.merge(
        linked_movies, on="source_movieId", how="inner", validate="many_to_one"
    )
    # Keep the earliest event, then the smallest source movieId: a later or
    # higher rating never replaces it.
    prepared = (
        linked.sort_values(["timestamp", "source_movieId", "userId"])
        .drop_duplicates(["userId", "tmdbId"], keep="first")
        .sort_values(["timestamp", "userId", "movieId"])
        .reset_index(drop=True)
    )
    prepared["is_positive"] = prepared["rating"].ge(positive_threshold)
    prepared = prepared[INTERACTION_COLUMNS]
    summary = {
        "input_ratings": len(ratings),
        "ratings_without_tmdb": len(ratings) - len(linked),
        "ratings_with_tmdb": len(linked),
        "remapped_ratings": int(linked["source_movieId"].ne(linked["movieId"]).sum()),
        "duplicate_user_movies_removed": len(linked) - len(prepared),
        "prepared_interactions": len(prepared),
        "positive_interactions": int(prepared["is_positive"].sum()),
        "other_interactions": int((~prepared["is_positive"]).sum()),
        "users": int(prepared["userId"].nunique()),
        "movies": int(prepared["tmdbId"].nunique()),
        "ambiguous_tmdb_ids": len(mapper.ambiguous_mappings),
    }
    return prepared, summary


def temporal_split(
    interactions: pd.DataFrame,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Split by UTC cutoffs; a timestamp equal to a cutoff starts the next split."""
    train_cutoff, validation_cutoff = utc_cutoff(train_end), utc_cutoff(validation_end)
    # Reversed cutoffs would silently put the same interactions in train and test.
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be before validation_end.")
    times = pd.to_datetime(interactions["timestamp"], unit="s", utc=True)
    masks = {
        "train": times.lt(train_cutoff),
        "validation": times.ge(train_cutoff) & times.lt(validation_cutoff),
        "test": times.ge(validation_cutoff),
    }
    return {
        name: interactions.loc[mask].reset_index(drop=True)
        for name, mask in masks.items()
    }
