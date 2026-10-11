"""Prepare TMDB-linked interactions without using future ratings as history."""

from numbers import Real

import numpy as np
import pandas as pd

from src.data.id_mapper import IdMapper


INTERACTION_COLUMNS = [
    "userId", "movieId", "tmdbId", "rating", "timestamp", "is_positive",
]
MAX_INT64 = np.iinfo(np.int64).max
# Last UTC second representable with a four-digit year (9999-12-31).
MAX_UNIX_TIMESTAMP = 253402300799


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Interactions must be a pandas DataFrame.")
    if any(list(frame.columns).count(column) != 1 for column in columns):
        raise ValueError(f"Expected one of each column: {', '.join(columns)}.")


def _require_integers(values: pd.Series, minimum: int) -> None:
    if (
        not pd.api.types.is_integer_dtype(values.dtype)
        or values.isna().any()
        or values.lt(minimum).any()
        or values.gt(MAX_INT64).any()
    ):
        raise ValueError(f"{values.name} must contain integers between {minimum} and {MAX_INT64}.")


def _event_times(timestamps: pd.Series) -> pd.Series:
    _require_integers(timestamps, 0)
    if timestamps.gt(MAX_UNIX_TIMESTAMP).any():
        raise ValueError("timestamp exceeds the supported UTC calendar (year 9999).")
    try:
        return pd.to_datetime(timestamps, unit="s", utc=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError("timestamp must contain valid Unix seconds.") from exc


def utc_cutoff(value: str | pd.Timestamp) -> pd.Timestamp:
    """Read an explicit timezone-aware cutoff and express it in UTC."""
    if not isinstance(value, (str, pd.Timestamp)):
        raise ValueError("Cutoffs must be timezone-aware dates.")
    try:
        cutoff = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Cutoffs must be timezone-aware dates.") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("Cutoffs must include a timezone, for example 2016-01-01T00:00:00Z.")
    return cutoff.tz_convert("UTC")


def prepare_interactions(
    ratings: pd.DataFrame,
    links: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Link ratings to TMDB, mark positives and keep the first user/film event.

    Lower ratings remain available for the data audit. Only positive feedback
    simulates favorites; lower ratings do not add recommendation exclusions.
    Missing TMDB associations are counted and excluded. Each TMDB film uses
    the smallest MovieLens ID in links, even if that ID has no rating of its own.

    Repeated user/TMDB pairs keep the earliest timestamp, then the smallest
    original MovieLens ID. This rule never prefers a higher or later rating.
    Conflicting ratings for the same user/source movie/timestamp are rejected.
    Inputs are left unchanged; timestamps remain integer Unix seconds.
    """
    _require_columns(ratings, ["userId", "movieId", "rating", "timestamp"])
    if (
        isinstance(positive_threshold, (bool, np.bool_))
        or not isinstance(positive_threshold, Real)
        or not np.isfinite(positive_threshold)
        or not 0.5 <= positive_threshold <= 5.0
    ):
        raise ValueError("positive_threshold must be a finite number between 0.5 and 5.")
    for column in ("userId", "movieId"):
        _require_integers(ratings[column], 1)
    _event_times(ratings["timestamp"])
    scores = ratings["rating"]
    if (
        not (pd.api.types.is_integer_dtype(scores.dtype) or pd.api.types.is_float_dtype(scores.dtype))
        or scores.isna().any()
    ):
        raise ValueError("rating must contain numbers from 0.5 to 5 in steps of 0.5.")
    values = scores.to_numpy(dtype=float)
    if not (
        np.isfinite(values).all()
        and ((values >= 0.5) & (values <= 5.0)).all()
        and (values * 2 == np.floor(values * 2)).all()
    ):
        raise ValueError("rating must contain numbers from 0.5 to 5 in steps of 0.5.")
    conflicts = ratings.groupby(["userId", "movieId", "timestamp"])["rating"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Conflicting ratings for the same user, movie and timestamp.")

    mapper = IdMapper(links)
    associations = []
    for source_movie_id in links["movieId"]:
        source_movie_id = int(source_movie_id)
        if source_movie_id > MAX_INT64:
            raise ValueError("links movieId exceeds the supported integer range.")
        tmdb_id = mapper.to_tmdb_id(source_movie_id)
        if tmdb_id is not None:
            if tmdb_id > MAX_INT64:
                raise ValueError("links tmdbId exceeds the supported integer range.")
            associations.append((source_movie_id, mapper.to_movie_id(tmdb_id), tmdb_id))
    linked_movies = pd.DataFrame(
        associations, columns=["source_movieId", "movieId", "tmdbId"],
    ).astype("int64")
    source = ratings.loc[:, ["userId", "movieId", "rating", "timestamp"]].rename(
        columns={"movieId": "source_movieId"},
    )
    linked = source.merge(linked_movies, on="source_movieId", how="inner", validate="many_to_one")
    remapped = int(linked["source_movieId"].ne(linked["movieId"]).sum())
    prepared = (
        linked.sort_values(["timestamp", "source_movieId", "userId"])
        .drop_duplicates(["userId", "tmdbId"], keep="first")
        .sort_values(["timestamp", "userId", "movieId"])
        .reset_index(drop=True)
    )
    prepared["is_positive"] = prepared["rating"].ge(positive_threshold)
    prepared = prepared.loc[:, INTERACTION_COLUMNS].copy()
    for column in ("userId", "movieId", "tmdbId", "timestamp"):
        prepared[column] = prepared[column].astype("int64")
    prepared["rating"] = prepared["rating"].astype("float64")
    summary = {
        "input_ratings": len(ratings),
        "ratings_without_tmdb": len(ratings) - len(linked),
        "ratings_with_tmdb": len(linked),
        "remapped_ratings": remapped,
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
    """Apply global UTC cutoffs, keeping boundary timestamps in the next split.

    Train is strictly before train_end; validation includes train_end but is
    strictly before validation_end; test includes validation_end and later.
    Empty splits are allowed and no users or films are filtered by future
    activity. A repeated user/TMDB pair must be resolved before splitting.
    """
    _require_columns(interactions, ["userId", "tmdbId", "timestamp"])
    for column in ("userId", "tmdbId"):
        _require_integers(interactions[column], 1)
    if interactions.duplicated(["userId", "tmdbId"]).any():
        raise ValueError("Resolve repeated user/TMDB interactions before splitting.")
    times = _event_times(interactions["timestamp"])
    train_cutoff, validation_cutoff = utc_cutoff(train_end), utc_cutoff(validation_end)
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be strictly before validation_end.")
    masks = {
        "train": times.lt(train_cutoff),
        "validation": times.ge(train_cutoff) & times.lt(validation_cutoff),
        "test": times.ge(validation_cutoff),
    }
    return {
        name: interactions.loc[mask].copy().reset_index(drop=True)
        for name, mask in masks.items()
    }
