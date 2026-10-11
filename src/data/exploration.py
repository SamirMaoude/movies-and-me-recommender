"""Load MovieLens tables and summarize ratings, genres and TMDB coverage."""

from pathlib import Path

import pandas as pd


SCHEMAS = {
    "movies": {"movieId": "int64", "title": "string", "genres": "string"},
    "ratings": {
        "userId": "int64", "movieId": "int64", "rating": "float64",
        "timestamp": "int64",
    },
    "links": {"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"},
    "tags": {
        "userId": "int64", "movieId": "int64", "tag": "string",
        "timestamp": "int64",
    },
}

TABLE_KEYS = {
    "movies": ["movieId"],
    "ratings": ["userId", "movieId"],
    "links": ["movieId"],
    # Multiple distinct tags for the same user/movie pair are legitimate.
    "tags": ["userId", "movieId", "tag", "timestamp"],
}


def load_movielens(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the four CSVs, keeping leading IMDb zeros and missing TMDB IDs.

    Raise an error if a file is missing or its columns differ from the schema.
    """
    data_dir = Path(data_dir)
    missing = [f"{name}.csv" for name in SCHEMAS if not (data_dir / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens files in {data_dir}: {', '.join(missing)}")

    tables = {}
    for name, schema in SCHEMAS.items():
        path = data_dir / f"{name}.csv"
        frame = pd.read_csv(path, dtype=schema, encoding="utf-8")
        if set(frame.columns) != set(schema):
            raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
        tables[name] = frame
    return tables


def data_quality_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count missing cells, duplicate rows and extra rows sharing a logical key."""
    rows = []
    for name, frame in tables.items():
        rows.append({
            "dataset": name,
            "num_rows": len(frame),
            "num_columns": len(frame.columns),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_keys": int(frame.duplicated(subset=TABLE_KEYS[name]).sum()),
        })
    return pd.DataFrame(rows).set_index("dataset")


def interaction_tables(
    movies: pd.DataFrame, ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return counts and mean scores per user and per catalog movie.

    Users come from ratings. Every catalog movie is retained, including those
    with zero ratings; their mean score stays missing rather than becoming 0.
    """
    aggregates = {"num_ratings": ("rating", "size"), "avg_rating": ("rating", "mean")}
    per_user = ratings.groupby("userId").agg(**aggregates)
    per_movie = ratings.groupby("movieId").agg(**aggregates).reindex(movies["movieId"])
    per_movie["num_ratings"] = per_movie["num_ratings"].fillna(0).astype("int64")
    return per_user, per_movie


def genre_counts(movies: pd.DataFrame) -> pd.Series:
    """Count movie memberships per real genre, excluding the missing sentinel.

    A multi-genre movie contributes once to each genre. The total can therefore
    exceed the size of the catalog. Repeated labels in one movie count once.
    """
    memberships = movies[["movieId", "genres"]].copy()
    memberships["genres"] = memberships["genres"].str.split("|")
    memberships = memberships.explode("genres")
    memberships["genres"] = memberships["genres"].str.strip()
    memberships = memberships.dropna(subset=["genres"])
    memberships = memberships.loc[~memberships["genres"].isin(["", "(no genres listed)"])]
    return (
        memberships.drop_duplicates(["movieId", "genres"])["genres"]
        .value_counts().rename("num_movies")
    )


def summarize_dataset(tables: dict[str, pd.DataFrame]) -> dict[str, int | float]:
    """Summarize ratings and TMDB coverage across the full movie catalog.

    TMDB coverage means an ID is present in links.csv, not that its current
    existence or correctness has been verified against the TMDB service.
    """
    movies, ratings, links = tables["movies"], tables["ratings"], tables["links"]
    num_movies = len(movies)
    num_users = ratings["userId"].nunique()
    num_ratings = len(ratings)
    if num_movies == 0 or num_ratings == 0 or num_users == 0:
        raise ValueError("The exploration summary requires nonempty movies and ratings.")
    movies_with_links = movies.merge(links, on="movieId", how="left", validate="one_to_one")
    tmdb_coverage = movies_with_links["tmdbId"].notna().mean()
    return {
        "num_movies": num_movies,
        "num_users": int(num_users),
        "num_ratings": num_ratings,
        "avg_ratings_per_user": num_ratings / num_users,
        "avg_ratings_per_movie": num_ratings / num_movies,
        "tmdb_coverage": float(tmdb_coverage),
    }
