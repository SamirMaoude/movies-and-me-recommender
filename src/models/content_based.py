"""Genre-only TF-IDF baseline using MovieLens movieId identifiers."""

from collections.abc import Iterable
from functools import partial
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_genres(value: str | None, *, include_imax: bool = False) -> list[str]:
    """Split pipe-separated genres, preserving labels such as Sci-Fi and Film-Noir.

    Missing values and the MovieLens sentinel add no feature. IMAX is excluded
    by default because it describes a projection format. Repeated genres in a
    movie count once.
    """
    if value is None or pd.isna(value):
        return []
    if not isinstance(value, str):
        raise ValueError("Genres must be strings or missing values.")
    ignored = {"", "(no genres listed)"}
    if not include_imax:
        ignored.add("IMAX")
    return sorted({genre.strip() for genre in value.split("|")} - ignored)


class ContentBasedRecommender:
    """Fit TF-IDF genres and rank movies against an average favorite profile.

    Inputs and outputs use MovieLens movieId, not TMDB IDs. Item vectors are
    L2-normalized; recommendations use cosine similarity to their mean.
    Scores express genre similarity, not probabilities or predicted ratings.
    """

    def __init__(self, movies: pd.DataFrame, *, include_imax: bool = False):
        required_columns = ["movieId", "title", "genres"]
        if not isinstance(movies, pd.DataFrame) or not set(required_columns).issubset(
            movies.columns
        ):
            raise ValueError("Catalog must contain movieId, title and genres columns.")
        if movies.empty:
            raise ValueError("Catalog must contain at least one movie.")
        if not isinstance(include_imax, bool):
            raise ValueError("include_imax must be a boolean.")
        ids = movies["movieId"]
        if ids.isna().any() or any(
            isinstance(movie_id, (bool, np.bool_))
            or not isinstance(movie_id, Integral)
            or movie_id <= 0
            for movie_id in ids
        ):
            raise ValueError("Catalog movieId values must be positive integers.")
        if ids.duplicated().any():
            raise ValueError("Catalog movieId values must be unique.")

        self.movies = movies.loc[:, required_columns].copy(deep=True)
        self.movies["movieId"] = self.movies["movieId"].astype("int64")
        self.movies = self.movies.sort_values("movieId").reset_index(drop=True)
        self.include_imax = include_imax
        self._id_to_position = dict(zip(self.movies["movieId"], self.movies.index))
        genre_documents = self.movies["genres"].fillna("").tolist()
        analyzer = partial(tokenize_genres, include_imax=include_imax)
        if not any(analyzer(value) for value in genre_documents):
            raise ValueError("Catalog has no usable genres after filtering.")
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, norm="l2")
        self.feature_matrix = self.vectorizer.fit_transform(genre_documents).tocsr()
        self.feature_names = self.vectorizer.get_feature_names_out()

    def recommend(self, favorite_movie_ids: Iterable[int], n: int = 10) -> pd.DataFrame:
        """Return up to n distinct positive-similarity movies, excluding favorites.

        Favorites are deduplicated so repeats do not alter their weight. Empty
        favorites, unknown IDs, an all-zero profile and invalid inputs raise
        ValueError. If fewer than n candidates share genres, return only those
        candidates. Ties are resolved by ascending movieId for reproducibility.
        Only a profile-to-catalog score vector is computed, never an all-pairs
        movie similarity matrix.
        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("n must be a positive integer.")
        try:
            favorites = list(favorite_movie_ids)
        except TypeError as exc:
            raise ValueError(
                "favorite_movie_ids must be an iterable of positive integers."
            ) from exc
        if not favorites:
            raise ValueError("At least one favorite movieId is required.")
        if any(
            isinstance(movie_id, (bool, np.bool_))
            or not isinstance(movie_id, Integral)
            or movie_id <= 0
            for movie_id in favorites
        ):
            raise ValueError("Favorite movieId values must be positive integers.")
        favorites = sorted({int(movie_id) for movie_id in favorites})
        unknown = sorted(set(favorites) - self._id_to_position.keys())
        if unknown:
            raise ValueError(f"Unknown MovieLens movieId values: {unknown}")

        positions = [self._id_to_position[movie_id] for movie_id in favorites]
        profile = np.asarray(self.feature_matrix[positions].mean(axis=0))
        if not np.any(profile):
            raise ValueError("Favorites have no usable genres; cannot build a profile.")
        scores = np.clip(
            cosine_similarity(profile, self.feature_matrix).ravel(), 0.0, 1.0
        )
        candidates = self.movies.copy()
        candidates["score"] = scores
        candidates = candidates.loc[
            ~candidates["movieId"].isin(favorites) & candidates["score"].gt(0)
        ]
        return (
            candidates.sort_values(["score", "movieId"], ascending=[False, True])
            .head(int(n))
            .reset_index(drop=True)
        )
