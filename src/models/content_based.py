"""Genre-only TF-IDF recommendations over MovieLens movie IDs."""

from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_genres(value: str | None, *, include_imax: bool = False) -> list[str]:
    """Split pipe-separated genres; IMAX, a projection format, is ignored by default."""
    if pd.isna(value):
        return []
    ignored = {"", "(no genres listed)"}
    if not include_imax:
        ignored.add("IMAX")
    return sorted({genre.strip() for genre in value.split("|")} - ignored)


class ContentBasedRecommender:
    """Rank movies by cosine similarity to the mean TF-IDF genre vector of favorites."""

    def __init__(self, movies: pd.DataFrame, *, include_imax: bool = False):
        self.movies = (
            movies[["movieId", "title", "genres"]]
            .sort_values("movieId")
            .reset_index(drop=True)
        )
        self.include_imax = include_imax
        self._id_to_position = dict(zip(self.movies["movieId"], self.movies.index))
        self.vectorizer = TfidfVectorizer(
            analyzer=partial(tokenize_genres, include_imax=include_imax), norm="l2"
        )
        genres = self.movies["genres"].fillna("").tolist()
        self.feature_matrix = self.vectorizer.fit_transform(genres).tocsr()
        self.feature_names = self.vectorizer.get_feature_names_out()

    def recommend(self, favorite_movie_ids: Iterable[int], n: int = 10) -> pd.DataFrame:
        """Return up to n movies that share genres with the favorites, best first."""
        favorites = sorted(set(favorite_movie_ids))
        if not favorites:
            raise ValueError("At least one favorite movieId is required.")
        unknown = [
            movie_id for movie_id in favorites if movie_id not in self._id_to_position
        ]
        if unknown:
            raise ValueError(f"Unknown MovieLens movieId values: {unknown}")
        positions = [self._id_to_position[movie_id] for movie_id in favorites]
        profile = np.asarray(self.feature_matrix[positions].mean(axis=0))
        if not profile.any():
            raise ValueError("Favorites have no usable genres; cannot build a profile.")
        scores = np.clip(
            cosine_similarity(profile, self.feature_matrix).ravel(), 0.0, 1.0
        )
        candidates = self.movies.assign(score=scores)
        candidates = candidates[
            ~candidates["movieId"].isin(favorites) & candidates["score"].gt(0)
        ]
        # Equal scores are ordered by movieId so that rankings are reproducible.
        return (
            candidates.sort_values(["score", "movieId"], ascending=[False, True])
            .head(n)
            .reset_index(drop=True)
        )
