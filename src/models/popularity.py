"""Recommend the films that most users rated positively in train."""

from collections.abc import Iterable

import pandas as pd


class PopularityRecommender:
    """Rank films by distinct positive train users; ties go to the smaller TMDB ID."""

    def __init__(self, train_interactions: pd.DataFrame):
        positive = train_interactions.loc[
            train_interactions["is_positive"], ["userId", "tmdbId"]
        ]
        ranking = (
            positive.drop_duplicates()
            .groupby("tmdbId")
            .size()
            .rename("score")
            .reset_index()
            .sort_values(["score", "tmdbId"], ascending=[False, True])
        )
        self._ranking = tuple(
            (int(row.tmdbId), int(row.score)) for row in ranking.itertuples(index=False)
        )

    def recommend(
        self, favorite_tmdb_ids: Iterable[int], n: int = 10
    ) -> list[dict[str, int | float]]:
        """Return the n most popular films, excluding the favorites."""
        favorites = set(favorite_tmdb_ids)
        recommendations = [
            {"tmdb_id": tmdb_id, "score": float(users)}
            for tmdb_id, users in self._ranking
            if tmdb_id not in favorites
        ]
        return recommendations[:n]
