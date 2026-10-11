"""Recommend movies by the number of users with positive training feedback."""

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value <= 0
    ):
        raise ValueError(f"{name} must contain positive integers.")
    return int(value)


class PopularityRecommender:
    """Count distinct positive users per TMDB movie in the supplied train split.

    The caller must pass training data only. Required columns are userId,
    tmdbId and is_positive; ratings and timestamps do not affect the score.
    Repeated positive user/movie pairs count once. Other interactions do not
    contribute to the ranking or exclude movies from recommendations.

    Scores are user counts, not probabilities. Ties use ascending TMDB ID.
    The ranking is stored independently from the original DataFrame.
    """

    def __init__(self, train_interactions: pd.DataFrame):
        required = ["userId", "tmdbId", "is_positive"]
        if not isinstance(train_interactions, pd.DataFrame) or any(
            list(train_interactions.columns).count(column) != 1 for column in required
        ):
            raise ValueError(
                "Training data must contain userId, tmdbId and is_positive columns."
            )
        frame = train_interactions.loc[:, required].copy(deep=True)
        for column in ("userId", "tmdbId"):
            for value in frame[column]:
                _positive_integer(value, column)
        if any(
            not isinstance(value, (bool, np.bool_)) for value in frame["is_positive"]
        ):
            raise ValueError(
                "is_positive must contain booleans without missing values."
            )

        positive = frame.loc[frame["is_positive"].astype(bool), ["userId", "tmdbId"]]
        ranking = (
            positive.drop_duplicates(["userId", "tmdbId"])
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
        self,
        favorite_tmdb_ids: Iterable[int],
        n: int = 10,
    ) -> list[dict[str, int | float]]:
        """Return up to n ranked movies, excluding only the supplied favorites.

        Empty favorites or valid IDs outside the training catalogue return
        the global ranking. A train split without positive feedback returns [].
        IDs and n must be positive integers. Scores are native floats holding
        the number of distinct positive users and can exceed one.
        """
        limit = _positive_integer(n, "n")
        if isinstance(favorite_tmdb_ids, (str, bytes)):
            raise ValueError(
                "favorite_tmdb_ids must be an iterable of positive integers."
            )
        try:
            favorites = list(favorite_tmdb_ids)
        except TypeError as exc:
            raise ValueError(
                "favorite_tmdb_ids must be an iterable of positive integers."
            ) from exc
        excluded = {
            _positive_integer(value, "favorite_tmdb_ids") for value in favorites
        }
        recommendations = []
        for tmdb_id, positive_users in self._ranking:
            if tmdb_id in excluded:
                continue
            recommendations.append({"tmdb_id": tmdb_id, "score": float(positive_users)})
            if len(recommendations) == limit:
                break
        return recommendations
