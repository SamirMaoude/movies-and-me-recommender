"""Recommend TMDB IDs with the existing MovieLens genre model."""

from collections.abc import Iterable
from numbers import Integral

import numpy as np

from src.data.id_mapper import IdMapper
from src.models.content_based import ContentBasedRecommender


class TMDBRecommender:
    """Convert favorite TMDB IDs, rank movies, and return TMDB IDs and scores.

    Unknown favorites are ignored. Empty favorites or favorites without usable
    genres return an empty list. A shared TMDB ID uses the smallest MovieLens
    ID available in the model; all its other entries are excluded from results.
    """

    def __init__(self, recommender: ContentBasedRecommender, id_mapper: IdMapper):
        self.recommender = recommender
        self.id_mapper = id_mapper
        self._catalog_ids = set(recommender.movies["movieId"])

    def recommend(
        self,
        favorite_tmdb_ids: Iterable[int],
        n: int = 10,
    ) -> list[dict[str, int | float]]:
        """Return up to n unique TMDB IDs with positive similarity scores.

        IDs must be positive integers, and n must be a positive integer.
        Invalid input raises ValueError. Valid IDs missing from links or from
        the model are skipped. Missing output IDs and duplicate TMDB IDs are
        filtered before the limit, keeping the highest score for each film.
        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("n must be a positive integer.")
        try:
            favorites = list(favorite_tmdb_ids)
        except TypeError as exc:
            raise ValueError(
                "favorite_tmdb_ids must be an iterable of positive integers."
            ) from exc

        favorite_aliases = {}
        for tmdb_id in favorites:
            aliases = self.id_mapper.movie_ids_for_tmdb(tmdb_id)
            favorite_aliases[int(tmdb_id)] = aliases

        favorite_movie_ids = set()
        for aliases in favorite_aliases.values():
            for movie_id in aliases:
                if movie_id in self._catalog_ids:
                    favorite_movie_ids.add(movie_id)
                    break
        if not favorite_movie_ids:
            return []

        positions = self.recommender.movies.index[
            self.recommender.movies["movieId"].isin(favorite_movie_ids)
        ].to_list()
        if self.recommender.feature_matrix[positions].nnz == 0:
            return []

        # Get the whole ranking so unmapped movies do not consume the limit.
        ranked = self.recommender.recommend(
            sorted(favorite_movie_ids),
            n=len(self.recommender.movies),
        )
        seen_tmdb_ids = set(favorite_aliases)
        recommendations = []
        for row in ranked.itertuples(index=False):
            tmdb_id = self.id_mapper.to_tmdb_id(row.movieId)
            if tmdb_id is None or tmdb_id in seen_tmdb_ids:
                continue
            recommendations.append({"tmdb_id": int(tmdb_id), "score": float(row.score)})
            seen_tmdb_ids.add(tmdb_id)
            if len(recommendations) == n:
                break
        return recommendations
