"""Recommend TMDB IDs with the MovieLens genre model."""

from collections.abc import Iterable

from src.data.id_mapper import IdMapper
from src.models.content_based import ContentBasedRecommender


class TMDBRecommender:
    """Translate TMDB favorites to MovieLens IDs, rank by genres and translate back."""

    def __init__(self, recommender: ContentBasedRecommender, id_mapper: IdMapper):
        self.recommender = recommender
        self.id_mapper = id_mapper
        self._catalog_ids = set(recommender.movies["movieId"])

    def recommend(
        self, favorite_tmdb_ids: Iterable[int], n: int = 10
    ) -> list[dict[str, int | float]]:
        """Return up to n TMDB films; unknown favorites are ignored."""
        favorites = set(favorite_tmdb_ids)
        # A film with several MovieLens IDs uses the smallest one in the catalog.
        favorite_movie_ids = set()
        for tmdb_id in favorites:
            for movie_id in self.id_mapper.movie_ids_for_tmdb(tmdb_id):
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

        # Rank the whole catalog so that movies without a TMDB ID do not use up
        # the limit; a TMDB film shared by several movies keeps its best score.
        ranked = self.recommender.recommend(
            sorted(favorite_movie_ids), n=len(self.recommender.movies)
        )
        seen = set(favorites)
        recommendations = []
        for row in ranked.itertuples(index=False):
            tmdb_id = self.id_mapper.to_tmdb_id(row.movieId)
            if tmdb_id is None or tmdb_id in seen:
                continue
            seen.add(tmdb_id)
            recommendations.append({"tmdb_id": tmdb_id, "score": float(row.score)})
            if len(recommendations) == n:
                break
        return recommendations
