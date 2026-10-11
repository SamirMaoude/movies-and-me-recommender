"""Translate between MovieLens and TMDB IDs using links.csv."""

import pandas as pd


class IdMapper:
    def __init__(self, links: pd.DataFrame):
        self._movie_to_tmdb: dict[int, int] = {}
        tmdb_to_movies: dict[int, list[int]] = {}
        rows = links[["movieId", "tmdbId"]].itertuples(index=False, name=None)
        for movie_id, tmdb_id in rows:
            if pd.isna(tmdb_id):
                continue
            self._movie_to_tmdb[int(movie_id)] = int(tmdb_id)
            tmdb_to_movies.setdefault(int(tmdb_id), []).append(int(movie_id))
        self._tmdb_to_movies = {
            tmdb_id: tuple(sorted(movie_ids))
            for tmdb_id, movie_ids in tmdb_to_movies.items()
        }

    def to_tmdb_id(self, movie_id: int) -> int | None:
        return self._movie_to_tmdb.get(movie_id)

    def to_movie_id(self, tmdb_id: int) -> int | None:
        """Return the smallest movieId linked to the TMDB ID, or None."""
        movie_ids = self.movie_ids_for_tmdb(tmdb_id)
        return movie_ids[0] if movie_ids else None

    def movie_ids_for_tmdb(self, tmdb_id: int) -> tuple[int, ...]:
        return self._tmdb_to_movies.get(tmdb_id, ())

    @property
    def ambiguous_mappings(self) -> dict[int, tuple[int, ...]]:
        """TMDB IDs shared by several MovieLens movies."""
        return {
            tmdb_id: movie_ids
            for tmdb_id, movie_ids in self._tmdb_to_movies.items()
            if len(movie_ids) > 1
        }
