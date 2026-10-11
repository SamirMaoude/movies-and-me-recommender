"""Translate MovieLens and TMDB IDs using the supplied links table."""

from types import MappingProxyType

import numpy as np
import pandas as pd


def _table_id(value: object, column: str) -> int:
    """Read a positive integer ID, including integral floats from CSV data."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must contain positive integer IDs.")
    if isinstance(value, (int, np.integer)):
        identifier = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"{column} must contain positive integer IDs.")
        identifier = int(value)
    else:
        raise ValueError(f"{column} must contain positive integer IDs.")
    if identifier <= 0:
        raise ValueError(f"{column} must contain positive integer IDs.")
    return identifier


def _lookup_id(value: object, name: str) -> int:
    """Require integer scalars for public lookups without coercing input."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer ID.")
    identifier = int(value)
    if identifier <= 0:
        raise ValueError(f"{name} must be a positive integer ID.")
    return identifier


class IdMapper:
    """Keep a snapshot of the ID associations declared in a links DataFrame.

    The table must have unique positive movie IDs. Missing TMDB IDs are
    allowed. Multiple movies can share one TMDB ID; reverse lookup then uses
    the smallest movie ID and exposes every candidate separately. These
    associations come from the table and are not verified against TMDB.
    """

    def __init__(self, links: pd.DataFrame):
        if not isinstance(links, pd.DataFrame):
            raise ValueError("links must be a pandas DataFrame.")
        for column in ("movieId", "tmdbId"):
            if list(links.columns).count(column) != 1:
                raise ValueError(f"links must contain exactly one {column} column.")

        movie_to_tmdb: dict[int, int] = {}
        tmdb_to_movies: dict[int, list[int]] = {}
        seen_movies: set[int] = set()
        rows = links.loc[:, ["movieId", "tmdbId"]].itertuples(index=False, name=None)
        for movie_value, tmdb_value in rows:
            movie_id = _table_id(movie_value, "movieId")
            if movie_id in seen_movies:
                raise ValueError(f"movieId must be unique; duplicate ID: {movie_id}.")
            seen_movies.add(movie_id)

            if pd.api.types.is_scalar(tmdb_value) and pd.isna(tmdb_value):
                continue
            tmdb_id = _table_id(tmdb_value, "tmdbId")
            movie_to_tmdb[movie_id] = tmdb_id
            tmdb_to_movies.setdefault(tmdb_id, []).append(movie_id)

        self._movie_to_tmdb = MappingProxyType(movie_to_tmdb)
        self._tmdb_to_movies = MappingProxyType(
            {
                tmdb_id: tuple(sorted(movie_ids))
                for tmdb_id, movie_ids in tmdb_to_movies.items()
            }
        )

    def to_tmdb_id(self, movie_id: int) -> int | None:
        """Return the linked TMDB ID, or None for an unknown or unlinked movie.

        Invalid IDs raise ValueError. Only positive integer scalars are accepted.
        """
        return self._movie_to_tmdb.get(_lookup_id(movie_id, "movie_id"))

    def to_movie_id(self, tmdb_id: int) -> int | None:
        """Return the smallest linked movie ID, or None for an unknown TMDB ID.

        Invalid IDs raise ValueError. Only positive integer scalars are accepted.
        """
        movie_ids = self.movie_ids_for_tmdb(tmdb_id)
        return movie_ids[0] if movie_ids else None

    def movie_ids_for_tmdb(self, tmdb_id: int) -> tuple[int, ...]:
        """Return all linked movie IDs in ascending order, or an empty tuple.

        Invalid IDs raise ValueError. Only positive integer scalars are accepted.
        """
        return self._tmdb_to_movies.get(_lookup_id(tmdb_id, "tmdb_id"), ())

    @property
    def ambiguous_mappings(self) -> dict[int, tuple[int, ...]]:
        """Return a fresh dictionary of TMDB IDs linked to multiple movies."""
        return {
            tmdb_id: movie_ids
            for tmdb_id, movie_ids in self._tmdb_to_movies.items()
            if len(movie_ids) > 1
        }
