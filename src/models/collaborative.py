"""Train implicit ALS on binary favorites and recommend from TMDB favorites."""

import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz, save_npz
from threadpoolctl import threadpool_limits

from src.data.id_mapper import IdMapper
from src.files import read_json, sha256, write_json

DATA_FILES = ("model.npz", "interactions.npz", "catalog.json")


@dataclass(frozen=True)
class ALSConfig:
    factors: int = 32
    regularization: float = 0.1
    alpha: float = 20.0
    iterations: int = 30
    random_state: int = 42
    num_threads: int = 1


@dataclass(frozen=True)
class TrainingData:
    """Binary user-by-film matrix and the external IDs of its rows and columns."""

    matrix: csr_matrix
    user_ids: tuple[int, ...]
    tmdb_ids: tuple[int, ...]
    movie_ids: tuple[int, ...]


def build_interaction_matrix(train: pd.DataFrame, links: pd.DataFrame) -> TrainingData:
    """Mark each positive user/film pair once; lower ratings add no signal."""
    positives = train.loc[train["is_positive"], ["userId", "tmdbId"]]
    pairs = sorted(
        {(int(user), int(tmdb)) for user, tmdb in positives.itertuples(index=False)}
    )
    if not pairs:
        raise ValueError("ALS training needs at least one positive interaction.")
    user_ids = tuple(sorted({user for user, _ in pairs}))
    tmdb_ids = tuple(sorted({tmdb for _, tmdb in pairs}))
    rows = {user: row for row, user in enumerate(user_ids)}
    columns = {tmdb: column for column, tmdb in enumerate(tmdb_ids)}
    # Values stay binary: implicit multiplies them by alpha itself, both in
    # fit() and when recalculate_user() builds a request profile.
    matrix = csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float32),
            ([rows[user] for user, _ in pairs], [columns[tmdb] for _, tmdb in pairs]),
        ),
        shape=(len(user_ids), len(tmdb_ids)),
        dtype=np.float32,
    )
    mapper = IdMapper(links)
    movie_ids = tuple(mapper.to_movie_id(tmdb) for tmdb in tmdb_ids)
    return TrainingData(matrix, user_ids, tmdb_ids, movie_ids)


def _popularity(data: TrainingData) -> tuple[tuple[int, int], ...]:
    """Training films and their positive users, ties going to the smaller TMDB ID."""
    counts = data.matrix.getnnz(axis=0)
    return tuple(
        sorted(
            ((tmdb, int(count)) for tmdb, count in zip(data.tmdb_ids, counts)),
            key=lambda pair: (-pair[1], pair[0]),
        )
    )


def _dimensions(data: TrainingData, config: ALSConfig) -> dict[str, int]:
    return {
        "users": len(data.user_ids),
        "items": len(data.tmdb_ids),
        "factors": config.factors,
        "interactions": data.matrix.nnz,
    }


class ImplicitRecommender:
    """ALS recommender that recalculates a profile from each request's favorites."""

    def __init__(
        self,
        model: AlternatingLeastSquares,
        training_data: TrainingData,
        config: ALSConfig,
        metadata: dict | None = None,
    ):
        self.model = model
        self.training_data = training_data
        self.config = config
        self.metadata = {} if metadata is None else metadata
        self._item_indices = {
            tmdb: index for index, tmdb in enumerate(training_data.tmdb_ids)
        }
        self._ranking = _popularity(training_data)

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        links: pd.DataFrame,
        config: ALSConfig | None = None,
    ) -> "ImplicitRecommender":
        config = ALSConfig() if config is None else config
        data = build_interaction_matrix(train, links)
        # implicit warns that a multithreaded BLAS makes ALS far slower, so every
        # call into implicit runs with a single BLAS thread.
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(**asdict(config), dtype=np.float32)
            model.fit(data.matrix, show_progress=False)
        return cls(model, data, config)

    def recommend(
        self, favorite_tmdb_ids: Iterable[int], n: int = 10
    ) -> list[dict[str, int | float]]:
        return self.recommend_with_method(favorite_tmdb_ids, n)["recommendations"]

    def count_known(self, favorite_tmdb_ids: Iterable[int]) -> int:
        """Count the distinct favorites that belong to the training catalog."""
        return sum(tmdb in self._item_indices for tmdb in set(favorite_tmdb_ids))

    def recommend_with_method(
        self, favorite_tmdb_ids: Iterable[int], n: int = 10
    ) -> dict:
        """Rank with ALS from the known favorites, or by popularity if none is known."""
        favorites = set(favorite_tmdb_ids)
        known = sorted(
            self._item_indices[tmdb] for tmdb in favorites if tmdb in self._item_indices
        )
        if not known:
            recommendations = [
                {"tmdb_id": tmdb, "score": float(count)}
                for tmdb, count in self._ranking
                if tmdb not in favorites
            ]
            return {"method": "popularity", "recommendations": recommendations[:n]}

        item_count = len(self.training_data.tmdb_ids)
        profile = csr_matrix(
            (np.ones(len(known), dtype=np.float32), ([0] * len(known), known)),
            shape=(1, item_count),
            dtype=np.float32,
        )
        # recalculate_user solves a user factor from the favorites alone: no
        # stored training user is involved, and the model is left unchanged.
        with threadpool_limits(limits=1, user_api="blas"):
            indices, scores = self.model.recommend(
                0,
                profile,
                N=item_count,
                filter_already_liked_items=False,
                recalculate_user=True,
            )
        # Equal scores go to the smaller TMDB ID so that rankings are reproducible.
        ranked = sorted(
            (
                (self.training_data.tmdb_ids[index], float(score))
                for index, score in zip(indices, scores)
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        recommendations = [
            {"tmdb_id": tmdb, "score": score}
            for tmdb, score in ranked
            if tmdb not in favorites
        ]
        return {"method": "als", "recommendations": recommendations[:n]}

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        """Write the bundle into a new directory; a failed save leaves nothing."""
        target = Path(directory)
        if target.exists():
            raise FileExistsError(f"Artifact directory already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}-", dir=target.parent
        ) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            self.model.save(bundle / "model.npz")
            save_npz(
                bundle / "interactions.npz", self.training_data.matrix, compressed=True
            )
            write_json(
                bundle / "catalog.json",
                {
                    "user_ids": self.training_data.user_ids,
                    "tmdb_ids": self.training_data.tmdb_ids,
                    "movie_ids": self.training_data.movie_ids,
                    "popularity": self._ranking,
                },
            )
            write_json(
                bundle / "manifest.json",
                {
                    "format_version": 1,
                    "implicit_version": implicit.__version__,
                    "config": asdict(self.config),
                    "dimensions": _dimensions(self.training_data, self.config),
                    "sha256": {name: sha256(bundle / name) for name in DATA_FILES},
                    "metadata": self.metadata if metadata is None else metadata,
                },
            )
            bundle.rename(target)

    @classmethod
    def load(cls, directory: str | Path) -> "ImplicitRecommender":
        """Load a bundle after checking its format, checksums and dimensions."""
        bundle = Path(directory)
        try:
            manifest = read_json(bundle / "manifest.json")
            if (
                manifest["format_version"] != 1
                or manifest["implicit_version"] != implicit.__version__
            ):
                raise ValueError("Unsupported artifact format or implicit version.")
            for name in DATA_FILES:
                if sha256(bundle / name) != manifest["sha256"][name]:
                    raise ValueError(f"Artifact checksum mismatch: {name}.")
            config = ALSConfig(**manifest["config"])
            catalog = read_json(bundle / "catalog.json")
            data = TrainingData(
                load_npz(bundle / "interactions.npz"),
                tuple(catalog["user_ids"]),
                tuple(catalog["tmdb_ids"]),
                tuple(catalog["movie_ids"]),
            )
            with threadpool_limits(limits=1, user_api="blas"):
                model = AlternatingLeastSquares.load(bundle / "model.npz")
            if manifest["dimensions"] != _dimensions(data, config):
                raise ValueError("Artifact dimensions do not match its contents.")
            for name, rows in (
                ("user_factors", len(data.user_ids)),
                ("item_factors", len(data.tmdb_ids)),
            ):
                if getattr(model, name).shape != (rows, config.factors):
                    raise ValueError(f"Artifact {name} do not match its dimensions.")
            return cls(model, data, config, manifest["metadata"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ALS artifact bundle: {exc}") from exc
