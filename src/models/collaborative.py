"""Train CPU implicit ALS and recommend from a request's TMDB favorites."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import tempfile

import implicit
from implicit.cpu.als import AlternatingLeastSquares
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from threadpoolctl import threadpool_limits

from src.data.id_mapper import IdMapper


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return int(value)


@dataclass(frozen=True)
class ALSConfig:
    """Parameters used for both training and recalculating a request profile."""

    factors: int = 32
    regularization: float = 0.1
    alpha: float = 20.0
    iterations: int = 30
    random_state: int = 42
    num_threads: int = 1

    def __post_init__(self) -> None:
        for name in ("factors", "iterations", "num_threads", "random_state"):
            value = _integer(getattr(self, name), name, 0 if name == "random_state" else 1)
            if name == "random_state" and value > 2**32 - 1:
                raise ValueError("random_state must be at most 2**32 - 1.")
            object.__setattr__(self, name, value)
        for name in ("regularization", "alpha"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class TrainingData:
    """Binary user-by-film matrix, with its ordered external IDs."""

    matrix: csr_matrix
    user_ids: tuple[int, ...]
    tmdb_ids: tuple[int, ...]
    movie_ids: tuple[int, ...]


def build_interaction_matrix(train: pd.DataFrame, links: pd.DataFrame) -> TrainingData:
    """Use only positive training interactions, counting each user/film once.

    The caller supplies the training split. A TMDB film's canonical MovieLens
    ID is the smallest associated ID in links, including unrated aliases.
    Lower ratings do not contribute confidence or create negative feedback.
    """
    required = ["userId", "movieId", "tmdbId", "is_positive"]
    if not isinstance(train, pd.DataFrame) or any(
        list(train.columns).count(column) != 1 for column in required
    ):
        raise ValueError(f"Training data must contain one of each column: {', '.join(required)}.")
    mapper = IdMapper(links)
    pairs: set[tuple[int, int]] = set()
    for user, movie, tmdb, positive in train.loc[:, required].itertuples(index=False, name=None):
        user = _integer(user, "userId")
        movie = _integer(movie, "movieId")
        tmdb = _integer(tmdb, "tmdbId")
        if not isinstance(positive, (bool, np.bool_)):
            raise ValueError("is_positive must contain booleans without missing values.")
        if mapper.to_tmdb_id(movie) != tmdb:
            raise ValueError("Training movieId/tmdbId associations must agree with links.")
        if positive:
            pairs.add((user, tmdb))
    if not pairs:
        raise ValueError("ALS training requires at least one positive interaction.")

    user_ids = tuple(sorted({user for user, _ in pairs}))
    tmdb_ids = tuple(sorted({tmdb for _, tmdb in pairs}))
    movie_ids = tuple(mapper.to_movie_id(tmdb) for tmdb in tmdb_ids)
    users = {user: index for index, user in enumerate(user_ids)}
    items = {tmdb: index for index, tmdb in enumerate(tmdb_ids)}
    ordered_pairs = sorted(pairs)
    matrix = csr_matrix(
        (
            np.ones(len(ordered_pairs), dtype=np.float32),
            ([users[user] for user, _ in ordered_pairs], [items[tmdb] for _, tmdb in ordered_pairs]),
        ),
        shape=(len(user_ids), len(tmdb_ids)), dtype=np.float32,
    )
    return TrainingData(matrix, user_ids, tmdb_ids, movie_ids)


def _popularity(data: TrainingData) -> tuple[tuple[int, int], ...]:
    counts = data.matrix.getnnz(axis=0)
    return tuple(sorted(
        ((tmdb, int(count)) for tmdb, count in zip(data.tmdb_ids, counts)),
        key=lambda pair: (-pair[1], pair[0]),
    ))


def _json_copy(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object.")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain JSON-compatible, finite values.") from exc


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class ImplicitRecommender:
    """ALS recommendations without a stored user ID or stateful profile updates.

    Favorites are binary. implicit applies alpha once during training and once
    when recalculating the request profile. Unknown favorites are ignored;
    requests with no known favorites use training popularity instead.
    """

    def __init__(
        self, model: AlternatingLeastSquares, training_data: TrainingData,
        config: ALSConfig, metadata: dict | None = None,
    ):
        self.model = model
        self.training_data = training_data
        self.config = config
        self.metadata = _json_copy({} if metadata is None else metadata)
        self._validate_state()
        self._item_indices = {tmdb: index for index, tmdb in enumerate(training_data.tmdb_ids)}
        self._ranking = _popularity(training_data)

    @classmethod
    def fit(
        cls, train: pd.DataFrame, links: pd.DataFrame, config: ALSConfig | None = None,
    ) -> "ImplicitRecommender":
        """Fit on the supplied training split; never inspect validation or test."""
        config = ALSConfig() if config is None else config
        if not isinstance(config, ALSConfig):
            raise ValueError("config must be an ALSConfig.")
        data = build_interaction_matrix(train, links)
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(**asdict(config), dtype=np.float32)
            model.fit(data.matrix, show_progress=False)
        return cls(model, data, config)

    def recommend(
        self, favorite_tmdb_ids: Iterable[int], n: int = 10,
    ) -> list[dict[str, int | float]]:
        """Return TMDB IDs and scores; exclude only the supplied favorites."""
        return self.recommend_with_method(favorite_tmdb_ids, n)["recommendations"]

    def recommend_with_method(self, favorite_tmdb_ids: Iterable[int], n: int = 10) -> dict:
        """Return recommendations and identify ALS or the popularity fallback."""
        limit = _integer(n, "n")
        if isinstance(favorite_tmdb_ids, (str, bytes)):
            raise ValueError("favorite_tmdb_ids must be an iterable of positive integers.")
        try:
            favorites = {_integer(value, "favorite_tmdb_ids") for value in favorite_tmdb_ids}
        except TypeError as exc:
            raise ValueError("favorite_tmdb_ids must be an iterable of positive integers.") from exc
        known = sorted(self._item_indices[tmdb] for tmdb in favorites if tmdb in self._item_indices)
        if not known:
            recommendations = [
                {"tmdb_id": tmdb, "score": float(count)}
                for tmdb, count in self._ranking if tmdb not in favorites
            ][:limit]
            return {"method": "popularity", "recommendations": recommendations}
        if len(known) == len(self.training_data.tmdb_ids):
            return {"method": "als", "recommendations": []}

        profile = csr_matrix(
            (np.ones(len(known), dtype=np.float32), ([0] * len(known), known)),
            shape=(1, len(self.training_data.tmdb_ids)), dtype=np.float32,
        )
        with threadpool_limits(limits=1, user_api="blas"):
            indices, scores = self.model.recommend(
                0, profile, N=len(self.training_data.tmdb_ids),
                filter_already_liked_items=False, recalculate_user=True,
            )
        if not np.isfinite(scores).all():
            raise ValueError("ALS produced non-finite recommendation scores.")
        ranked = sorted(
            (
                (self.training_data.tmdb_ids[int(index)], float(score))
                for index, score in zip(indices, scores)
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        recommendations = [
            {"tmdb_id": tmdb, "score": score}
            for tmdb, score in ranked if tmdb not in favorites
        ][:limit]
        return {"method": "als", "recommendations": recommendations}

    def _validate_state(self) -> None:
        data = self.training_data
        for name in ("user_ids", "tmdb_ids", "movie_ids"):
            ids = getattr(data, name)
            if not ids or any(_integer(value, name) != value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Artifact {name} must contain unique positive integer IDs.")
        if data.user_ids != tuple(sorted(data.user_ids)) or data.tmdb_ids != tuple(sorted(data.tmdb_ids)):
            raise ValueError("Artifact user and TMDB IDs must be sorted.")
        if len(data.movie_ids) != len(data.tmdb_ids):
            raise ValueError("Artifact MovieLens and TMDB mappings must have equal length.")
        matrix = data.matrix
        if not isinstance(matrix, csr_matrix):
            raise ValueError("Artifact interactions must use CSR format.")
        matrix.check_format(full_check=True)
        if (
            matrix.dtype != np.float32
            or matrix.shape != (len(data.user_ids), len(data.tmdb_ids))
            or not matrix.has_canonical_format or not np.equal(matrix.data, 1).all()
            or np.any(np.diff(matrix.indptr) == 0) or np.any(np.asarray(matrix.sum(axis=0)) == 0)
        ):
            raise ValueError("Artifact interactions must be a nonempty binary float32 CSR matrix.")
        for name, rows in (("user_factors", len(data.user_ids)), ("item_factors", len(data.tmdb_ids))):
            factors = getattr(self.model, name)
            if (
                not isinstance(factors, np.ndarray) or factors.shape != (rows, self.config.factors)
                or factors.dtype != np.float32 or not np.isfinite(factors).all()
            ):
                raise ValueError(f"Artifact {name} have invalid dimensions, dtype or values.")
        for name, value in asdict(self.config).items():
            if getattr(self.model, name) != value:
                raise ValueError(f"Artifact model and configuration disagree on {name}.")

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        """Save a verified model bundle to a new directory without overwriting."""
        target = Path(directory)
        if target.exists():
            raise FileExistsError(f"Artifact directory already exists: {target}")
        self._validate_state()
        if self._ranking != _popularity(self.training_data):
            raise ValueError("Training interactions changed after fitting the popularity fallback.")
        metadata = _json_copy(self.metadata if metadata is None else metadata)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            self.model.save(bundle / "model.npz")
            save_npz(bundle / "interactions.npz", self.training_data.matrix, compressed=True)
            _write_json(bundle / "catalog.json", {
                "user_ids": self.training_data.user_ids,
                "tmdb_ids": self.training_data.tmdb_ids,
                "movie_ids": self.training_data.movie_ids,
                "popularity": self._ranking,
            })
            _write_json(bundle / "manifest.json", {
                "format_version": 1,
                "implicit_version": implicit.__version__,
                "config": asdict(self.config),
                "dimensions": {
                    "users": len(self.training_data.user_ids),
                    "items": len(self.training_data.tmdb_ids),
                    "factors": self.config.factors,
                    "interactions": self.training_data.matrix.nnz,
                },
                "sha256": {name: _sha256(bundle / name) for name in (
                    "model.npz", "interactions.npz", "catalog.json",
                )},
                "metadata": metadata,
            })
            if target.exists():
                raise FileExistsError(f"Artifact directory already exists: {target}")
            bundle.rename(target)

    @classmethod
    def load(cls, directory: str | Path) -> "ImplicitRecommender":
        """Load without CSV files; reject inconsistent or corrupted bundles."""
        bundle = Path(directory)
        try:
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            if (
                type(manifest["format_version"]) is not int or manifest["format_version"] != 1
                or manifest["implicit_version"] != implicit.__version__
            ):
                raise ValueError("Unsupported artifact format or implicit version.")
            files = ("model.npz", "interactions.npz", "catalog.json")
            if set(manifest["sha256"]) != set(files):
                raise ValueError("Artifact manifest must declare the expected data files.")
            for name in files:
                if _sha256(bundle / name) != manifest["sha256"][name]:
                    raise ValueError(f"Artifact checksum mismatch: {name}.")
            if set(manifest["config"]) != set(asdict(ALSConfig())):
                raise ValueError("Artifact configuration must declare every ALS parameter.")
            config = ALSConfig(**manifest["config"])
            catalog = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
            data = TrainingData(
                load_npz(bundle / "interactions.npz"), tuple(catalog["user_ids"]),
                tuple(catalog["tmdb_ids"]), tuple(catalog["movie_ids"]),
            )
            with threadpool_limits(limits=1, user_api="blas"):
                model = AlternatingLeastSquares.load(bundle / "model.npz")
            instance = cls(model, data, config, metadata=manifest["metadata"])
            expected_dimensions = {
                "users": len(data.user_ids), "items": len(data.tmdb_ids),
                "factors": config.factors, "interactions": data.matrix.nnz,
            }
            if manifest["dimensions"] != expected_dimensions or any(
                type(value) is not int for value in manifest["dimensions"].values()
            ):
                raise ValueError("Artifact dimensions do not match its contents.")
            if catalog["popularity"] != [list(pair) for pair in instance._ranking] or any(
                type(value) is not int for pair in catalog["popularity"] for value in pair
            ):
                raise ValueError("Artifact popularity does not match training interactions.")
            return instance
        except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise ValueError(f"Invalid ALS artifact bundle: {exc}") from exc
