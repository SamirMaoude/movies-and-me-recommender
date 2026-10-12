"""Serve one loaded ALS model independently of the HTTP application."""

from pathlib import Path
from threading import Lock

from src.api.schemas import ModelInfo, RecommendationResponse
from src.models.registry import LoadedRecommender, load_recommender

# The model calls run under threadpool_limits, which changes process-wide BLAS
# settings: loads and predictions take turns, even across services.
_prediction_lock = Lock()


class RecommendationService:
    def __init__(self, loaded: LoadedRecommender):
        self._loaded = loaded
        self._model_info = ModelInfo(
            name=loaded.name,
            version=loaded.version,
            run_id=loaded.run_id,
            model_uri=loaded.model_uri,
        )

    @classmethod
    def from_source(
        cls,
        model_uri: str | Path,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> "RecommendationService":
        with _prediction_lock:
            loaded = load_recommender(
                model_uri, tracking_uri=tracking_uri, registry_uri=registry_uri
            )
        return cls(loaded)

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    def recommend(
        self, favorite_movie_ids: list[int], n: int = 10
    ) -> RecommendationResponse:
        """Recommend from favorites already validated by the request schema."""
        with _prediction_lock:
            result = self._loaded.recommend_with_method(favorite_movie_ids, n=n)
        # The response schema rejects non-finite scores before they reach a client.
        return RecommendationResponse(
            recommendations=result["recommendations"],
            method=result["method"],
            model=self._model_info,
        )

    def count_known(self, favorite_movie_ids: list[int]) -> int:
        return self._loaded.model.count_known(favorite_movie_ids)
