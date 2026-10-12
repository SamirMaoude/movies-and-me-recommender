"""Serve recommendations from one model loaded at startup."""

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    RecommendationRequest,
    RecommendationResponse,
    VersionInfo,
)
from src.api.service import RecommendationService

DEFAULT_MODEL_URI = "models:/movies-and-me-als/1"
logger = logging.getLogger(__name__)


def _ready_service(request: Request) -> RecommendationService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return service


ReadyService = Annotated[RecommendationService, Depends(_ready_service)]


def _setting(
    value: str | None, variable: str, default: str | None = None
) -> str | None:
    """Prefer an explicit value, otherwise read the environment at startup."""
    return os.getenv(variable, default) if value is None else value


def create_app(
    model_uri: str | None = None,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
    revision: str | None = None,
    service_factory: Callable[..., RecommendationService] | None = None,
) -> FastAPI:
    """Create the API; the model is loaded when the application starts."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            factory = service_factory or RecommendationService.from_source
            application.state.service = factory(
                _setting(model_uri, "MODEL_URI", DEFAULT_MODEL_URI),
                tracking_uri=_setting(tracking_uri, "MLFLOW_TRACKING_URI"),
                registry_uri=_setting(registry_uri, "MLFLOW_REGISTRY_URI"),
            )
            # An image built without VCS_REF has an empty APP_REVISION.
            application.state.revision = _setting(revision, "APP_REVISION") or None
            yield
        finally:
            application.state.service = None

    application = FastAPI(title="Movies & Me Recommender", lifespan=lifespan)
    application.state.service = None
    application.state.revision = None

    @application.get("/health", response_model=HealthResponse)
    def health(service: ReadyService) -> HealthResponse:
        """Confirm that the model is ready to answer requests."""
        return HealthResponse(status="ok")

    @application.get("/model/info", response_model=ModelInfo)
    def model_info(service: ReadyService) -> ModelInfo:
        """Return the identity of the model loaded for this process."""
        return service.model_info

    @application.get("/version", response_model=VersionInfo)
    def version(request: Request, service: ReadyService) -> VersionInfo:
        """Return the code revision and the model served by this process."""
        return VersionInfo(
            revision=request.app.state.revision, model=service.model_info
        )

    @application.post("/recommendations", response_model=RecommendationResponse)
    def recommendations(
        request: RecommendationRequest,
        service: ReadyService,
    ) -> RecommendationResponse:
        """Recommend movies from a list of favorite TMDB IDs."""
        try:
            return service.recommend(request.favorite_movie_ids, n=request.n)
        except Exception:
            logger.exception("Recommendation computation failed.")
            raise HTTPException(
                status_code=500,
                detail="Recommendation computation failed.",
            ) from None

    return application


app = create_app()
