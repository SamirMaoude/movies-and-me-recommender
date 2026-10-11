"""Request validation and responses for the recommendation API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_FAVORITES = 500
MAX_RECOMMENDATIONS = 50
PositiveId = Annotated[int, Field(strict=True, gt=0)]
RecommendationLimit = Annotated[int, Field(strict=True, ge=1, le=MAX_RECOMMENDATIONS)]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    favorite_movie_ids: list[PositiveId] = Field(max_length=MAX_FAVORITES)
    n: RecommendationLimit = 10


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str | None = None
    version: str | None = None
    run_id: str | None = None
    model_uri: str = Field(min_length=1)


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tmdb_id: PositiveId
    score: float = Field(strict=True, allow_inf_nan=False)


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendations: list[RecommendationItem] = Field(max_length=MAX_RECOMMENDATIONS)
    method: Literal["als", "popularity"]
    model: ModelInfo


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
