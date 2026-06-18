from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings


class EmbedRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=22,
        max_length=22,
        description=(
            f"22 normalized crime signature features in order: {settings.FEATURE_COLS}"
        ),
    )


class EmbedResponse(BaseModel):
    embedding: list[float] = Field(..., description="8-dimensional latent vector")
