from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.config import settings
from app.dependencies import get_data_store, get_encoder
from app.routers import cells, clusters, embed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Eagerly warm up singletons so the first request is never slow
    get_data_store()
    get_encoder()
    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "REST API for the UrbanCrime9 project. Serves precomputed crime trajectory "
        "embeddings, cluster profiles, and Markov transition statistics for H3 cells "
        "in Mexico City (2016–2024). Also exposes the trained autoencoder encoder for "
        "online crime signature embedding."
    ),
    lifespan=lifespan,
)

app.include_router(cells.router)
app.include_router(clusters.router)
app.include_router(embed.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    store = get_data_store()
    return {
        "status": "ok",
        "n_cells": len(store.trajectories),
        "n_clusters": len(store.cluster_profiles),
    }
