"""
FastAPI dependency injection singletons.

Both DataStore and EncoderService are created once during app lifespan and
reused across all requests — no per-request initialization overhead.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services.data_store import DataStore
from app.services.encoder import EncoderService


@lru_cache(maxsize=1)
def get_data_store() -> DataStore:
    store = DataStore(settings.data_dir)
    store.load()
    return store


@lru_cache(maxsize=1)
def get_encoder() -> EncoderService:
    return EncoderService(
        models_dir=settings.models_dir,
        scaler_path=settings.data_dir / "scaler_firmas.pkl",
    )
