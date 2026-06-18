from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_encoder
from app.schemas.embed import EmbedRequest, EmbedResponse
from app.services.encoder import EncoderService

router = APIRouter(prefix="/embed", tags=["inference"])


@router.post("", response_model=EmbedResponse)
def embed(
    body: EmbedRequest,
    encoder: EncoderService = Depends(get_encoder),
) -> EmbedResponse:
    """
    Given a raw crime signature (22 features), return its 8-dimensional latent
    embedding produced by the trained autoencoder encoder.

    Features must be in this order:
    cm_transeunte, cm_vehiculo, cm_negocio, cm_repartidor, cm_metro,
    cm_violacion, cm_homicidio, cm_microbus, cm_lesiones, cm_casa,
    hr_madrugada, hr_manana, hr_tarde, hr_noche,
    ds_0..ds_6, log_n
    """
    embedding = encoder.encode(body.features)
    return EmbedResponse(embedding=embedding)
