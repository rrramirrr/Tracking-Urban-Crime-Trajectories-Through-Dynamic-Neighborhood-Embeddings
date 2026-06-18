from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_data_store
from app.schemas.cluster import ClusterDetail, ClusterProfile, TransitionMatrix
from app.services.data_store import DataStore

router = APIRouter(prefix="/clusters", tags=["clusters"])


def _build_profile(cluster_id: int, raw: dict, n_cells: int) -> ClusterProfile:
    return ClusterProfile(
        cluster_id=cluster_id,
        cm_transeunte=raw["cm_transeunte"],
        cm_vehiculo=raw["cm_vehiculo"],
        cm_negocio=raw["cm_negocio"],
        cm_repartidor=raw["cm_repartidor"],
        cm_metro=raw["cm_metro"],
        cm_violacion=raw["cm_violacion"],
        cm_homicidio=raw["cm_homicidio"],
        cm_microbus=raw["cm_microbus"],
        cm_lesiones=raw["cm_lesiones"],
        cm_casa=raw["cm_casa"],
        hr_madrugada=raw["hr_madrugada"],
        hr_manana=raw["hr_manana"],
        hr_tarde=raw["hr_tarde"],
        hr_noche=raw["hr_noche"],
        ds_0=raw["ds_0"],
        ds_1=raw["ds_1"],
        ds_2=raw["ds_2"],
        ds_3=raw["ds_3"],
        ds_4=raw["ds_4"],
        ds_5=raw["ds_5"],
        ds_6=raw["ds_6"],
        log_n=raw["log_n"],
        n_cells=n_cells,
    )


@router.get("/transitions", response_model=TransitionMatrix)
def get_transitions(store: DataStore = Depends(get_data_store)) -> TransitionMatrix:
    """Return the Markov transition matrix between clusters (row = from, col = to)."""
    return TransitionMatrix(matrix=store.transition_matrix)


@router.get("", response_model=list[ClusterProfile])
def list_clusters(store: DataStore = Depends(get_data_store)) -> list[ClusterProfile]:
    """List all cluster profiles with crime composition and temporal patterns."""
    return [
        _build_profile(cid, raw, len(store.cells_by_cluster.get(int(cid), [])))
        for cid, raw in store.cluster_profiles.items()
    ]


@router.get("/{cluster_id}", response_model=ClusterDetail)
def get_cluster(
    cluster_id: int, store: DataStore = Depends(get_data_store)
) -> ClusterDetail:
    """Return profile and member cells for a single cluster."""
    raw = store.cluster_profiles.get(cluster_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    cells = store.cells_by_cluster.get(cluster_id, [])
    return ClusterDetail(
        profile=_build_profile(cluster_id, raw, len(cells)),
        cell_h3_indices=cells,
    )
