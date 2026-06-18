from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_data_store
from app.schemas.cell import CellSummary, CellTrajectory, ClusterYearRecord
from app.services.data_store import DataStore

router = APIRouter(prefix="/cells", tags=["cells"])


def _require_cell(h3_index: str, store: DataStore) -> tuple[dict, dict]:
    traj = store.trajectories.get(h3_index)
    profile = store.cell_profiles.get(h3_index)
    if traj is None or profile is None:
        raise HTTPException(status_code=404, detail=f"Cell '{h3_index}' not found")
    return traj, profile


@router.get("/{h3_index}", response_model=CellSummary)
def get_cell(h3_index: str, store: DataStore = Depends(get_data_store)) -> CellSummary:
    """Return a summary for a single H3 cell: stability, modal cluster, and trajectory type."""
    traj, profile = _require_cell(h3_index, store)
    pred = store.predictions_2024.get(h3_index)
    return CellSummary(
        h3_index=h3_index,
        stability=float(profile["estabilidad"]),
        modal_cluster=int(profile["cluster_modal"]),
        n_years=int(profile["n_años"]),
        trajectory_type=str(traj["tipo"]),
        is_atypical=bool(traj["is_atipica"]),
        prediction_2024=int(pred["cluster_pred"]) if pred else None,
    )


@router.get("/{h3_index}/trajectory", response_model=CellTrajectory)
def get_cell_trajectory(
    h3_index: str, store: DataStore = Depends(get_data_store)
) -> CellTrajectory:
    """Return the full year-by-year cluster trajectory for a single H3 cell."""
    traj, _ = _require_cell(h3_index, store)
    yearly = store.cluster_assignments.get(h3_index, [])
    return CellTrajectory(
        h3_index=h3_index,
        trajectory_type=str(traj["tipo"]),
        n_years=int(traj["n_años"]),
        year_start=int(traj["año_inicio"]),
        year_end=int(traj["año_fin"]),
        n_cluster_changes=int(traj["n_cambios"]),
        n_distinct_clusters=int(traj["n_clusters"]),
        cluster_start=int(traj["cluster_inicio"]),
        cluster_end=int(traj["cluster_fin"]),
        drift_index=float(traj["drift_idx"]),
        path_length_ae=float(traj["path_length_ae"]),
        net_displacement_ae=float(traj["net_disp_ae"]),
        is_atypical=bool(traj["is_atipica"]),
        cluster_sequence=str(traj["secuencia"]),
        years_observed=str(traj["años"]),
        year_by_year=[ClusterYearRecord(**r) for r in yearly],
    )
