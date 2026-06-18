from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClusterYearRecord(BaseModel):
    year: int = Field(alias="año_hecho")
    cluster_km_ae: int
    cluster_km_ae_cos: int
    cluster_km_pca: int
    cluster_km_raw: int
    cluster_hdbscan: int

    model_config = {"populate_by_name": True}


class CellTrajectory(BaseModel):
    h3_index: str
    trajectory_type: str
    n_years: int
    year_start: int
    year_end: int
    n_cluster_changes: int
    n_distinct_clusters: int
    cluster_start: int
    cluster_end: int
    drift_index: float
    path_length_ae: float
    net_displacement_ae: float
    is_atypical: bool
    cluster_sequence: str
    years_observed: str
    year_by_year: list[ClusterYearRecord]


class CellSummary(BaseModel):
    h3_index: str
    stability: float
    modal_cluster: int
    n_years: int
    trajectory_type: str
    is_atypical: bool
    prediction_2024: Optional[int]
