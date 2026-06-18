from __future__ import annotations

from pydantic import BaseModel


class ClusterProfile(BaseModel):
    cluster_id: int
    # crime type composition
    cm_transeunte: float
    cm_vehiculo: float
    cm_negocio: float
    cm_repartidor: float
    cm_metro: float
    cm_violacion: float
    cm_homicidio: float
    cm_microbus: float
    cm_lesiones: float
    cm_casa: float
    # temporal patterns
    hr_madrugada: float
    hr_manana: float
    hr_tarde: float
    hr_noche: float
    # day-of-week distribution
    ds_0: float
    ds_1: float
    ds_2: float
    ds_3: float
    ds_4: float
    ds_5: float
    ds_6: float
    log_n: float
    n_cells: int


class TransitionMatrix(BaseModel):
    # from_cluster → {to_cluster_str: probability}
    matrix: dict[int, dict[str, float]]


class ClusterDetail(BaseModel):
    profile: ClusterProfile
    cell_h3_indices: list[str]
