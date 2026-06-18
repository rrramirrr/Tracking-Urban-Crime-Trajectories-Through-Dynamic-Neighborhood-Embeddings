"""
In-memory store for all precomputed pipeline outputs.

Design decision: we load every CSV once at startup into Python dicts keyed by
h3_8 or cluster id.  The dataset is static (<10 MB total) so RAM is cheap and
every request becomes an O(1) dict lookup instead of a disk read.  A database
would add operational complexity without any benefit at this scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._loaded = False

    def load(self) -> None:
        d = self._data_dir

        self.trajectories: dict[str, dict] = self._csv_to_dict(
            d / "trajectories.csv", key="h3_8"
        )
        self.cell_profiles: dict[str, dict] = self._csv_to_dict(
            d / "cell_profiles.csv", key="h3_8"
        )
        # cluster_assignments indexed as {h3: [{año, cluster_km_ae, ...}, ...]}
        raw_ca = pd.read_csv(d / "cluster_assignments.csv")
        self.cluster_assignments: dict[str, list[dict]] = (
            raw_ca.groupby("h3_8")
            .apply(lambda g: g.to_dict("records"), include_groups=False)
            .to_dict()
        )

        self.cluster_profiles: dict[int, dict] = self._csv_to_dict(
            d / "cluster_profiles.csv", key="cluster"
        )

        raw_tm = pd.read_csv(d / "transition_matrix.csv")
        self.transition_matrix: dict[int, dict[str, float]] = (
            raw_tm.set_index("from")
            .apply(lambda r: r.to_dict(), axis=1)
            .to_dict()
        )

        self.predictions_2024: dict[str, dict] = self._csv_to_dict(
            d / "predictions_2024.csv", key="h3_8"
        )

        # Reverse index: cluster → list of h3 cells (using modal cluster)
        self.cells_by_cluster: dict[int, list[str]] = {}
        for h3, profile in self.cell_profiles.items():
            c = int(profile["cluster_modal"])
            self.cells_by_cluster.setdefault(c, []).append(h3)

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _csv_to_dict(path: Path, key: str) -> dict[Any, dict]:
        df = pd.read_csv(path)
        return df.set_index(key).to_dict("index")
