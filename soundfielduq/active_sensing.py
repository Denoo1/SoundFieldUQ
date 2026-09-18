"""Optimal Sensor Placement using UQ minimization and spatial distance constraints."""

import numpy as np


class ActiveSensorPlacement:
    """Greedy spatial microphone placement engine."""

    def __init__(
        self, candidate_grid: np.ndarray, min_distance: float = 0.20
    ) -> None:
        grid = np.asarray(candidate_grid, dtype=np.float64)
        if grid.ndim != 2 or grid.shape[1] not in (2, 3):
            raise ValueError(f"candidate_grid must be (N, 2) or (N, 3), got {grid.shape}")
        if min_distance < 0.0:
            raise ValueError(f"min_distance must be non-negative, got {min_distance}")

        self.candidate_grid: np.ndarray = grid
        self.min_distance: float = min_distance
        self.selected_indices: list[int] = []

    def select_next_sensor(self, spatial_uncertainty: np.ndarray) -> int:
        """Selects optimal sensor position based on spatial prediction interval widths."""
        uncertainties = np.asarray(spatial_uncertainty, dtype=np.float64).ravel()
        if uncertainties.size != self.candidate_grid.shape[0]:
            raise ValueError(
                f"Uncertainty array length ({uncertainties.size}) must match candidate grid count ({self.candidate_grid.shape[0]})"
            )

        ranked_indices = np.argsort(uncertainties)[::-1]

        for idx in ranked_indices:
            candidate_pos = self.candidate_grid[idx]

            if self.selected_indices:
                selected_positions = self.candidate_grid[self.selected_indices]
                distances = np.linalg.norm(selected_positions - candidate_pos, axis=1)

                if np.any(distances < self.min_distance):
                    continue

            self.selected_indices.append(int(idx))
            return int(idx)

        raise RuntimeError(
            "No valid candidate positions remain under distance constraints."
        )

    def get_selected_coordinates(self) -> np.ndarray:
        """Returns array of selected sensor physical coordinates."""
        if not self.selected_indices:
            return np.empty((0, self.candidate_grid.shape[1]), dtype=np.float64)
        return self.candidate_grid[self.selected_indices]
