"""Spatial consistency neighborhood filter."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.geometry.base import GeometricEstimator
from src.models.domain import TiePoint


class SpatialConsistencyFilter(GeometricEstimator):
    """
    Removes matches whose local affine deviation exceeds threshold.
    Enforces spatially uniform, geometrically consistent tie points.
    """

    @property
    def name(self) -> str:
        return "spatial_consistency"

    def estimate(
        self, tie_points: list[TiePoint], params: dict[str, Any]
    ) -> tuple[list[TiePoint], np.ndarray]:
        if len(tie_points) < 6:
            from src.geometry.magsac import MAGSACEstimator
            return MAGSACEstimator().estimate(tie_points, params)

        k = params.get("neighbors", 6)
        max_dev = params.get("max_local_deviation", 3.0)

        pts = np.array([[tp.ref_x, tp.ref_y, tp.mov_x, tp.mov_y] for tp in tie_points])
        n = len(pts)
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            dists = np.linalg.norm(pts[:, :2] - pts[i, :2], axis=1)
            nn_idx = np.argsort(dists)[1: k + 1]
            if len(nn_idx) < 3:
                continue

            ref_nn = pts[nn_idx, :2]
            mov_nn = pts[nn_idx, 2:]
            ref_c = pts[i, :2]
            mov_c = pts[i, 2:]

            # Local translation estimate
            shifts = (mov_nn - ref_nn) - (mov_c - ref_c)
            med_shift = np.median(shifts, axis=0)
            dev = np.linalg.norm(shifts - med_shift, axis=1).max()
            if dev > max_dev:
                keep[i] = False

        filtered: list[TiePoint] = []
        for tp, k_flag in zip(tie_points, keep):
            tp.inlier = bool(k_flag)
            if k_flag:
                filtered.append(tp)

        from src.geometry.magsac import MAGSACEstimator
        return MAGSACEstimator().estimate(filtered, params)
