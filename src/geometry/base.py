"""Robust geometric estimation and outlier filtering."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import cv2
import numpy as np

from src.models.domain import TiePoint
from src.patterns import Strategy


class GeometricEstimator(Strategy[tuple[list[TiePoint], dict], tuple[list[TiePoint], np.ndarray]]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def execute(
        self, context: tuple[list[TiePoint], dict]
    ) -> tuple[list[TiePoint], np.ndarray]:
        tie_points, params = context
        return self.estimate(tie_points, params)

    @abstractmethod
    def estimate(
        self, tie_points: list[TiePoint], params: dict[str, Any]
    ) -> tuple[list[TiePoint], np.ndarray]:
        ...


def _to_point_arrays(tie_points: list[TiePoint]) -> tuple[np.ndarray, np.ndarray]:
    src = np.float32([[tp.mov_x, tp.mov_y] for tp in tie_points])
    dst = np.float32([[tp.ref_x, tp.ref_y] for tp in tie_points])
    return src, dst


def _apply_inlier_mask(tie_points: list[TiePoint], mask: np.ndarray) -> list[TiePoint]:
    filtered: list[TiePoint] = []
    for tp, keep in zip(tie_points, mask.ravel()):
        tp.inlier = bool(keep)
        if keep:
            filtered.append(tp)
    return filtered
