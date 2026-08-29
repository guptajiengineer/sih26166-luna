"""Domain models for lunar image registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SensorType(str, Enum):
    OHRC = "OHRC"
    TMC2 = "TMC2"
    IIRS = "IIRS"


@dataclass
class LunarImage:
    """Multi-temporal, multi-sensor lunar imagery metadata + array."""

    data: np.ndarray
    sensor: SensorType
    sun_angle_deg: float
    resolution_m: float
    geotransform: tuple[float, float, float, float, float, float] | None = None
    crs: str = "EPSG:4326"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def scale_ratio_hint(self) -> float:
        """Relative scale hint vs OHRC baseline (0.25 m)."""
        return self.resolution_m / 0.25


@dataclass
class Keypoint:
    x: float
    y: float
    descriptor: np.ndarray | None = None
    response: float = 0.0
    source: str = "unknown"


@dataclass
class TiePoint:
    """Correspondence between reference and moving image."""

    ref_x: float
    ref_y: float
    mov_x: float
    mov_y: float
    confidence: float = 1.0
    inlier: bool = True

    def as_array(self) -> np.ndarray:
        return np.array([self.ref_x, self.ref_y, self.mov_x, self.mov_y])


@dataclass
class RegistrationResult:
    tie_points: list[TiePoint]
    transform_matrix: np.ndarray
    registered_image: np.ndarray | None
    rmse_px: float
    inlier_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)
