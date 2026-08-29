"""Measurable Part of the Project — quantifiable registration metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models.domain import TiePoint


@dataclass
class StageTiming:
    name: str
    duration_s: float


@dataclass
class MPPReport:
    """Measurable Part of the Project report."""

    registration_rmse_px: float = 0.0
    sub_pixel_accuracy: bool = False
    tie_point_count: int = 0
    inlier_ratio: float = 0.0
    tie_point_uniformity_entropy: float = 0.0
    grid_occupancy_pct: float = 0.0
    stage_timings: list[StageTiming] = field(default_factory=list)
    rag_retrieval_relevance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "registration_rmse_px": self.registration_rmse_px,
            "sub_pixel_accuracy": self.sub_pixel_accuracy,
            "tie_point_count": self.tie_point_count,
            "inlier_ratio": self.inlier_ratio,
            "tie_point_uniformity_entropy": self.tie_point_uniformity_entropy,
            "grid_occupancy_pct": self.grid_occupancy_pct,
            "stage_timings": [{"name": t.name, "duration_s": t.duration_s} for t in self.stage_timings],
            "rag_retrieval_relevance": self.rag_retrieval_relevance,
            "metadata": self.metadata,
            "passes_mpp": self.passes_mpp(),
        }

    def passes_mpp(self) -> bool:
        return (
            self.registration_rmse_px < 0.5
            and self.inlier_ratio >= 0.6
            and self.tie_point_uniformity_entropy >= 3.5
            and self.grid_occupancy_pct >= 70.0
            and self.sub_pixel_accuracy
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class MPPMetrics:
    """Computes all measurable project metrics."""

    @staticmethod
    def registration_rmse(tie_points: list[TiePoint], transform: np.ndarray) -> float:
        if len(tie_points) < 1:
            return float("inf")
        errors = []
        H = transform
        for tp in tie_points:
            pt = np.array([tp.mov_x, tp.mov_y, 1.0])
            proj = H @ pt
            proj /= proj[2] + 1e-12
            err = np.sqrt((proj[0] - tp.ref_x) ** 2 + (proj[1] - tp.ref_y) ** 2)
            errors.append(err)
        return float(np.mean(errors))

    @staticmethod
    def inlier_ratio(tie_points: list[TiePoint]) -> float:
        if not tie_points:
            return 0.0
        return sum(1 for tp in tie_points if tp.inlier) / len(tie_points)

    @staticmethod
    def spatial_entropy(tie_points: list[TiePoint], image_shape: tuple[int, int], grid: int = 16) -> float:
        if not tie_points:
            return 0.0
        h, w = image_shape[:2]
        cell_h, cell_w = h / grid, w / grid
        counts = np.zeros((grid, grid))
        for tp in tie_points:
            gi = min(int(tp.ref_y / cell_h), grid - 1)
            gj = min(int(tp.ref_x / cell_w), grid - 1)
            counts[gi, gj] += 1
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts.flatten() / total
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log2(probs)))
        return entropy

    @staticmethod
    def grid_occupancy(tie_points: list[TiePoint], image_shape: tuple[int, int], grid: int = 16) -> float:
        if not tie_points:
            return 0.0
        h, w = image_shape[:2]
        cell_h, cell_w = h / grid, w / grid
        occupied = set()
        for tp in tie_points:
            gi = min(int(tp.ref_y / cell_h), grid - 1)
            gj = min(int(tp.ref_x / cell_w), grid - 1)
            occupied.add((gi, gj))
        return 100.0 * len(occupied) / (grid * grid)

    @classmethod
    def build_report(
        cls,
        tie_points: list[TiePoint],
        transform: np.ndarray,
        image_shape: tuple[int, int],
        stage_timings: list[StageTiming] | None = None,
        rag_relevance: float = 0.0,
    ) -> MPPReport:
        rmse = cls.registration_rmse(tie_points, transform)
        return MPPReport(
            registration_rmse_px=rmse,
            sub_pixel_accuracy=rmse < 0.5,
            tie_point_count=len(tie_points),
            inlier_ratio=cls.inlier_ratio(tie_points),
            tie_point_uniformity_entropy=cls.spatial_entropy(tie_points, image_shape),
            grid_occupancy_pct=cls.grid_occupancy(tie_points, image_shape),
            stage_timings=stage_timings or [],
            rag_retrieval_relevance=rag_relevance,
        )


class StageTimer:
    """Context manager for stage latency measurement."""

    def __init__(self, name: str, timings: list[StageTiming]) -> None:
        self.name = name
        self.timings = timings
        self._start = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.timings.append(StageTiming(self.name, time.perf_counter() - self._start))
