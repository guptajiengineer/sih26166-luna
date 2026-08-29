"""Pipeline execution context passed between stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.domain import Keypoint, LunarImage, RegistrationResult, TiePoint


@dataclass
class PipelineContext:
    reference: LunarImage
    moving: LunarImage
    config: dict[str, Any] = field(default_factory=dict)

    # Stage outputs
    rag_params: dict[str, Any] = field(default_factory=dict)
    ref_keypoints: list[Keypoint] = field(default_factory=list)
    mov_keypoints: list[Keypoint] = field(default_factory=list)
    tie_points: list[TiePoint] = field(default_factory=list)
    inlier_tie_points: list[TiePoint] = field(default_factory=list)
    transform_matrix: np.ndarray | None = None
    registered_image: np.ndarray | None = None
    result: RegistrationResult | None = None
    stage_timings: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
