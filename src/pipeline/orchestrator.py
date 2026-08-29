"""Main registration pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models.domain import LunarImage, SensorType
from src.mpp.metrics import MPPMetrics
from src.patterns import Pipeline
from src.pipeline.context import PipelineContext
from src.pipeline.stages import (
    FeatureExtractionStage,
    GeometricFilteringStage,
    MatchingStage,
    PreprocessingStage,
    RAGParameterStage,
    RegistrationStage,
)


class LunarRegistrationPipeline:
    """
    End-to-end RAG-augmented multi-sensor lunar image registration.

    Pipeline order:
      RAG → Preprocess → Features → Match → Geometric Filter → Register
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._pipeline = self._build_pipeline()

    @classmethod
    def from_yaml(cls, path: str | Path) -> LunarRegistrationPipeline:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def _build_pipeline(self) -> Pipeline[PipelineContext]:
        pipe = Pipeline[PipelineContext]()
        pipe.add_stage(RAGParameterStage())
        pipe.add_stage(PreprocessingStage())
        pipe.add_stage(FeatureExtractionStage())
        pipe.add_stage(MatchingStage())
        pipe.add_stage(GeometricFilteringStage())
        pipe.add_stage(RegistrationStage())
        return pipe

    def register(
        self,
        reference: LunarImage,
        moving: LunarImage,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[PipelineContext, Any]:
        ctx = PipelineContext(
            reference=reference,
            moving=moving,
            config={**self.config.get("pipeline", {}), **(overrides or {})},
        )
        ctx = self._pipeline.run(ctx)

        assert ctx.result is not None
        mpp = MPPMetrics.build_report(
            tie_points=ctx.inlier_tie_points,
            transform=ctx.transform_matrix,
            image_shape=reference.data.shape,
            stage_timings=ctx.stage_timings,
            rag_relevance=ctx.metadata.get("rag_mean_relevance", 0.0),
        )
        ctx.metadata["mpp"] = mpp.to_dict()
        return ctx, mpp

    @staticmethod
    def load_image(
        path: str | Path,
        sensor: SensorType,
        sun_angle_deg: float,
        resolution_m: float,
    ) -> LunarImage:
        import cv2
        import numpy as np

        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        if data.ndim == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return LunarImage(
            data=data.astype(np.float32),
            sensor=sensor,
            sun_angle_deg=sun_angle_deg,
            resolution_m=resolution_m,
        )
