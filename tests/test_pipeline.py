"""Tests for lunar registration pipeline."""

import numpy as np
import pytest

from src.models.domain import Keypoint, LunarImage, SensorType, TiePoint
from src.mpp.metrics import MPPMetrics
from src.pipeline.orchestrator import LunarRegistrationPipeline
from src.rag.retriever import LunarRegistrationRetriever


def _synthetic_pair(seed: int = 0) -> tuple[LunarImage, LunarImage]:
    rng = np.random.default_rng(seed)
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    terrain = np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2 * 40 ** 2))
    ref = terrain + rng.normal(0, 0.02, (h, w))
    mov = terrain * 0.9 + 0.1 * (x / w) + rng.normal(0, 0.03, (h, w))
    return (
        LunarImage(data=ref.astype(np.float32), sensor=SensorType.OHRC, sun_angle_deg=30, resolution_m=0.25),
        LunarImage(data=mov.astype(np.float32), sensor=SensorType.TMC2, sun_angle_deg=50, resolution_m=5.0),
    )


def test_rag_retriever_keyword_fallback():
    retriever = LunarRegistrationRetriever()
    results = retriever.retrieve("OHRC TMC-2 phase congruency registration", top_k=3)
    assert len(results) > 0
    assert results[0].get("relevance", 0) >= 0


def test_rag_suggest_parameters():
    retriever = LunarRegistrationRetriever()
    params = retriever.suggest_parameters("OHRC", "TMC2", sun_diff=25.0)
    assert "feature_extractors" in params
    assert "matcher" in params
    assert params["rag_mean_relevance"] >= 0


def test_mpp_metrics():
    tie_points = [
        TiePoint(ref_x=10, ref_y=10, mov_x=10.1, mov_y=10.1, inlier=True),
        TiePoint(ref_x=50, ref_y=50, mov_x=50.2, mov_y=49.8, inlier=True),
        TiePoint(ref_x=100, ref_y=100, mov_x=200, mov_y=200, inlier=False),
    ]
    H = np.eye(3)
    rmse = MPPMetrics.registration_rmse(tie_points[:2], H)
    assert rmse < 1.0
    entropy = MPPMetrics.spatial_entropy(tie_points[:2], (256, 256))
    assert entropy >= 0


def test_pipeline_end_to_end():
    ref, mov = _synthetic_pair()
    pipeline = LunarRegistrationPipeline({"pipeline": {"feature_extractors": ["contour", "phase_congruency"]}})
    ctx, mpp = pipeline.register(ref, mov)
    assert ctx.result is not None
    assert mpp.tie_point_count >= 0
    assert ctx.transform_matrix is not None
