"""Individual pipeline stages."""

from __future__ import annotations

import cv2
import numpy as np

from src.mpp.metrics import StageTimer
from src.patterns import BaseRegistrationStage
from src.patterns.factory import RegistrationComponentFactory
from src.pipeline.context import PipelineContext
from src.rag.context_builder import build_rag_chain


class RAGParameterStage(BaseRegistrationStage[PipelineContext]):
    """Stage 0: RAG retrieval for adaptive parameter selection."""

    @property
    def name(self) -> str:
        return "rag_parameter_selection"

    def process(self, context: PipelineContext) -> PipelineContext:
        with StageTimer(self.name, context.stage_timings):
            chain = build_rag_chain()
            query = {
                "ref_sensor": context.reference.sensor.value,
                "mov_sensor": context.moving.sensor.value,
                "sun_angle_ref": context.reference.sun_angle_deg,
                "sun_angle_mov": context.moving.sun_angle_deg,
            }
            enriched = chain.handle(query)
            rag_params = enriched.get("rag_params", {})
            context.rag_params = rag_params
            context.config = {**context.config, **{k: v for k, v in rag_params.items() if k != "rag_context"}}
            context.metadata["rag_mean_relevance"] = rag_params.get("rag_mean_relevance", 0.0)
        return context


class PreprocessingStage(BaseRegistrationStage[PipelineContext]):
    """Stage 1: Illumination normalization when sun angles differ."""

    @property
    def name(self) -> str:
        return "preprocessing"

    def process(self, context: PipelineContext) -> PipelineContext:
        with StageTimer(self.name, context.stage_timings):
            if context.config.get("preprocess_illumination"):
                for img_attr in ("reference", "moving"):
                    lunar = getattr(context, img_attr)
                    data = lunar.data.astype(np.float32)
                    if data.ndim == 2:
                        low = cv2.GaussianBlur(data, (0, 0), sigmaX=50)
                        data = data - low + np.mean(data)
                        lunar.data = np.clip(data, 0, None)
        return context


class FeatureExtractionStage(BaseRegistrationStage[PipelineContext]):
    """Stage 2: Extract illumination-invariant keypoints."""

    @property
    def name(self) -> str:
        return "feature_extraction"

    def process(self, context: PipelineContext) -> PipelineContext:
        with StageTimer(self.name, context.stage_timings):
            factory = RegistrationComponentFactory()
            names = context.config.get(
                "feature_extractors",
                ["phase_congruency", "contour"],
            )
            extractors = factory.create_feature_extractors(names)
            params = context.config.get("feature_params", {})

            ref_kps, mov_kps = [], []
            for ext in extractors:
                ref_kps.extend(ext.execute((context.reference, params)))
                mov_kps.extend(ext.execute((context.moving, params)))

            context.ref_keypoints = ref_kps
            context.mov_keypoints = mov_kps
            context.metadata["ref_keypoint_count"] = len(ref_kps)
            context.metadata["mov_keypoint_count"] = len(mov_kps)
        return context


class MatchingStage(BaseRegistrationStage[PipelineContext]):
    """Stage 3: Semi-dense / dense tie-point matching across scale gap."""

    @property
    def name(self) -> str:
        return "matching"

    def process(self, context: PipelineContext) -> PipelineContext:
        with StageTimer(self.name, context.stage_timings):
            factory = RegistrationComponentFactory()
            matcher_name = context.config.get("matcher", "semi_dense")
            matcher = factory.create_matcher(matcher_name)

            match_params = {
                "scale_ref": context.reference.resolution_m,
                "scale_mov": context.moving.resolution_m,
                "ratio_threshold": context.config.get("ratio_threshold", 0.75),
                "ref_image": context.reference.data,
                "mov_image": context.moving.data,
            }
            context.tie_points = matcher.execute(
                (context.ref_keypoints, context.mov_keypoints, match_params)
            )
            context.metadata["raw_match_count"] = len(context.tie_points)
        return context


class GeometricFilteringStage(BaseRegistrationStage[PipelineContext]):
    """Stage 4: Robust outlier rejection (MAGSAC++, graph matching, spatial checks)."""

    @property
    def name(self) -> str:
        return "geometric_filtering"

    def process(self, context: PipelineContext) -> PipelineContext:
        with StageTimer(self.name, context.stage_timings):
            factory = RegistrationComponentFactory()
            est_name = context.config.get("geometric_estimator", "magsac")
            estimator = factory.create_estimator(est_name)

            est_params = {
                "reproj_threshold": context.config.get("reproj_threshold", 2.0),
                "model": context.config.get("transform_model", "homography"),
            }
            inliers, H = estimator.execute((context.tie_points, est_params))

            # Optional secondary spatial filter
            secondary = context.config.get("secondary_filter")
            if secondary:
                sec_est = factory.create_estimator(secondary)
                inliers, H = sec_est.execute((inliers, est_params))

            context.inlier_tie_points = inliers
            context.transform_matrix = H
            context.metadata["inlier_count"] = len(inliers)
        return context


class RegistrationStage(BaseRegistrationStage[PipelineContext]):
    """Stage 5: Apply transform and produce co-registered output."""

    @property
    def name(self) -> str:
        return "registration"

    def process(self, context: PipelineContext) -> PipelineContext:
        with StageTimer(self.name, context.stage_timings):
            from src.models.domain import RegistrationResult
            from src.mpp.metrics import MPPMetrics

            H = context.transform_matrix
            if H is None:
                H = np.eye(3)

            ref_h, ref_w = context.reference.data.shape[:2]
            mov = context.moving.data
            if mov.ndim == 2:
                registered = cv2.warpPerspective(
                    mov.astype(np.float32), H, (ref_w, ref_h), flags=cv2.INTER_LINEAR
                )
            else:
                registered = np.stack([
                    cv2.warpPerspective(mov[:, :, c].astype(np.float32), H, (ref_w, ref_h))
                    for c in range(mov.shape[2])
                ], axis=2)

            context.registered_image = registered

            inliers = context.inlier_tie_points
            rmse = MPPMetrics.registration_rmse(inliers, H)
            inlier_ratio = MPPMetrics.inlier_ratio(context.tie_points)

            context.result = RegistrationResult(
                tie_points=inliers,
                transform_matrix=H,
                registered_image=registered,
                rmse_px=rmse,
                inlier_ratio=inlier_ratio,
                metadata=context.metadata,
            )
        return context
