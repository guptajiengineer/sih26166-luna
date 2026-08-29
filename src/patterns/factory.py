"""Factory for feature extractors, matchers, and geometric estimators."""

from __future__ import annotations

from typing import Any

from src.features.contour import ContourFeatureExtractor
from src.features.deep_embeddings import DeepEmbeddingExtractor
from src.features.phase_congruency import PhaseCongruencyExtractor
from src.geometry.graph_matching import GraphMatchingFilter
from src.geometry.magsac import MAGSACEstimator
from src.geometry.spatial_consistency import SpatialConsistencyFilter
from src.matching.dense_matcher import DenseMatcher
from src.matching.semi_dense_matcher import SemiDenseMatcher
from src.patterns import ComponentFactory


class RegistrationComponentFactory(ComponentFactory):
    """Creates pipeline components from configuration strings."""

    _FEATURES = {
        "contour": ContourFeatureExtractor,
        "phase_congruency": PhaseCongruencyExtractor,
        "deep_embedding": DeepEmbeddingExtractor,
    }

    _MATCHERS = {
        "dense": DenseMatcher,
        "semi_dense": SemiDenseMatcher,
    }

    _ESTIMATORS = {
        "magsac": MAGSACEstimator,
        "graph_matching": GraphMatchingFilter,
        "spatial_consistency": SpatialConsistencyFilter,
    }

    def create(self, component_type: str, **kwargs: Any) -> Any:
        registry = {
            "feature": self._FEATURES,
            "matcher": self._MATCHERS,
            "estimator": self._ESTIMATORS,
        }
        kind = kwargs.pop("kind", component_type)
        name = kwargs.pop("name", component_type)
        mapping = registry.get(kind, {})
        if name not in mapping:
            raise ValueError(f"Unknown {kind}: {name}. Available: {list(mapping)}")
        return mapping[name](**kwargs)

    def create_feature_extractors(self, names: list[str]) -> list[Any]:
        return [self.create("feature", kind="feature", name=n) for n in names]

    def create_matcher(self, name: str) -> Any:
        return self.create("matcher", kind="matcher", name=name)

    def create_estimator(self, name: str) -> Any:
        return self.create("estimator", kind="estimator", name=name)
