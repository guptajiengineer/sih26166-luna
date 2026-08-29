"""Graph-based matching consistency filter."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from src.geometry.base import GeometricEstimator, _to_point_arrays
from src.models.domain import TiePoint


class GraphMatchingFilter(GeometricEstimator):
    """
    Builds a consistency graph: nodes = matches, edges = geometric agreement.
    Keeps largest clique as high-confidence inliers.
    """

    @property
    def name(self) -> str:
        return "graph_matching"

    def estimate(
        self, tie_points: list[TiePoint], params: dict[str, Any]
    ) -> tuple[list[TiePoint], np.ndarray]:
        if len(tie_points) < 4:
            return tie_points, np.eye(3)

        tol = params.get("distance_tolerance", 5.0)
        src, dst = _to_point_arrays(tie_points)
        n = len(tie_points)

        G = nx.Graph()
        for i in range(n):
            G.add_node(i)

        for i in range(n):
            for j in range(i + 1, n):
                ds = np.linalg.norm(src[i] - src[j])
                dd = np.linalg.norm(dst[i] - dst[j])
                if abs(ds - dd) < tol:
                    G.add_edge(i, j)

        cliques = list(nx.find_cliques(G))
        if not cliques:
            return tie_points, np.eye(3)

        best = max(cliques, key=len)
        inlier_set = set(best)

        filtered: list[TiePoint] = []
        for idx, tp in enumerate(tie_points):
            tp.inlier = idx in inlier_set
            if tp.inlier:
                filtered.append(tp)

        # Delegate final transform to affine fit on inliers
        from src.geometry.magsac import MAGSACEstimator
        return MAGSACEstimator().estimate(filtered, {**params, "model": "affine"})
