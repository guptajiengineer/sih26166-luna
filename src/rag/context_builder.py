"""Chain-of-responsibility handlers for RAG context enrichment."""

from __future__ import annotations

from typing import Any

from src.patterns import ContextHandler
from src.rag.retriever import LunarRegistrationRetriever


class SensorContextHandler(ContextHandler):
    def handle(self, query: dict[str, Any]) -> dict[str, Any]:
        ref = query.get("ref_sensor", "OHRC")
        mov = query.get("mov_sensor", "TMC2")
        query["sensor_context"] = f"Reference: {ref}, Moving: {mov}, scale gap registration"
        return super().handle(query)


class SunAngleContextHandler(ContextHandler):
    def handle(self, query: dict[str, Any]) -> dict[str, Any]:
        diff = abs(query.get("sun_angle_ref", 0) - query.get("sun_angle_mov", 0))
        query["sun_diff"] = diff
        query["illumination_note"] = (
            "High sun angle difference — prioritize phase congruency"
            if diff > 25 else "Moderate sun angle difference"
        )
        return super().handle(query)


class RAGRetrievalHandler(ContextHandler):
    def __init__(self, retriever: LunarRegistrationRetriever | None = None) -> None:
        super().__init__()
        self.retriever = retriever or LunarRegistrationRetriever()

    def handle(self, query: dict[str, Any]) -> dict[str, Any]:
        params = self.retriever.suggest_parameters(
            ref_sensor=query.get("ref_sensor", "OHRC"),
            mov_sensor=query.get("mov_sensor", "TMC2"),
            sun_diff=query.get("sun_diff", 0),
        )
        query["rag_params"] = params
        return super().handle(query)


def build_rag_chain(retriever: LunarRegistrationRetriever | None = None) -> ContextHandler:
    sensor = SensorContextHandler()
    sun = SunAngleContextHandler()
    rag = RAGRetrievalHandler(retriever)
    sensor.set_next(sun).set_next(rag)
    return sensor
