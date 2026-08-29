"""Design patterns used across the registration pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Strategy Pattern — swappable algorithms (features, matchers, estimators)
# ---------------------------------------------------------------------------

class Strategy(ABC, Generic[T, R]):
    """Base strategy interface."""

    @abstractmethod
    def execute(self, context: T) -> R:
        ...


# ---------------------------------------------------------------------------
# Pipeline Pattern — sequential stage execution
# ---------------------------------------------------------------------------

class PipelineStage(ABC, Generic[T]):
    """Single stage in the registration pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, context: T) -> T:
        ...


class Pipeline(Generic[T]):
    """Orchestrates ordered stage execution with optional observers."""

    def __init__(self, stages: list[PipelineStage[T]] | None = None):
        self._stages: list[PipelineStage[T]] = stages or []
        self._observers: list[Callable[[str, T], None]] = []

    def add_stage(self, stage: PipelineStage[T]) -> Pipeline:
        self._stages.append(stage)
        return self

    def attach(self, observer: Callable[[str, T], None]) -> None:
        self._observers.append(observer)

    def run(self, context: T) -> T:
        for stage in self._stages:
            for obs in self._observers:
                obs(f"stage_start:{stage.name}", context)
            context = stage.run(context)
            for obs in self._observers:
                obs(f"stage_end:{stage.name}", context)
        return context


# ---------------------------------------------------------------------------
# Factory Pattern — create components by sensor / config
# ---------------------------------------------------------------------------

class ComponentFactory(ABC):
    @abstractmethod
    def create(self, component_type: str, **kwargs: Any) -> Any:
        ...


# ---------------------------------------------------------------------------
# Observer Pattern — metrics / logging hooks
# ---------------------------------------------------------------------------

class PipelineObserver(ABC):
    @abstractmethod
    def on_event(self, event: str, payload: Any) -> None:
        ...


class MetricsObserver(PipelineObserver):
    """Collects stage timing and counts."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_event(self, event: str, payload: Any) -> None:
        self.events.append((event, payload))

    def stage_names(self) -> list[str]:
        return [e[0] for e in self.events if e[0].startswith("stage_end:")]


# ---------------------------------------------------------------------------
# Chain of Responsibility — RAG context enrichment
# ---------------------------------------------------------------------------

class ContextHandler(ABC):
    def __init__(self) -> None:
        self._next: ContextHandler | None = None

    def set_next(self, handler: ContextHandler) -> ContextHandler:
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, query: dict[str, Any]) -> dict[str, Any]:
        if self._next:
            return self._next.handle(query)
        return query


# ---------------------------------------------------------------------------
# Template Method — shared stage skeleton
# ---------------------------------------------------------------------------

class BaseRegistrationStage(PipelineStage[T], ABC):
    """Template method: validate → process → annotate context."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def run(self, context: T) -> T:
        self.validate(context)
        result = self.process(context)
        return self.annotate(result)

    def validate(self, context: T) -> None:
        pass

    @abstractmethod
    def process(self, context: T) -> T:
        ...

    def annotate(self, context: T) -> T:
        return context
