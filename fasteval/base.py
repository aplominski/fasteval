from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    name: str
    model: str
    accuracy: float
    correct: int
    total: int
    time_seconds: float
    batch_size: int = 0
    config: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    @abstractmethod
    def generate(self, prompts: list[str], batch_size: int) -> list[str]:
        ...

    @abstractmethod
    def auto_batch(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class Benchmark(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def build_prompts(self) -> list[str]:
        ...

    @abstractmethod
    def score(self, responses: list[str]) -> BenchmarkResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
