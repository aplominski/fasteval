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

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError(f"total must be >= 0, got {self.total}")
        if self.correct < 0:
            raise ValueError(f"correct must be >= 0, got {self.correct}")
        if self.correct > self.total:
            raise ValueError(
                f"correct ({self.correct}) cannot exceed total ({self.total})"
            )
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {self.accuracy}")
        if self.time_seconds < 0:
            raise ValueError(
                f"time_seconds must be >= 0, got {self.time_seconds}"
            )
        if self.batch_size < 0:
            raise ValueError(
                f"batch_size must be >= 0, got {self.batch_size}"
            )


class Backend(ABC):
    @abstractmethod
    def generate(self, prompts: list[str], batch_size: int) -> list[str]: ...

    def score_answers(
        self, prompts: list[str], batch_size: int, choices: list[list[str]]
    ) -> list[int | None]:
        if len(prompts) != len(choices):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(choices)={len(choices)}"
            )
        responses = self.generate(prompts, batch_size)
        if len(responses) != len(prompts):
            raise ValueError(
                f"generate() returned {len(responses)} responses "
                f"for {len(prompts)} prompts"
            )
        results: list[int | None] = []
        for resp, candidates in zip(responses, choices):
            results.append(self._match_answer(resp, candidates))
        return results

    @staticmethod
    def _match_answer(text: str, candidates: list[str]) -> int | None:
        """Match response against candidates on word boundaries.

        Returns index or None when nothing matches (never defaults to 0).
        """
        import re

        if not candidates:
            return None
        if text is None:
            return None
        text_clean = text.strip()
        if not text_clean:
            return None
        for i, c in enumerate(candidates):
            if not c:
                continue
            # whole-token match, case-insensitive (avoids "A" in "ANSWER")
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", text_clean, re.IGNORECASE):
                return i
        return None

    @abstractmethod
    def auto_batch(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class Benchmark(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def build_prompts(self) -> list[str]: ...

    @abstractmethod
    def score(self, responses: list[str]) -> BenchmarkResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def mode(self) -> str:
        """Inference mode: 'fullgen' (free generation) or 'onepass' (logprobs)."""
        return "fullgen"


__all__ = ["Backend", "Benchmark", "BenchmarkResult"]
