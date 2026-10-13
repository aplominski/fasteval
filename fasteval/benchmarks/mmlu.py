from datasets import load_dataset

from fasteval.base import Benchmark, BenchmarkResult
import re

_EXPECTED_RE = re.compile(r"\b([A-D])\b")


class MMLUBenchmark(Benchmark):
    def __init__(self, num_samples: int | None = None):
        if num_samples is not None and num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        self._num_samples = num_samples
        self._data: list[dict] | None = None

    @property
    def name(self) -> str:
        return "mmlu"

    @property
    def mode(self) -> str:
        return "onepass"

    def load(self) -> None:
        if self._num_samples == 0:
            self._data = []
            return
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=False)
        if self._num_samples is not None:
            dataset = dataset.select(range(min(self._num_samples, len(dataset))))
        self._data = list(dataset)

    def build_prompts(self) -> list[str]:
        if self._data is None:
            raise RuntimeError("call load() first")
        prompts: list[str] = []
        for item in self._data:
            question = item["question"]
            choices = item["choices"]
            if len(choices) != 4:
                raise ValueError(f"expected 4 choices, got {len(choices)}")
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += "Answer: "
            prompts.append(prompt)
        return prompts

    def score(self, responses: list[str]) -> BenchmarkResult:
        if self._data is None:
            raise RuntimeError("call load() first")
        if len(responses) != len(self._data):
            raise ValueError(
                f"expected {len(self._data)} responses, got {len(responses)}"
            )

        correct = 0
        total = len(responses)

        for resp, item in zip(responses, self._data):
            predicted = self._extract_answer(resp)
            expected = self._extract_expected(item["answer"])
            if predicted and predicted == expected:
                correct += 1

        return BenchmarkResult(
            name=self.name,
            model="",
            accuracy=correct / total if total > 0 else 0.0,
            correct=correct,
            total=total,
            time_seconds=0.0,
        )

    @staticmethod
    def _extract_answer(text: str) -> str:
        if not text or not text.strip():
            return ""
        upper = text.strip().upper()
        # Prefer the letter after ANSWER:, then the last occurrence.
        m = re.search(r"ANSWER\s*[:\-]?\s*([A-D])\b", upper)
        if m:
            return m.group(1)
        matches = _EXPECTED_RE.findall(upper)
        if matches:
            return matches[-1]
        # Fallback: first non-empty A-D letter.
        first = upper.strip()[:1]
        if first in ("A", "B", "C", "D"):
            return first
        return ""

    @staticmethod
    def _extract_expected(answer) -> str:
        if isinstance(answer, bool):
            raise ValueError(f"invalid MMLU answer index: {answer!r}")
        if isinstance(answer, int):
            if not 0 <= answer < 4:
                raise ValueError(f"MMLU answer index out of range 0-3: {answer}")
            return chr(65 + answer)
        return str(answer).strip().upper()
