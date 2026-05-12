import re

from datasets import load_dataset

from fasteval.base import Benchmark, BenchmarkResult


class GSM8KBenchmark(Benchmark):
    def __init__(
        self,
        split: str = "test",
        num_samples: int | None = None,
        num_shots: int = 8,
    ):
        self._split = split
        self._num_samples = num_samples
        self._num_shots = num_shots
        self._data: list[dict] | None = None
        self._exemplars: list[dict] | None = None

    @property
    def name(self) -> str:
        return "gsm8k"

    def load(self) -> None:
        dataset = load_dataset("openai/gsm8k", "main", split=self._split)
        if self._num_samples is not None:
            dataset = dataset.select(
                range(min(self._num_samples, len(dataset)))
            )
        self._data = list(dataset)

        train = load_dataset("openai/gsm8k", "main", split="train")
        self._exemplars = [
            self._parse_example(train[i])
            for i in range(min(self._num_shots, len(train)))
        ]

    @staticmethod
    def _parse_example(item: dict) -> dict:
        clean = re.sub(r"<<[^>]*>>", "", item["answer"]).strip()
        solution = re.sub(r"####.*", "", clean).strip()
        match = re.search(r"####\s*(-?\d+\.?\d*)", item["answer"])
        answer = float(match.group(1)) if match else None
        return {
            "question": item["question"],
            "solution": solution,
            "answer": answer,
        }

    def build_prompts(self) -> list[str]:
        assert self._data is not None, "call load() first"
        assert self._exemplars is not None

        exemplar_parts: list[str] = []
        for ex in self._exemplars:
            if ex["answer"] is None:
                continue
            if ex["answer"] == int(ex["answer"]):
                answer_str = str(int(ex["answer"]))
            else:
                answer_str = str(ex["answer"])
            exemplar_parts.append(
                f"Question: {ex['question']}\nAnswer: {ex['solution']}\n#### {answer_str}"
            )
        exemplar_text = "\n\n".join(exemplar_parts)

        prompts: list[str] = []
        for item in self._data:
            prompts.append(
                f"{exemplar_text}\n\nQuestion: {item['question']}\nAnswer:"
            )
        return prompts

    def score(self, responses: list[str]) -> BenchmarkResult:
        assert self._data is not None, "call load() first"
        assert len(responses) == len(self._data), (
            f"expected {len(self._data)} responses, got {len(responses)}"
        )

        correct = 0
        total = len(responses)

        for response, item in zip(responses, self._data):
            predicted = self._extract_answer(response)
            expected = self._extract_expected(item["answer"])
            if predicted is not None and expected is not None and predicted == expected:
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
    def _extract_answer(text: str) -> float | None:
        match = re.search(r"####\s*(-?\d+\.?\d*)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_expected(answer_text: str) -> float | None:
        match = re.search(r"####\s*(-?\d+\.?\d*)", answer_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
