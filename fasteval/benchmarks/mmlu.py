from datasets import load_dataset

from fasteval.base import Benchmark, BenchmarkResult

class MMLUBenchmark(Benchmark):
    def __init__(self, num_samples: int | None = None):
        self._num_samples = num_samples
        self._data: list[dict] | None = None

    @property
    def name(self) -> str:
        return "mmlu"

    def load(self) -> None:
        dataset = load_dataset("cais/mmlu", "all", split="test")
        if self._num_samples is not None:
            dataset = dataset.select(range(min(self._num_samples, len(dataset))))
        self._data = list(dataset)

    def build_prompts(self) -> list[str]:
        assert self._data is not None, "call load() first"
        prompts: list[str] = []
        for item in self._data:
            question = item["question"]
            choices = item["choices"]
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += "Answer:"
            prompts.append(prompt)
        return prompts

    def score(self, responses: list[str]) -> BenchmarkResult:
        assert self._data is not None, "call load() first"
        assert len(responses) == len(self._data), (
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
        text = text.strip().upper()
        for letter in ["A", "B", "C", "D"]:
            if letter in text:
                return letter
        return ""

    @staticmethod
    def _extract_expected(answer) -> str:
        if isinstance(answer, int):
            return chr(65 + answer)
        return str(answer).strip().upper()
