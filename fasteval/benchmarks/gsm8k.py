import math
import re

from datasets import load_dataset

from fasteval.base import Benchmark, BenchmarkResult

_HASH_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)")
_NUMBER_RE = re.compile(r"(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)")


def _parse_number(raw: str) -> float | None:
    cleaned = raw.replace(",", "").strip()
    if not cleaned or cleaned in ("-", ".", "-.", "e", "E"):
        return None
    try:
        v = float(cleaned)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


class GSM8KBenchmark(Benchmark):
    def __init__(
        self,
        split: str = "test",
        num_samples: int | None = None,
        num_shots: int = 8,
    ):
        if num_samples is not None and num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_shots < 0:
            raise ValueError(f"num_shots must be >= 0, got {num_shots}")
        self._split = split
        self._num_samples = num_samples
        self._num_shots = num_shots
        self._data: list[dict] | None = None
        self._exemplars: list[dict] | None = None

    @property
    def name(self) -> str:
        return "gsm8k"

    def load(self) -> None:
        if self._split not in ("test", "train"):
            raise ValueError(f"unknown GSM8K split: {self._split!r} (expected 'test'|'train')")
        if self._num_samples == 0:
            self._data = []
        else:
            dataset = load_dataset("openai/gsm8k", "main", split=self._split, trust_remote_code=False)
            if self._num_samples is not None:
                dataset = dataset.select(
                    range(min(self._num_samples, len(dataset)))
                )
            self._data = list(dataset)

        if self._num_shots == 0:
            self._exemplars = []
        else:
            # Load only the first num_shots training examples, not the whole split.
            train = load_dataset(
                "openai/gsm8k", "main", split=f"train[:{self._num_shots}]",
                trust_remote_code=False,
            )
            self._exemplars = [
                self._parse_example(train[i])
                for i in range(min(self._num_shots, len(train)))
            ]

    @staticmethod
    def _parse_example(item: dict) -> dict:
        clean = re.sub(r"<<[^>]*>>", "", item["answer"]).strip()
        solution = re.sub(r"####.*", "", clean).strip()
        matches = list(_HASH_RE.finditer(item["answer"]))
        answer = _parse_number(matches[-1].group(1)) if matches else None
        return {
            "question": item["question"],
            "solution": solution,
            "answer": answer,
        }

    def build_prompts(self) -> list[str]:
        if self._data is None or self._exemplars is None:
            raise RuntimeError("call load() first")

        exemplar_parts: list[str] = []
        for ex in self._exemplars:
            if ex["answer"] is None:
                continue
            try:
                as_int = int(ex["answer"])
                answer_str = str(as_int) if ex["answer"] == as_int else str(ex["answer"])
            except (ValueError, OverflowError):
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
        if self._data is None:
            raise RuntimeError("call load() first")
        if len(responses) != len(self._data):
            raise ValueError(
                f"expected {len(self._data)} responses, got {len(responses)}"
            )

        correct = 0
        total = len(responses)

        for response, item in zip(responses, self._data):
            predicted = self._extract_answer(response)
            expected = self._extract_expected(item["answer"])
            if (
                predicted is not None
                and expected is not None
                and math.isclose(predicted, expected, rel_tol=1e-9, abs_tol=1e-6)
            ):
                correct += 1

        return BenchmarkResult(
            name=self.name,
            model="",
            accuracy=correct / total if total > 0 else 0.0,
            correct=correct,
            total=total,
            time_seconds=0.0,
            config={"split": self._split, "num_shots": self._num_shots},
        )

    @staticmethod
    def _extract_answer(text: str) -> float | None:
        if not text:
            return None
        # Prefer the last #### (the answer), not the first (exemplar echo).
        matches = list(_HASH_RE.finditer(text))
        if matches:
            return _parse_number(matches[-1].group(1))
        # Fallback: last number in the text (instruct models often skip ####).
        fallback = list(_NUMBER_RE.finditer(text))
        if fallback:
            return _parse_number(fallback[-1].group(1))
        return None

    @staticmethod
    def _extract_expected(answer_text: str) -> float | None:
        if not answer_text:
            return None
        matches = list(_HASH_RE.finditer(answer_text))
        if matches:
            return _parse_number(matches[-1].group(1))
        return None
