import json
from pathlib import Path

from vllm import LLM, SamplingParams

from fasteval.base import Backend
from fasteval.progress import ProgressBar


class VLLMBackend(Backend):
    def __init__(
        self, model: str, batch_size: int | str = 1, quiet: bool = False, **llm_kwargs
    ):
        self._model_name = model
        self._quiet = quiet
        self._llm = LLM(
            model=model, **{k: v for k, v in llm_kwargs.items() if v is not None}
        )
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=400,
        )
        self._batch_size = batch_size
        self._cache_dir = Path(".fasteval_cache") / self._sanitize_model_name(model)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        return name.replace("/", "__").replace(":", "_")

    def generate(self, prompts: list[str], batch_size: int) -> list[str]:
        results: list[str] = []
        pbar = (
            ProgressBar(total=len(prompts), desc=self._model_name)
            if not self._quiet
            else None
        )
        if pbar:
            pbar.__enter__()
        try:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                outputs = self._llm.generate(batch, self._sampling_params)
                for output in outputs:
                    results.append(output.outputs[0].text.strip())
                if pbar:
                    pbar.update(len(batch))
        finally:
            if pbar:
                pbar.__exit__(None, None, None)
        return results

    def score_answers(
        self, prompts: list[str], batch_size: int, choices: list[list[str]]
    ) -> list[int]:
        logprob_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=10,
        )
        tokenizer = self._llm.get_tokenizer()
        results: list[int] = []
        pbar = (
            ProgressBar(total=len(prompts), desc=f"{self._model_name} (logprobs)")
            if not self._quiet
            else None
        )
        if pbar:
            pbar.__enter__()
        try:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                batch_choices = choices[i : i + batch_size]
                outputs = self._llm.generate(batch, logprob_params)
                for output, candidates in zip(outputs, batch_choices):
                    logprobs_map = output.outputs[0].logprobs[0]
                    best_idx = 0
                    best_lp = float("-inf")
                    for j, c in enumerate(candidates):
                        tid = tokenizer.encode(c, add_special_tokens=False)
                        if len(tid) != 1:
                            continue
                        if tid[0] in logprobs_map:
                            lp = logprobs_map[tid[0]].logprob
                            if lp > best_lp:
                                best_lp = lp
                                best_idx = j
                    results.append(best_idx)
                if pbar:
                    pbar.update(len(batch))
        finally:
            if pbar:
                pbar.__exit__(None, None, None)
        return results

    def auto_batch(self) -> int:
        cache_file = self._cache_dir / "batch_size.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["batch_size"]

        try:
            import torch
        except ImportError:
            return 1

        if not torch.cuda.is_available():
            return 1

        probe = ["What is 2+3?"] * 128
        last_ok = 1

        for size in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                self._llm.generate(probe[:size], self._sampling_params)
                last_ok = size
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA out of memory" in str(e).lower():
                    break
                raise

        cache_file.write_text(
            json.dumps({"batch_size": last_ok, "model": self._model_name}, indent=2)
        )
        return last_ok

    @property
    def batch_size(self) -> int:
        if self._batch_size == "auto" or self._batch_size is None:
            return self.auto_batch()
        return int(self._batch_size)
