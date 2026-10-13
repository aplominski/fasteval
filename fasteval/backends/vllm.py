import json
import logging
import re
import tempfile
from pathlib import Path

from fasteval.base import Backend
from fasteval.progress import ProgressBar

log = logging.getLogger(__name__)


def validate_batch_size(value) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid batch size: {value!r}") from e
    if iv <= 0:
        raise ValueError(f"batch size must be > 0, got {iv}")
    return iv


class VLLMBackend(Backend):
    def __init__(
        self,
        model: str,
        batch_size: int | str = 1,
        max_tokens: int = 400,
        temperature: float = 0.0,
        seed: int | None = None,
        **llm_kwargs,
    ):
        from vllm import LLM, SamplingParams

        self._model_name = model
        try:
            self._llm = LLM(
                model=model, **{k: v for k, v in llm_kwargs.items() if v is not None}
            )
        except Exception as e:
            raise RuntimeError(f"failed to load model {model!r}: {e}") from e
        self._SamplingParams = SamplingParams
        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["Question:", "\n\nQuestion:"],
            seed=seed,
        )
        self._logprob_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=5,
        )
        if batch_size != "auto" and batch_size is not None:
            validate_batch_size(batch_size)
        self._batch_size = batch_size
        self._cache_dir = Path(".fasteval_cache") / self._sanitize_model_name(model)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        # Strip unsafe characters; block traversal, keep readability.
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace("/", "__").replace(":", "_"))
        safe = safe.strip("._") or "model"
        if len(safe) > 120:
            import hashlib
            digest = hashlib.sha256(name.encode()).hexdigest()[:16]
            safe = safe[:100] + "_" + digest
        return safe

    def _check_batch(self, batch_size: int, n_prompts: int) -> int:
        bs = validate_batch_size(batch_size)
        if n_prompts == 0:
            return bs
        return bs

    def generate(self, prompts: list[str], batch_size: int) -> list[str]:
        bs = self._check_batch(batch_size, len(prompts))
        results: list[str] = []
        with ProgressBar(total=len(prompts), desc=self._model_name) as pbar:
            for i in range(0, len(prompts), bs):
                batch = prompts[i : i + bs]
                outputs = self._llm.generate(batch, self._sampling_params)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts"
                    )
                for output in outputs:
                    if not getattr(output, "outputs", None):
                        log.warning("empty vLLM output; substituting ''")
                        results.append("")
                        continue
                    results.append((output.outputs[0].text or "").strip())
                pbar.update(len(batch))
        return results

    def _candidate_token_ids(self, tokenizer, candidate: str) -> list[int]:
        ids: list[int] = []
        for variant in (candidate, " " + candidate, candidate.lower(), " " + candidate.lower()):
            try:
                tid = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if len(tid) == 1 and tid[0] not in ids:
                ids.append(tid[0])
        return ids

    def score_answers(
        self, prompts: list[str], batch_size: int, choices: list[list[str]]
    ) -> list[int | None]:
        if len(prompts) != len(choices):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(choices)={len(choices)}"
            )
        bs = self._check_batch(batch_size, len(prompts))
        try:
            tokenizer = self._llm.get_tokenizer()
        except AttributeError:
            tokenizer = self._llm.llm_engine.tokenizer  # type: ignore[attr-defined]
        results: list[int | None] = []
        with ProgressBar(
            total=len(prompts), desc=f"{self._model_name} (logprobs)"
        ) as pbar:
            for i in range(0, len(prompts), bs):
                batch = prompts[i : i + bs]
                batch_choices = choices[i : i + bs]
                outputs = self._llm.generate(batch, self._logprob_params)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts"
                    )
                for output, candidates in zip(outputs, batch_choices):
                    if not getattr(output, "outputs", None):
                        results.append(None)
                        continue
                    lp_list = getattr(output.outputs[0], "logprobs", None)
                    if not lp_list or lp_list[0] is None:
                        results.append(None)
                        continue
                    logprobs_map = lp_list[0]
                    best_idx: int | None = None
                    best_lp = float("-inf")
                    for j, c in enumerate(candidates):
                        for tid in self._candidate_token_ids(tokenizer, c):
                            entry = logprobs_map.get(tid)
                            if entry is None:
                                continue
                            lp = getattr(entry, "logprob", entry)
                            if isinstance(lp, dict):
                                lp = lp.get("logprob", float("-inf"))
                            if lp > best_lp:
                                best_lp = lp
                                best_idx = j
                    if best_idx is None:
                        log.warning("no candidate token in top-k logprobs; returning None")
                    results.append(best_idx)
                pbar.update(len(batch))
        return results

    def auto_batch(self, probe_prompts: list[str] | None = None) -> int:
        cache_file = self._cache_dir / "batch_size.json"
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                cached = validate_batch_size(payload["batch_size"])
                return cached
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
                log.warning("ignoring corrupt batch cache %s: %s", cache_file, e)

        try:
            import torch
        except ImportError:
            return 1

        if not torch.cuda.is_available():
            return 1

        # The probe should resemble real prompts; short and cheap by default.
        probe = probe_prompts or ["What is 2+3? Answer:"] * 32
        probe_params = self._SamplingParams(temperature=0, max_tokens=1)
        last_ok = 0

        for size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if size > len(probe):
                break
            try:
                self._llm.generate(probe[:size], probe_params)
                last_ok = size
            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg or "oom" in msg:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    break
                raise
            finally:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        if last_ok <= 0:
            raise RuntimeError("auto_batch: even batch_size=1 OOMs; cannot proceed")

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._cache_dir), prefix="batch_size.", suffix=".tmp"
        )
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                json.dump({"batch_size": last_ok, "model": self._model_name}, f, indent=2)
            Path(tmp_path).replace(cache_file)
        except OSError:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
        return last_ok

    @property
    def batch_size(self) -> int:
        if self._batch_size == "auto" or self._batch_size is None:
            return self.auto_batch()
        return validate_batch_size(self._batch_size)

    def close(self) -> None:
        llm, self._llm = getattr(self, "_llm", None), None
        if llm is not None:
            try:
                if hasattr(llm, "llm_engine"):
                    llm.llm_engine.shutdown()  # type: ignore[attr-defined]
            except Exception:
                pass

    def __del__(self):  # defensive, call is not guaranteed
        try:
            self.close()
        except Exception:
            pass
