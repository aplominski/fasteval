import importlib
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


# Error signatures meaning "this GPU cannot run the default kernel/config"
# (e.g. older GPUs with small shared memory vs a kernel wanting more).
# These appear during engine init / cudagraph capture, before any prompt runs,
# so retrying with a different backend or eager mode is safe. Anything else
# (bad model name, no GPU, real OOM from weights) must fail immediately.
_RECOVERABLE_INIT_PATTERNS = (
    "shared memory",
    "outofresources",
    "out of resources",
    "out of resource",
    "reducing block sizes",
    "cudagraph",
    "cuda graph",
    "capture_model",
    "capture model",
    "profile_cudagraph_memory",
    "engine core initialization failed",
    "enginecore failed to start",
    "failed to start",
)

# Error signatures meaning the chosen attention backend was rejected by this
# environment (missing optional dependency, unsupported platform or model).
# Generic phrases only — no backend names. Such failures move on to the next
# discovered backend instead of aborting. An *unknown* backend name (typo in
# an explicit user setting) is deliberately not included: it must fail fast
# with vLLM's own "valid options" error.
_BACKEND_REJECTED_PATTERNS = (
    "no module named",
    "not supported",
    "not implemented",
    "not available",
    "cannot be used",
    "incompatible",
)

# Upper bound on attention-backend alternatives tried during init fallback.
# Each attempt reloads model weights, so this caps worst-case startup cost.
# A count, not a choice of backend.
_MAX_BACKEND_FALLBACKS = 4


def _resolve_obj_by_qualname(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ImportError(f"invalid class path: {path!r}")
    return getattr(importlib.import_module(module_name), attr)


def _discover_attention_backends(exclude: str | None = None) -> list[str]:
    """List runnable attention backends offered by the installed vLLM.

    Names come from vLLM's own backend registry; every candidate is verified
    by importing its class (cheap, no model load). Entries with no class path
    and ones that fail to import (platform-specific or missing optional
    dependencies on this machine) are skipped. No backend names are hardcoded
    here, so this keeps working with future vLLM releases and third-party
    backends registered at runtime.
    """
    for registry_path in (
        "vllm.v1.attention.backends.registry",
        "vllm.attention.backends.registry",
    ):
        try:
            registry = importlib.import_module(registry_path)
        except Exception:
            continue
        enum_cls = getattr(registry, "AttentionBackendEnum", None)
        if enum_cls is None:
            continue
        names: list[str] = []
        try:
            members = list(enum_cls)
        except Exception:
            return []
        for member in members:
            name = getattr(member, "name", "")
            path = getattr(member, "value", "")
            if not name or not path or name == exclude:
                continue
            try:
                _resolve_obj_by_qualname(path)
            except Exception:
                continue
            names.append(name)
        return names
    return []


def _iter_cause_chain(exc: BaseException):
    seen = set()
    queue = [exc]
    while queue:
        cur = queue.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        for nxt in (getattr(cur, "__cause__", None), getattr(cur, "__context__", None)):
            if isinstance(nxt, BaseException):
                queue.append(nxt)


def _is_backend_rejected_error(exc: BaseException) -> bool:
    for err in _iter_cause_chain(exc):
        text = f"{type(err).__name__}: {err}".lower()
        if any(p in text for p in _BACKEND_REJECTED_PATTERNS):
            return True
    return False


def _is_recoverable_init_error(exc: BaseException) -> bool:
    for err in _iter_cause_chain(exc):
        name = type(err).__name__.lower()
        text = f"{name}: {err}".lower()
        if "outofresources" in name or "outofresources" in text:
            return True
        if any(p in text for p in _RECOVERABLE_INIT_PATTERNS):
            return True
    return False


def _summarize_params(params: dict) -> str:
    interesting = (
        "attention_backend",
        "enforce_eager",
        "max_model_len",
        "enable_prefix_caching",
        "tensor_parallel_size",
    )
    return ", ".join(f"{k}={params[k]!r}" for k in interesting if k in params) or "{}"


class VLLMBackend(Backend):
    def __init__(
        self,
        model: str,
        batch_size: int | str = 1,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        enable_prefix_caching: bool = True,
        max_num_seqs: int | None = None,
        gpu_memory_utilization: float | None = None,
        **llm_kwargs,
    ):
        from vllm import LLM, SamplingParams

        self._model_name = model
        llm_params: dict = {k: v for k, v in llm_kwargs.items() if v is not None}
        # Automatic prefix caching: GSM8K shares a ~2k-token few-shot prefix
        # across all prompts; a single prefill then serves every request.
        llm_params.setdefault("enable_prefix_caching", enable_prefix_caching)
        if max_num_seqs is not None:
            llm_params.setdefault("max_num_seqs", max_num_seqs)
        if gpu_memory_utilization is not None:
            llm_params.setdefault("gpu_memory_utilization", gpu_memory_utilization)
        self._llm, self._llm_init_info = self._init_llm_with_fallback(
            LLM, model, llm_params
        )
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
        # Cache of candidate -> token ids. A/B/C/D variants are constant,
        # so encode once instead of ~16 encodes per prompt in the hot loop.
        self._token_id_cache: dict[str, list[int]] = {}

    @staticmethod
    def _init_attempts(
        base_params: dict, backends: list[str] | None = None
    ) -> list[dict]:
        """Build init configs from most to least demanding.

        Backend alternatives come from vLLM's own registry (see
        _discover_attention_backends), never from a hardcoded list. Later
        attempts trade speed for compatibility: portable attention backend,
        no torch.compile/CUDA graphs, shorter context (smaller KV cache and
        cheaper capture). Duplicates of the requested config are skipped so
        explicit user settings are never retried pointlessly.
        """
        if backends is None:
            backends = _discover_attention_backends(
                exclude=base_params.get("attention_backend")
            )
        attempts = [dict(base_params)]

        def differs(params: dict) -> bool:
            return any(
                params.get(k) != base_params.get(k)
                for k in set(params) | set(base_params)
            ) and params not in attempts

        for name in backends[:_MAX_BACKEND_FALLBACKS]:
            candidate = dict(base_params)
            candidate["attention_backend"] = name
            if differs(candidate):
                attempts.append(candidate)
        eager_base = dict(attempts[-1])
        eager_base["enforce_eager"] = True
        if differs(eager_base):
            attempts.append(eager_base)
        if base_params.get("max_model_len") is None:
            short_ctx = dict(attempts[-1])
            short_ctx["max_model_len"] = 8192
            if differs(short_ctx):
                attempts.append(short_ctx)
        return attempts

    @classmethod
    def _init_llm_with_fallback(cls, LLM, model: str, base_params: dict):
        attempts = cls._init_attempts(base_params)
        requested_backend = base_params.get("attention_backend")
        last_exc: Exception | None = None
        for i, params in enumerate(attempts):
            try:
                llm = LLM(model=model, **params)
            except Exception as e:
                pinned = params.get("attention_backend", requested_backend)
                backend_switched = pinned != requested_backend
                if (
                    _is_backend_rejected_error(e)
                    and backend_switched
                    and i + 1 < len(attempts)
                ):
                    # Environment rejected this alternative backend; the next
                    # discovered one (or the generic fallbacks) may still work.
                    last_exc = e
                    log.warning(
                        "LLM init attempt %d/%d rejected (%s); retrying with %s",
                        i + 1,
                        len(attempts),
                        _summarize_params(params),
                        _summarize_params(attempts[i + 1]),
                    )
                    continue
                if not _is_recoverable_init_error(e) or i + 1 >= len(attempts):
                    raise RuntimeError(
                        f"failed to load model {model!r}: {e}"
                    ) from e
                last_exc = e
                log.warning(
                    "LLM init attempt %d/%d failed (%s); retrying with %s",
                    i + 1,
                    len(attempts),
                    _summarize_params(params),
                    _summarize_params(attempts[i + 1]),
                )
                continue
            info = {
                "attempt": i + 1,
                "attempts": len(attempts),
                "params": _summarize_params(params),
                "fallback_used": i > 0,
            }
            if i > 0:
                log.warning(
                    "LLM started with compatibility fallback (%s)", info["params"]
                )
            return llm, info
        raise RuntimeError(
            f"failed to load model {model!r}: all {len(attempts)} init attempts "
            f"failed (last: {last_exc})"
        ) from last_exc

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
        # Validate for API compatibility, but issue a single llm.generate call:
        # vLLM's continuous batching schedules internally and beats a Python loop.
        self._check_batch(batch_size, len(prompts))
        if not prompts:
            return []
        with ProgressBar(total=len(prompts), desc=self._model_name) as pbar:
            try:
                outputs = self._llm.generate(prompts, self._sampling_params)
            except Exception as e:
                if "out of memory" not in str(e).lower() and "oom" not in str(e).lower():
                    raise
                outputs = self._generate_chunked(prompts, batch_size, self._sampling_params)
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
                )
            results: list[str] = []
            for output in outputs:
                if not getattr(output, "outputs", None):
                    log.warning("empty vLLM output; substituting ''")
                    results.append("")
                    continue
                results.append((output.outputs[0].text or "").strip())
            pbar.update(len(prompts))
        return results

    def _generate_chunked(self, prompts: list[str], batch_size: int, params) -> list:
        """Fallback path: chunked generate used only after an OOM on the full pass."""
        bs = validate_batch_size(batch_size)
        outputs_all: list = []
        for i in range(0, len(prompts), bs):
            outputs_all.extend(self._llm.generate(prompts[i : i + bs], params))
        return outputs_all

    def _candidate_token_ids(self, tokenizer, candidate: str) -> list[int]:
        cached = self._token_id_cache.get(candidate)
        if cached is not None:
            return cached
        ids: list[int] = []
        for variant in (candidate, " " + candidate, candidate.lower(), " " + candidate.lower()):
            try:
                tid = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if len(tid) == 1 and tid[0] not in ids:
                ids.append(tid[0])
        self._token_id_cache[candidate] = ids
        return ids

    def _prewarm_token_cache(self, tokenizer, candidates: list[str] | None = None) -> None:
        for c in candidates or ["A", "B", "C", "D"]:
            self._candidate_token_ids(tokenizer, c)

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
        self._prewarm_token_cache(tokenizer)
        results: list[int | None] = []
        with ProgressBar(
            total=len(prompts), desc=f"{self._model_name} (logprobs)"
        ) as pbar:
            try:
                outputs = self._llm.generate(prompts, self._logprob_params)
            except Exception as e:
                if "out of memory" not in str(e).lower() and "oom" not in str(e).lower():
                    raise
                outputs = self._generate_chunked(prompts, bs, self._logprob_params)
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
                )
            for output, candidates in zip(outputs, choices):
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
            pbar.update(len(prompts))
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
