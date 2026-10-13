import argparse
import importlib
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from pathlib import Path


BENCHMARK_REGISTRY: dict[str, tuple[str, str]] = {
    "gsm8k": ("benchmarks.gsm8k", "GSM8KBenchmark"),
    "mmlu": ("benchmarks.mmlu", "MMLUBenchmark"),
}


def _resolve(submodule: str, name: str):
    try:
        mod = importlib.import_module(f"fasteval.{submodule}")
    except ModuleNotFoundError as e:
        raise ImportError(f"cannot resolve fasteval.{submodule}: {e}") from e
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise ImportError(f"{submodule} has no attribute {name!r}: {e}") from e


class WorkerError(RuntimeError):
    pass


def validate_device(device: str) -> str:
    m = re.fullmatch(r"cuda:(\d+)", device.strip().lower())
    if not m:
        raise ValueError(f"invalid device {device!r}, expected e.g. 'cuda:0'")
    return f"cuda:{int(m.group(1))}"


def split_prompts(prompts: list[str], n: int) -> list[list[str]]:
    if n <= 0:
        raise ValueError(f"n devices must be > 0, got {n}")
    # Contiguous chunks: better vLLM prefix-caching than round-robin i::n.
    if not prompts:
        return [[] for _ in range(n)]
    size = (len(prompts) + n - 1) // n
    return [prompts[i * size:(i + 1) * size] for i in range(n)]


def merge_results(ordered: list[list[str] | None], total: int) -> list[str]:
    merged: list[str] = []
    for idx, chunk in enumerate(ordered):
        if chunk is None:
            raise RuntimeError(f"worker {idx} returned no results (crash/OOM?)")
        merged.extend(chunk)
    if len(merged) != total:
        raise RuntimeError(f"expected {total} responses, got {len(merged)}")
    if any(m is None for m in merged):
        raise RuntimeError("missing responses after merge")
    return merged


def _worker(
    device: str,
    prompts: list[str],
    model: str,
    batch: str,
    tensor_parallel_size: int,
    attention_backend: str | None,
    queue: mp.Queue,
    worker_id: int,
    mode: str = "fullgen",
):
    try:
        dev = validate_device(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = dev.split(":")[1]
        VLLMBackend = _resolve("backends.vllm", "VLLMBackend")
        backend = VLLMBackend(
            model=model,
            batch_size=batch,
            tensor_parallel_size=tensor_parallel_size,
            attention_backend=attention_backend,
        )
        try:
            bs = backend.batch_size
            if mode == "onepass":
                choices = [["A", "B", "C", "D"] for _ in range(len(prompts))]
                answer_indices = backend.score_answers(prompts, bs, choices)
                # None (missing from top-k) -> empty answer, benchmark counts it as wrong.
                responses = [chr(65 + idx) if idx is not None else "" for idx in answer_indices]
            elif mode == "fullgen":
                responses = backend.generate(prompts, batch_size=bs)
            else:
                raise ValueError(f"unknown mode: {mode!r}")
            queue.put((worker_id, None, responses))
        finally:
            try:
                backend.close()
            except Exception:
                pass
    except BaseException as e:  # also KeyboardInterrupt/SystemExit -> parent never hangs
        import traceback
        queue.put((worker_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}", None))


def run_distributed(
    prompts: list[str],
    model: str,
    devices: list[str],
    batch: str,
    tensor_parallel_size: int,
    attention_backend: str | None = None,
    mode: str = "fullgen",
    join_timeout: float | None = None,
) -> list[str]:
    devices = [validate_device(d) for d in devices]
    n = len(devices)
    if n == 0:
        raise ValueError("--devices requires at least one device")
    if mode not in ("fullgen", "onepass"):
        raise ValueError(f"unknown mode: {mode!r}")
    chunks = split_prompts(prompts, n)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = []
    for i, device in enumerate(devices):
        p = ctx.Process(
            target=_worker,
            args=(
                device,
                chunks[i],
                model,
                batch,
                tensor_parallel_size,
                attention_backend,
                queue,
                i,
                mode,
            ),
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join(timeout=join_timeout)
            if p.is_alive():
                raise WorkerError("worker hung (join timeout); terminated")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()

    ordered: list[list[str] | None] = [None] * n
    errors: dict[str, str] = {}
    for _ in range(n):
        try:
            idx, err, responses = queue.get(timeout=60)
        except Exception as e:
            raise WorkerError(f"missing result from worker: {e}") from e
        if err:
            errors[devices[idx]] = err
        else:
            ordered[idx] = responses

    for p in processes:
        if p.exitcode not in (0, None) and ordered is not None:
            pass  # exit codes already reflected in missing/failed results
    if errors:
        for dev, err in errors.items():
            print(f"FAILED {dev}: {err}", file=sys.stderr)
        raise WorkerError(f"{len(errors)} worker(s) failed: {', '.join(errors)}")

    # Detect silent deaths (segfault/OOM-killer without put).
    for i, p in enumerate(processes):
        if ordered[i] is None and p.exitcode not in (0, None):
            raise WorkerError(f"worker {devices[i]} died with exitcode {p.exitcode}")

    return merge_results(ordered, len(prompts))


def _run_benchmark(benchmark, backend, prompts, batch_size, mode: str):
    if mode == "onepass":
        choices = [["A", "B", "C", "D"] for _ in range(len(prompts))]
        answer_indices = backend.score_answers(prompts, batch_size, choices)
        responses = [chr(65 + idx) if idx is not None else "" for idx in answer_indices]
    elif mode == "fullgen":
        responses = backend.generate(prompts, batch_size)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return benchmark.score(responses)


def _validate_args(args) -> None:
    if args.batch != "auto":
        try:
            iv = int(args.batch)
        except ValueError as e:
            raise ValueError(f"--batch must be 'auto' or a positive int, got {args.batch!r}") from e
        if iv <= 0:
            raise ValueError(f"--batch must be > 0, got {iv}")
    if args.samples is not None and args.samples < 0:
        raise ValueError(f"--samples must be >= 0, got {args.samples}")
    if args.tensor_parallel_size is None or args.tensor_parallel_size <= 0:
        raise ValueError(
            f"--tensor-parallel-size must be > 0, got {args.tensor_parallel_size}"
        )
    if args.devices is not None:
        if len(args.devices) == 0:
            raise ValueError("--devices requires at least one device")
        for d in args.devices:
            validate_device(d)


def _save_result(result, model: str, devices, batch_size, output_dir: Path) -> Path:
    from fasteval.backends.vllm import VLLMBackend
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = VLLMBackend._sanitize_model_name(model)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"{result.name}__{safe_name}__{timestamp}.json"
    payload = {
        "benchmark": result.name,
        "model": result.model,
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total,
        "time_seconds": result.time_seconds,
        "devices": devices,
        "batch_size": getattr(result, "batch_size", batch_size),
        "config": getattr(result, "config", {}),
    }
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(output_dir), prefix=result_path.name + ".", suffix=".tmp"
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        Path(tmp_path).replace(result_path)
    except OSError:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return result_path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="fasteval",
        description="Fast LLM evaluation framework",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name (e.g. meta-llama/Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "benchmark", help="Comma-separated benchmarks (e.g. gsm8k,mmlu)"
    )
    parser.add_argument(
        "--batch",
        default="auto",
        help='Batch size (number) or "auto" to probe optimal size',
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallelism per device (default: 1)",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="vLLM attention backend (e.g. FLASH_ATTN, FLASHINFER)",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="GPUs to use, e.g. --devices cuda:0 cuda:1 (default: single GPU)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON results (default: results)",
    )

    args = parser.parse_args(argv)

    benchmark_names = [b.strip().lower() for b in args.benchmark.split(",") if b.strip()]
    if not benchmark_names:
        sys.exit("No benchmarks specified")
    for name in benchmark_names:
        if name not in BENCHMARK_REGISTRY:
            sys.exit(
                f"Unknown benchmark: {name} (available: {', '.join(BENCHMARK_REGISTRY)})"
            )

    try:
        _validate_args(args)
    except ValueError as e:
        sys.exit(f"error: {e}")

    backend = None
    bs = None
    if not args.devices:
        VLLMBackend = _resolve("backends.vllm", "VLLMBackend")
        backend = VLLMBackend(
            model=args.model,
            batch_size=args.batch,
            tensor_parallel_size=args.tensor_parallel_size,
            attention_backend=args.attention_backend,
        )
        bs = backend.batch_size
        print(f"Batch size: {bs}")

    for idx, bench_name in enumerate(benchmark_names):
        module_name, class_name = BENCHMARK_REGISTRY[bench_name]
        bench_cls = _resolve(module_name, class_name)
        benchmark = bench_cls(num_samples=args.samples)

        mode = benchmark.mode

        print(f"[{bench_name}] Loading dataset...")
        benchmark.load()
        prompts = benchmark.build_prompts()
        print(f"[{bench_name}] Samples: {len(prompts)}, mode: {mode}")

        start = time.perf_counter()

        if args.devices:
            responses = run_distributed(
                prompts,
                args.model,
                args.devices,
                args.batch,
                args.tensor_parallel_size,
                args.attention_backend,
                mode=mode,
            )
            result = benchmark.score(responses)
            effective_bs = args.batch
        else:
            assert backend is not None and bs is not None
            result = _run_benchmark(benchmark, backend, prompts, bs, mode)
            effective_bs = bs

        elapsed = time.perf_counter() - start
        result.model = args.model
        result.time_seconds = elapsed
        try:
            result.batch_size = int(effective_bs) if effective_bs != "auto" else 0
        except (TypeError, ValueError):
            result.batch_size = 0

        if idx > 0:
            print()
        print("=" * 60)
        print(f"  Benchmark:  {result.name}")
        print(f"  Model:      {result.model}")
        print(f"  Accuracy:   {result.accuracy:.1%}  ({result.correct}/{result.total})")
        print(f"  Time:       {elapsed:.1f}s")
        if args.devices:
            print(f"  Devices:    {', '.join(args.devices)}")
        print("=" * 60)

        result_path = _save_result(
            result, args.model, args.devices, result.batch_size, Path(args.output_dir)
        )
        print(f"Results saved to {result_path}")
