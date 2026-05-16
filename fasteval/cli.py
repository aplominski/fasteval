import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from fasteval.base import BenchmarkResult


BENCHMARK_REGISTRY: dict[str, tuple[str, str]] = {
    "gsm8k": ("benchmarks.gsm8k", "GSM8KBenchmark"),
    "mmlu": ("benchmarks.mmlu", "MMLUBenchmark"),
}


def _resolve(submodule: str, name: str):
    mod = importlib.import_module(f"fasteval.{submodule}")
    return getattr(mod, name)


class WorkerError(Exception):
    pass


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
        os.environ["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")
        VLLMBackend = _resolve("backends.vllm", "VLLMBackend")
        backend = VLLMBackend(
            model=model,
            batch_size=batch,
            tensor_parallel_size=tensor_parallel_size,
            attention_backend=attention_backend,
        )
        bs = backend.batch_size
        if mode == "onepass":
            choices = [["A", "B", "C", "D"]] * len(prompts)
            answer_indices = backend.score_answers(prompts, bs, choices)
            responses = [chr(65 + idx) for idx in answer_indices]
        else:
            responses = backend.generate(prompts, batch_size=bs)
        queue.put((worker_id, None, responses))
    except Exception as e:
        queue.put((worker_id, str(e), None))


def run_distributed(
    prompts: list[str],
    model: str,
    devices: list[str],
    batch: str,
    tensor_parallel_size: int,
    attention_backend: str | None = None,
    mode: str = "fullgen",
) -> list[str]:
    mp.set_start_method("spawn", force=True)
    n = len(devices)
    chunks = [prompts[i::n] for i in range(n)]

    queue: mp.Queue = mp.Queue()
    processes = []
    for i, device in enumerate(devices):
        p = mp.Process(
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

    for p in processes:
        p.join()

    ordered = [None] * n
    errors = {}
    while not queue.empty():
        idx, err, responses = queue.get()
        if err:
            errors[devices[idx]] = err
        else:
            ordered[idx] = responses

    if errors:
        for dev, err in errors.items():
            print(f"FAILED {dev}: {err}", file=sys.stderr)
        sys.exit(1)

    all_results = [None] * len(prompts)
    for chunk_idx, responses in enumerate(ordered):
        for i, resp in enumerate(responses):
            all_results[chunk_idx + i * n] = resp

    return all_results


def _run_benchmark(benchmark, backend, prompts, batch_size, mode: str):
    if mode == "onepass":
        choices = [["A", "B", "C", "D"]] * len(prompts)
        answer_indices = backend.score_answers(prompts, batch_size, choices)
        responses = [chr(65 + idx) for idx in answer_indices]
    else:
        responses = backend.generate(prompts, batch_size)
    return benchmark.score(responses)


def main():
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

    args = parser.parse_args()

    benchmark_names = [b.strip() for b in args.benchmark.split(",")]
    for name in benchmark_names:
        if name not in BENCHMARK_REGISTRY:
            sys.exit(
                f"Unknown benchmark: {name} (available: {', '.join(BENCHMARK_REGISTRY)})"
            )

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

        mode = "fullgen" if bench_name == "gsm8k" else "onepass"

        print(f"[{bench_name}] Loading dataset...")
        benchmark.load()
        prompts = benchmark.build_prompts()
        print(f"[{bench_name}] Samples: {len(prompts)}, mode: {mode}")

        start = time.time()

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
        else:
            result = _run_benchmark(benchmark, backend, prompts, bs, mode)

        elapsed = time.time() - start
        result.model = args.model
        result.time_seconds = elapsed

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

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        safe_name = args.model.replace("/", "__").replace(":", "_")
        result_path = results_dir / f"{result.name}__{safe_name}.json"
        with open(result_path, "w") as f:
            json.dump(
                {
                    "benchmark": result.name,
                    "model": result.model,
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                    "time_seconds": result.time_seconds,
                    "devices": args.devices,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {result_path}")
