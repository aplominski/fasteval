# fasteval

> This is proof of concept! Dont use it in production!

Minimal evaluation harness for LLMs. Uses vLLM under the hood, runs
benchmarks from HuggingFace datasets.

## Install

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Usage

Single GPU:

```bash
python -m fasteval --model meta-llama/Llama-3-8B-Instruct gsm8k
```

Two GPUs — each runs its own vLLM instance, dataset split round-robin:

```bash
python -m fasteval --model meta-llama/Llama-3-8B-Instruct gsm8k --devices cuda:0 cuda:1
```

Output lands in `results/` as JSON and on stdout:

```
Loading gsm8k dataset...
Samples: 1319
Devices: cuda:0, cuda:1

============================================================
  Benchmark:  gsm8k
  Model:      meta-llama/Llama-3-8B-Instruct
  Accuracy:   73.0%  (963/1319)
  Time:       118.2s
  Devices:    cuda:0, cuda:1
============================================================

Results saved to results/gsm8k__meta-llama__Llama-3-8B-Instruct.json
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | HuggingFace model name |
| `--batch` | `auto` | Batch size or `auto` to probe |
| `--samples` | all | Limit number of test samples |
| `--tensor-parallel-size` | `1` | vLLM tensor parallelism per device |
| `--devices` | — | GPUs to use, e.g. `--devices cuda:0 cuda:1` |

## Benchmarks

### GSM8K

8-shot chain-of-thought evaluation. Follows the methodology from
Cobbe et al. 2021 — exact match on `#### {answer}`, greedy decoding
(temperature 0, max 400 tokens).

Few-shot exemplars are the first 8 samples from the training set
(stripped of `<<...>>` calculator annotations), loaded from
`openai/gsm8k` on HuggingFace.

## Batch auto

`--batch auto` probes the GPU by sending progressively larger batches
(1, 2, 4, 8… up to 128) until it hits CUDA OOM, then caches the
result per model in `.fasteval_cache/<model_name>/batch_size.json`.

Run once with `auto`, it reuses the cached value on subsequent runs.

## Structure

```
fasteval/
├── main.py
├── fasteval/
│   ├── base.py              # Backend / Benchmark ABCs
│   ├── cli.py               # argparse entry point
│   ├── progress.py           # tqdm wrapper (vLLM tqdm disabled)
│   ├── backends/
│   │   └── vllm.py           # VLLMBackend
│   └── benchmarks/
│       └── gsm8k.py          # GSM8KBenchmark
├── results/                  # JSON output
└── .fasteval_cache/          # per-model batch cache
```
