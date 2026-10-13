"""fasteval: minimal evaluation harness for LLMs (vLLM backend)."""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("fasteval")
except Exception:
    __version__ = "0.1.0"

from fasteval.base import Backend, Benchmark, BenchmarkResult

__all__ = ["Backend", "Benchmark", "BenchmarkResult", "__version__"]
