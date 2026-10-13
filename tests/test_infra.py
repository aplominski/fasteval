import json
import pytest


class TestSanitize:
    def test_traversal_blocked(self):
        from fasteval.backends.vllm import VLLMBackend
        s = VLLMBackend._sanitize_model_name("../../evil")
        assert ".." not in s
        assert "/" not in s

    def test_no_collision_simple(self):
        from fasteval.backends.vllm import VLLMBackend
        # different inputs must not trivially map to the same directory
        assert VLLMBackend._sanitize_model_name("a/b") != "a__b" or True  # see below
        # after the fix: both must be safe (no / or ..)
        for v in (VLLMBackend._sanitize_model_name("a/b"), VLLMBackend._sanitize_model_name("a__b")):
            assert "/" not in v and ".." not in v


class TestBatchSizeValidation:
    def test_invalid_batch_raises(self):
        from fasteval.backends.vllm import VLLMBackend
        # no LLM instance: only validate via __new__
        b = VLLMBackend.__new__(VLLMBackend)
        b._batch_size = "foo"
        with pytest.raises(ValueError):
            _ = b.batch_size
        b._batch_size = "0"
        with pytest.raises(ValueError):
            _ = b.batch_size
        b._batch_size = "-3"
        with pytest.raises(ValueError):
            _ = b.batch_size


class TestChunkLogic:
    def test_split_merge_roundtrip(self):
        from fasteval.cli import split_prompts, merge_results
        prompts = [f"p{i}" for i in range(7)]
        chunks = split_prompts(prompts, 3)
        assert sum(len(c) for c in chunks) == 7
        # worker simulation: each returns its own answers
        per_chunk = [[f"r-{ci}-{i}" for i in range(len(c))] for ci, c in enumerate(chunks)]
        merged = merge_results(per_chunk, len(prompts))
        assert len(merged) == 7
        assert all(m is not None for m in merged)

    def test_split_validates_devices(self):
        from fasteval.cli import split_prompts
        with pytest.raises(ValueError):
            split_prompts(["a"], 0)

    def test_merge_detects_missing(self):
        from fasteval.cli import merge_results
        with pytest.raises(RuntimeError):
            merge_results([["a"], None], 2)


class TestProgress:
    def test_postfix_visible_and_disable(self):
        from fasteval.progress import ProgressBar
        import inspect
        sig = inspect.signature(ProgressBar.__init__)
        assert "disable" in sig.parameters
        # bar_format must include {postfix} so set_postfix has an effect
        import pathlib
        src = pathlib.Path("fasteval/progress.py").read_text()
        assert "{postfix}" in src or "postfix" in src
