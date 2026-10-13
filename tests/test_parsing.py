"""Regression: MMLU / GSM8K parsing + base contract.

These tests define the CORRECT behavior. They FAIL on unfixed code.
"""
from fasteval.base import Backend, BenchmarkResult
import pytest


class TestMatchAnswer:
    def test_answer_b_not_a(self):
        # "ANSWER: B" contains the letter A inside ANSWER -> must be 1, not 0
        assert Backend._match_answer("ANSWER: B", ["A", "B", "C", "D"]) == 1

    def test_substring_false_positive(self):
        # "EYES" contains "YES" as a substring -> must not match
        assert Backend._match_answer("EYES", ["yes", "no"]) is None

    def test_no_match_returns_none(self):
        assert Backend._match_answer("I don't know", ["A", "B", "C", "D"]) is None

    def test_empty_candidates_returns_none(self):
        assert Backend._match_answer("A", []) is None

    def test_empty_text_returns_none(self):
        assert Backend._match_answer("", ["A", "B"]) is None
        assert Backend._match_answer("   ", ["A", "B"]) is None


class TestScoreAnswersValidation:
    def test_mismatched_lengths_raise(self):
        class Dummy(Backend):
            def generate(self, prompts, batch_size):
                return ["A", "B", "C"]

            def auto_batch(self):
                return 1

            @property
            def model_name(self):
                return "dummy"

        d = Dummy()
        with pytest.raises(ValueError):
            d.score_answers(["p1", "p2"], 1, [["A", "B"]] * 2)


class TestBenchmarkResult:
    def test_invalid_total_raises(self):
        with pytest.raises(ValueError):
            BenchmarkResult(name="x", model="m", accuracy=0.5, correct=1, total=-1, time_seconds=1.0)

    def test_correct_gt_total_raises(self):
        with pytest.raises(ValueError):
            BenchmarkResult(name="x", model="m", accuracy=0.5, correct=5, total=3, time_seconds=1.0)

    def test_accuracy_out_of_range_raises(self):
        with pytest.raises(ValueError):
            BenchmarkResult(name="x", model="m", accuracy=1.5, correct=1, total=1, time_seconds=1.0)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError):
            BenchmarkResult(name="x", model="m", accuracy=0.0, correct=0, total=1, time_seconds=-1.0)


class TestMMLUExtract:
    def test_answer_b_with_prefix(self):
        from fasteval.benchmarks.mmlu import MMLUBenchmark
        assert MMLUBenchmark._extract_answer("ANSWER: B") == "B"

    def test_elaboration_does_not_force_a(self):
        from fasteval.benchmarks.mmlu import MMLUBenchmark
        # sentence contains 'a' many times, the answer is C
        assert MMLUBenchmark._extract_answer("The answer is C") == "C"

    def test_single_letters(self):
        from fasteval.benchmarks.mmlu import MMLUBenchmark
        for L in ["A", "B", "C", "D"]:
            assert MMLUBenchmark._extract_answer(L) == L
            assert MMLUBenchmark._extract_answer(f"  {L.lower()}  ") == L

    def test_unparseable_returns_empty(self):
        from fasteval.benchmarks.mmlu import MMLUBenchmark
        assert MMLUBenchmark._extract_answer("I don't know ???") == ""

    def test_expected_range(self):
        from fasteval.benchmarks.mmlu import MMLUBenchmark
        assert MMLUBenchmark._extract_expected(0) == "A"
        assert MMLUBenchmark._extract_expected(3) == "D"
        with pytest.raises(ValueError):
            MMLUBenchmark._extract_expected(4)
        with pytest.raises(ValueError):
            MMLUBenchmark._extract_expected(-1)


class TestGSM8KExtract:
    def test_fallback_last_number(self):
        from fasteval.benchmarks.gsm8k import GSM8KBenchmark
        # no ####, but correct math at the end
        assert GSM8KBenchmark._extract_answer("The answer is 42") == 42.0

    def test_takes_last_hash(self):
        from fasteval.benchmarks.gsm8k import GSM8KBenchmark
        assert GSM8KBenchmark._extract_answer("#### 10 ... #### 42") == 42.0

    def test_comma_number(self):
        from fasteval.benchmarks.gsm8k import GSM8KBenchmark
        assert GSM8KBenchmark._extract_answer("#### 1,000") == 1000.0

    def test_score_uses_isclose(self):
        from fasteval.benchmarks.gsm8k import GSM8KBenchmark
        b = GSM8KBenchmark()
        b._data = [{"question": "q", "answer": "#### 0.3"}]
        # 0.1+0.2 != 0.3 in binary, but isclose should accept it
        r = b.score([str(0.1 + 0.2)])
        assert r.correct == 1
