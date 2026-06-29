# tests/unit/test_evaluation.py
"""Tests for the evaluation label mapping logic."""
import pytest


pytestmark = pytest.mark.unit


class TestLabelMapping:

    def test_dataset_label_zero_is_sadness(self):
        from evaluation.label_mapper import dataset_label_to_model_label
        assert dataset_label_to_model_label(0) == "sadness"

    def test_love_collapses_to_joy(self):
        from evaluation.label_mapper import dataset_label_to_model_label
        # Label 2 is 'love' in dair-ai/emotion
        assert dataset_label_to_model_label(2) == "joy"

    def test_unknown_label_raises(self):
        from evaluation.label_mapper import dataset_label_to_model_label
        with pytest.raises(ValueError):
            dataset_label_to_model_label(99)

    def test_disgust_normalizes_to_none(self):
        """Model emits 'disgust' but dataset has no equivalent."""
        from evaluation.label_mapper import normalize_model_prediction
        assert normalize_model_prediction("disgust") is None

    def test_neutral_normalizes_to_none(self):
        from evaluation.label_mapper import normalize_model_prediction
        assert normalize_model_prediction("neutral") is None

    def test_normalize_handles_case_and_whitespace(self):
        from evaluation.label_mapper import normalize_model_prediction
        assert normalize_model_prediction("  Sadness  ") == "sadness"
        assert normalize_model_prediction("JOY") == "joy"

    def test_eval_label_set_has_5_classes(self):
        from evaluation.label_mapper import get_eval_label_set
        labels = get_eval_label_set()
        assert len(labels) == 5
        assert "sadness" in labels
        assert "joy" in labels
        assert "anger" in labels
        assert "fear" in labels
        assert "surprise" in labels
        assert "love" not in labels  # collapsed into joy

    def test_eval_label_set_is_sorted(self):
        from evaluation.label_mapper import get_eval_label_set
        labels = get_eval_label_set()
        assert labels == sorted(labels)


class TestReportGenerator:

    def test_generates_valid_markdown(self, tmp_path):
        from evaluation.report_generator import generate_markdown_report
        fake_metrics = {
            "macro_f1": 0.75,
            "weighted_f1": 0.78,
            "accuracy": 0.80,
            "per_class": {
                "sadness": {"precision": 0.8, "recall": 0.75, "f1": 0.77, "support": 100},
                "joy":     {"precision": 0.85, "recall": 0.82, "f1": 0.83, "support": 200},
            },
            "labels_used": ["sadness", "joy"],
            "crisis_recall": 0.72,
            "crisis_precision": 0.81,
            "confusion_summary": "**Test summary**",
        }
        out = tmp_path / "TEST_REPORT.md"
        generate_markdown_report(
            fake_metrics, out,
            model_name="test-model",
            dataset_name="test-dataset",
            sample_count=300,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "0.750" in content  # macro_f1
        assert "300" in content    # sample_count
        assert "sadness" in content
        assert "test-model" in content