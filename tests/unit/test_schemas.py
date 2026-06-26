# tests/unit/test_schemas.py
"""Test Pydantic request/response schema validation."""
import pytest
from pydantic import ValidationError


pytestmark = pytest.mark.unit


# ── ClassificationRequest ─────────────────────────────────────────────
class TestClassificationRequest:

    def test_valid_input_accepted(self):
        from backend.api.schemas import ClassificationRequest
        req = ClassificationRequest(text="I feel sad")
        assert req.text == "I feel sad"
        assert req.include_explanations is False

    def test_empty_text_rejected(self):
        from backend.api.schemas import ClassificationRequest
        with pytest.raises(ValidationError):
            ClassificationRequest(text="")

    def test_whitespace_only_rejected(self):
        from backend.api.schemas import ClassificationRequest
        with pytest.raises(ValidationError):
            ClassificationRequest(text="     ")

    def test_text_is_stripped(self):
        from backend.api.schemas import ClassificationRequest
        req = ClassificationRequest(text="  hello  ")
        assert req.text == "hello"

    def test_text_max_length_enforced(self):
        from backend.api.schemas import ClassificationRequest
        with pytest.raises(ValidationError):
            ClassificationRequest(text="x" * 5001)


# ── ExplanationToken ───────────────────────────────────────────────────
class TestExplanationToken:

    def test_valid_positive_influence(self):
        from backend.api.schemas import ExplanationToken
        tok = ExplanationToken(token="sad", weight=0.85, influence="positive")
        assert tok.influence == "positive"

    def test_valid_negative_influence(self):
        from backend.api.schemas import ExplanationToken
        tok = ExplanationToken(token="good", weight=-0.1, influence="negative")
        assert tok.weight == -0.1

    def test_invalid_influence_rejected(self):
        from backend.api.schemas import ExplanationToken
        with pytest.raises(ValidationError):
            ExplanationToken(token="x", weight=0.5, influence="maybe")

    def test_weight_out_of_range_rejected(self):
        from backend.api.schemas import ExplanationToken
        with pytest.raises(ValidationError):
            ExplanationToken(token="x", weight=5.0, influence="positive")


# ── CrisisAssessment ───────────────────────────────────────────────────
class TestCrisisAssessment:

    def test_valid_assessment(self):
        from backend.api.schemas import CrisisAssessment
        ca = CrisisAssessment(
            is_crisis=False,
            risk_level="low",
            risk_score=0.05,
            crisis_indicators=[],
            immediate_resources=["988"],
        )
        assert ca.risk_level == "low"

    def test_invalid_risk_level_rejected(self):
        from backend.api.schemas import CrisisAssessment
        with pytest.raises(ValidationError):
            CrisisAssessment(
                is_crisis=False,
                risk_level="kinda-bad",  # not in Literal
                risk_score=0.5,
                crisis_indicators=[],
                immediate_resources=[],
            )