# tests/unit/test_explainer.py
"""Test the lexicon-based emotion explainer."""
import pytest


pytestmark = pytest.mark.unit


# ── Emotion normalization ──────────────────────────────────────────────
class TestEmotionNormalization:

    def test_strips_slash_suffix(self):
        from backend.explainability.explainer import EmotionExplainer
        assert EmotionExplainer._normalize_emotion("Sadness / Depression") == "sadness"
        assert EmotionExplainer._normalize_emotion("Fear / Anxiety") == "fear"

    def test_lowercases_input(self):
        from backend.explainability.explainer import EmotionExplainer
        assert EmotionExplainer._normalize_emotion("SADNESS") == "sadness"

    def test_handles_empty_string(self):
        from backend.explainability.explainer import EmotionExplainer
        assert EmotionExplainer._normalize_emotion("") == "neutral"

    def test_handles_none(self):
        from backend.explainability.explainer import EmotionExplainer
        assert EmotionExplainer._normalize_emotion(None) == "neutral"

    def test_unknown_falls_back_to_sadness(self):
        from backend.explainability.explainer import EmotionExplainer
        # Documented behavior — default fallback per implementation
        assert EmotionExplainer._normalize_emotion("alien_emotion") == "sadness"


# ── Token scoring ──────────────────────────────────────────────────────
class TestTokenScoring:

    def test_earning_triggers_sadness(self):
        """Regression test for Bug #2 — financial stressors should match."""
        from backend.explainability.explainer import EmotionExplainer
        explainer = EmotionExplainer()
        result = explainer.explain(
            text="I am not earning enough money",
            primary_emotion="Sadness / Depression",
        )
        tokens = {t["token"] for t in result["tokens"]}
        assert "earning" in tokens

    def test_spin_triggers_sadness(self):
        from backend.explainability.explainer import EmotionExplainer
        explainer = EmotionExplainer()
        result = explainer.explain(
            text="my head is spinning",
            primary_emotion="sadness",
        )
        tokens = {t["token"] for t in result["tokens"]}
        assert "spinning" in tokens or "spin" in tokens

    def test_clinical_sadness_keyword_detected(self):
        from backend.explainability.explainer import EmotionExplainer
        explainer = EmotionExplainer()
        result = explainer.explain(
            text="I feel hopeless and worthless",
            primary_emotion="sadness",
        )
        tokens = {t["token"] for t in result["tokens"]}
        assert "hopeless" in tokens
        assert "worthless" in tokens

    def test_no_matches_returns_empty_or_summary(self):
        from backend.explainability.explainer import EmotionExplainer
        explainer = EmotionExplainer()
        result = explainer.explain(
            text="xyz qwerty zzz",
            primary_emotion="sadness",
        )
        assert result["tokens"] == []
        assert "No strongly influential" in result["summary"]

    def test_response_shape_contract(self):
        """Ensures response keys match what routes.py expects."""
        from backend.explainability.explainer import EmotionExplainer
        explainer = EmotionExplainer()
        result = explainer.explain(text="I feel sad", primary_emotion="sadness")

        assert "tokens" in result
        assert "summary" in result
        assert "method" in result
        assert "confidence_indicators" in result

        if result["tokens"]:
            tok = result["tokens"][0]
            assert "token" in tok
            assert "weight" in tok
            assert "influence" in tok
            assert tok["influence"] in ("positive", "negative")

    def test_max_tokens_respected(self):
        """UI cap should never be exceeded."""
        from backend.config.settings import settings
        from backend.explainability.explainer import EmotionExplainer
        explainer = EmotionExplainer()
        # Throw the whole sadness lexicon at it
        text = "sad depressed hopeless lonely empty tired exhausted broke earning"
        result = explainer.explain(text=text, primary_emotion="sadness")
        assert len(result["tokens"]) <= settings.EXPLAINER_MAX_TOKENS


# ── Singleton ──────────────────────────────────────────────────────────
class TestSingleton:

    def test_singleton_is_instance(self):
        from backend.explainability.explainer import EmotionExplainer, explainer
        assert isinstance(explainer, EmotionExplainer)

    def test_alias_matches(self):
        from backend.explainability.explainer import explainer, emotion_explainer
        assert explainer is emotion_explainer