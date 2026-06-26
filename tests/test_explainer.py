def test_normalize_emotion_strips_classifier_suffix():
    assert EmotionExplainer._normalize_emotion("Sadness / Depression") == "sadness"
    assert EmotionExplainer._normalize_emotion("Fear / Anxiety") == "fear"

def test_earning_triggers_sadness_weight():
    explainer = EmotionExplainer()
    result = explainer.explain(
        text="I am not earning enough money",
        primary_emotion="Sadness / Depression"
    )
    tokens = {t["token"] for t in result["tokens"]}
    assert "earning" in tokens

def test_explainer_handles_empty_text_gracefully():
    explainer = EmotionExplainer()
    result = explainer.explain(text="", primary_emotion="sadness")
    assert result["tokens"] == []
    assert "No strongly influential" in result["summary"]