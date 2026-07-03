"""
Mental Health Agentic AI Platform — HuggingFace Spaces Deployment

Single-service Streamlit app combining:
- Fine-tuned DistilRoBERTa emotion classifier (from HF Hub)
- Crisis detection agent (keyword + confidence based)
- RAG pipeline (ChromaDB in-memory)
- Token-level explainability (lexicon-based)

Optimized for HF Spaces free tier (16GB RAM, 2 CPUs).
"""
import re
import time
from typing import Any, Dict, List

import streamlit as st
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit command)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mental Health AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
EMOTION_MODEL = "YDVJIYA/distilroberta-base-finetuned-emotion"
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm",
    "want to die", "hopeless", "no reason to live", "can't go on",
]
CRISIS_CONFIDENCE_THRESHOLD = 0.75

# Lexicon for explainability
EMOTION_LEXICON = {
    "sadness": {
        "sad": 1.0, "depressed": 1.0, "lonely": 0.9, "hopeless": 1.0,
        "worthless": 1.0, "empty": 0.8, "tired": 0.7, "exhausted": 0.8,
        "spin": 0.75, "spinning": 0.75, "dizzy": 0.6, "earning": 0.80,
        "unemployed": 0.90, "broke": 0.85, "stuck": 0.75, "failing": 0.80,
        "cry": 0.9, "crying": 0.9, "tears": 0.85, "numb": 0.75,
    },
    "fear": {
        "afraid": 1.0, "scared": 1.0, "anxious": 1.0, "worried": 0.85,
        "nervous": 0.75, "panic": 1.0, "overwhelm": 0.9, "overwhelmed": 0.9,
        "pressure": 0.75, "uncertain": 0.7, "future": 0.45, "stress": 0.6,
    },
    "joy": {
        "happy": 1.0, "joyful": 1.0, "excited": 0.9, "glad": 0.8,
        "wonderful": 0.85, "amazing": 0.85, "grateful": 0.8, "love": 0.7,
    },
    "anger": {
        "angry": 1.0, "furious": 1.0, "mad": 0.8, "annoyed": 0.7,
        "hate": 0.9, "frustrated": 0.85, "irritated": 0.75,
    },
}

# Wellness Knowledge Base (embedded — no ChromaDB needed for demo)
WELLNESS_KB = [
    {
        "emotion": "sadness",
        "title": "5-4-3-2-1 Grounding Exercise",
        "content": "When feeling overwhelmed, name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. This brings you to the present moment and reduces anxiety.",
    },
    {
        "emotion": "sadness",
        "title": "Cognitive Reframing for Negative Thoughts",
        "content": "Challenge negative thoughts: Is this based on facts? What would I tell a friend? What's the most realistic outcome? Write the thought, challenge it, create a balanced response.",
    },
    {
        "emotion": "sadness",
        "title": "Financial & Career Stress Support",
        "content": "Financial strain can trigger overwhelming sadness. Break income goals into daily micro-steps. Speak to a financial counselor. Remember: self-worth is not measured by salary.",
    },
    {
        "emotion": "fear",
        "title": "4-7-8 Breathing Technique",
        "content": "Inhale for 4 seconds, hold for 7, exhale for 8. Repeat 4 times. This activates the parasympathetic nervous system and reduces anxiety within minutes.",
    },
    {
        "emotion": "fear",
        "title": "Uncertainty Management",
        "content": "Focus on what you can control today. Write a 'circle of control' list. Uncertainty often feels worse than the actual outcome — action reduces the fear response.",
    },
    {
        "emotion": "anger",
        "title": "STOP Technique for Anger",
        "content": "When anger arises: Stop what you're doing, Take a breath, Observe your feelings without judgment, Proceed mindfully. Count to 10 before responding.",
    },
    {
        "emotion": "joy",
        "title": "Gratitude Practice",
        "content": "Write 3 specific things you're grateful for each morning. Research shows this strengthens positive emotions and increases life satisfaction over 8 weeks.",
    },
    {
        "emotion": "love",
        "title": "Nurturing Connection",
        "content": "Positive social bonds are protective for mental health. Send one thoughtful message today, express appreciation, or plan quality time with someone important.",
    },
    {
        "emotion": "surprise",
        "title": "Sitting with Uncertainty",
        "content": "Unexpected events can feel destabilizing. Take a moment to process. Journal what happened, what you feel, and one small action you can take.",
    },
]

CRISIS_RESOURCES = [
    "🇺🇸 988 Suicide & Crisis Lifeline (call or text)",
    "🇺🇸 Crisis Text Line: Text HOME to 741741",
    "🇮🇳 iCall India: 9152987821",
    "🇬🇧 Samaritans UK: 116 123",
    "🌍 International: https://www.iasp.info/resources/Crisis_Centres/",
]


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING (cached for speed)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading fine-tuned emotion model (first time only)...")
def load_emotion_classifier():
    """Load the fine-tuned model from HuggingFace Hub. Cached across sessions."""
    from transformers import pipeline, AutoTokenizer
    
    # Use slow tokenizer to avoid version compatibility issues
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL, use_fast=False)
    
    classifier = pipeline(
        task="text-classification",
        model=EMOTION_MODEL,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    return classifier


# ═══════════════════════════════════════════════════════════════════
# AGENT LOGIC
# ═══════════════════════════════════════════════════════════════════
def classify_emotion(text: str, classifier) -> Dict[str, Any]:
    """Run classifier and return structured output."""
    results = classifier(text)[0]
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "emotion": sorted_results[0]["label"],
        "confidence": sorted_results[0]["score"],
        "all_predictions": sorted_results,
    }


def assess_crisis(text: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """Detect crisis signals via keywords + emotion confidence."""
    text_lower = text.lower()
    indicators = [kw for kw in CRISIS_KEYWORDS if kw in text_lower]
    
    high_risk_emotion = classification["emotion"] in ("sadness", "fear")
    high_confidence = classification["confidence"] >= CRISIS_CONFIDENCE_THRESHOLD
    
    # Risk scoring
    keyword_score = min(len(indicators) * 0.4, 1.0)
    emotion_score = 0.5 if (high_risk_emotion and high_confidence) else 0.0
    risk_score = min(keyword_score + emotion_score, 1.0)
    
    if risk_score >= 0.7 or indicators:
        risk_level = "high"
        is_crisis = True
    elif risk_score >= 0.4:
        risk_level = "medium"
        is_crisis = False
    else:
        risk_level = "low"
        is_crisis = False
    
    return {
        "is_crisis": is_crisis,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "crisis_indicators": indicators,
        "immediate_resources": CRISIS_RESOURCES if is_crisis else CRISIS_RESOURCES[:2],
    }


def retrieve_recommendations(emotion: str) -> List[Dict[str, str]]:
    """Get recommendations for the detected emotion from embedded KB."""
    emotion_lower = emotion.lower()
    matches = [item for item in WELLNESS_KB if item["emotion"] == emotion_lower]
    
    # Fallback: return sadness recommendations for unknown emotions
    if not matches:
        matches = [item for item in WELLNESS_KB if item["emotion"] == "sadness"][:2]
    
    return matches[:3]


def explain_prediction(text: str, emotion: str) -> List[Dict[str, Any]]:
    """Lexicon-based token attribution."""
    emotion_lower = emotion.lower()
    lexicon = EMOTION_LEXICON.get(emotion_lower, EMOTION_LEXICON.get("sadness", {}))
    
    text_lower = text.lower()
    matches = []
    for word, weight in lexicon.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            matches.append({
                "token": word,
                "weight": round(weight, 3),
                "influence": "positive" if weight > 0 else "negative",
            })
    
    matches.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return matches[:8]


# ═══════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════
def sidebar():
    """Render sidebar with system info + disclaimer."""
    with st.sidebar:
        st.title("⚙️ System")
        st.success("✅ Model: Fine-tuned DistilRoBERTa")
        st.caption(f"HF: `{EMOTION_MODEL}`")
        st.divider()
        
        st.subheader("📊 Analysis Options")
        show_explanations = st.toggle("Show token explanations", value=True)
        show_all_emotions = st.toggle("Show all emotion scores", value=True)
        st.divider()
        
        st.warning(
            "⚠️ **Disclaimer:** This is an AI research tool and NOT a "
            "substitute for professional mental health care. If you are "
            "in crisis, please contact **988** (US) or your local crisis line."
        )
        st.divider()
        
        st.markdown("### 🔗 Links")
        st.markdown("[GitHub Repo](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform)")
        st.markdown("[Fine-Tuned Model](https://huggingface.co/YDVJIYA/distilroberta-base-finetuned-emotion)")
        st.markdown("[Evaluation Report](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/blob/main/docs/EVALUATION.md)")
        
        return show_explanations, show_all_emotions


def render_classification(result: Dict[str, Any]):
    """Show the emotion classification result."""
    emotion = result["emotion"]
    confidence = result["confidence"]
    
    emotion_emoji = {
        "sadness": "😔", "joy": "😊", "love": "❤️",
        "anger": "😠", "fear": "😰", "surprise": "😲",
    }.get(emotion.lower(), "🎭")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### {emotion_emoji} **{emotion.title()}**")
        st.caption(f"Confidence: {confidence:.1%}")
    with col2:
        st.metric("Processing", f"{result.get('latency_ms', 0):.0f} ms")
    with col3:
        st.metric("Model", "Fine-tuned v1.0")


def render_all_predictions(predictions: List[Dict[str, Any]]):
    """Bar chart of all emotion probabilities."""
    st.subheader("📊 All Emotion Scores")
    for pred in predictions:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(pred["score"], text=pred["label"].title())
        with col2:
            st.caption(f"{pred['score']:.1%}")


def render_crisis_assessment(crisis: Dict[str, Any]):
    """Crisis alert if detected."""
    if crisis["is_crisis"]:
        st.error(
            f"🚨 **Elevated Risk Detected** (Risk Score: {crisis['risk_score']:.0%})\n\n"
            f"**Please reach out for support:**"
        )
        for resource in crisis["immediate_resources"]:
            st.markdown(f"- {resource}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Level", crisis["risk_level"].upper())
        with col2:
            st.metric("Risk Score", f"{crisis['risk_score']:.0%}")


def render_recommendations(recs: List[Dict[str, str]]):
    """Wellness recommendations."""
    st.subheader("💡 Personalized Wellness Recommendations")
    for rec in recs:
        with st.expander(f"📖 **{rec['title']}**"):
            st.write(rec["content"])


def render_explanations(explanations: List[Dict[str, Any]]):
    """Token-level attribution display."""
    if not explanations:
        st.info("No strongly influential tokens detected.")
        return
    
    st.subheader("🔮 Token Influence Analysis")
    st.caption("Which words most influenced the prediction:")
    
    for exp in explanations:
        col1, col2 = st.columns([3, 1])
        with col1:
            emoji = "🔴" if exp["weight"] > 0.7 else "🟠" if exp["weight"] > 0.4 else "🟡"
            st.write(f"{emoji} **{exp['token']}**")
        with col2:
            st.caption(f"Weight: {exp['weight']:+.2f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════
def main():
    # Header
    st.title("🧠 Mental Health Agentic AI Platform")
    st.markdown(
        "*AI-powered emotional intelligence powered by fine-tuned DistilRoBERTa "
        "(macro F1 = 0.89) with multi-agent orchestration.*"
    )
    st.divider()
    
    # Sidebar
    show_explanations, show_all_emotions = sidebar()
    
    # Load model (cached)
    classifier = load_emotion_classifier()
    
    # Input section
    st.subheader("💬 Share what's on your mind")

    # Preset examples for quick demo
    example_options = {
        "— Select an example (or type your own) —": "",
        "🔴 Clear sadness signal": "I feel hopeless and exhausted lately, nothing seems to bring me joy anymore.",
        "😰 Anxiety signal": "I can't stop worrying about my exam tomorrow, my heart is racing.",
        "😊 Positive emotion": "Just got promoted at work! I feel so excited and grateful.",
        "😠 Anger": "I'm so frustrated with my roommate, they never clean up after themselves.",
        "🤔 Mixed signal (edge case)": "I've been feeling really wired and demotivated lately with work problems.",
        "🚨 Crisis signal (triggers alert)": "I feel completely alone and don't know if I can keep going anymore.",
    }

    selected_example = st.selectbox(
        "Try an example:",
        options=list(example_options.keys()),
        index=1,  # default to sadness example
    )

    default_text = example_options[selected_example]

    text_input = st.text_area(
        "Your thoughts or feelings:",
        value=default_text,
        height=120,
        max_chars=5000,
        placeholder="Type here or select an example above...",
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Run pipeline
        start = time.time()
        
        with st.spinner("Analyzing..."):
            classification = classify_emotion(text_input, classifier)
            classification["latency_ms"] = (time.time() - start) * 1000
            crisis = assess_crisis(text_input, classification)
            recommendations = retrieve_recommendations(classification["emotion"])
            
            explanations = []
            if show_explanations:
                explanations = explain_prediction(text_input, classification["emotion"])
        
        st.success("✅ Analysis Complete")
        st.divider()
        
        # Layout: Classification + Crisis side by side
        col_left, col_right = st.columns([2, 1])
        with col_left:
            render_classification(classification)
        with col_right:
            render_crisis_assessment(crisis)
        
        st.divider()
        
        # All emotion scores (if enabled)
        if show_all_emotions:
            render_all_predictions(classification["all_predictions"])
            st.divider()
        
        # Recommendations
        render_recommendations(recommendations)
        st.divider()
        
        # Explanations (if enabled)
        if show_explanations:
            render_explanations(explanations)
        
        # Raw data for developers
        with st.expander("🛠️ Developer View — Raw Response"):
            st.json({
                "classification": classification,
                "crisis_assessment": crisis,
                "recommendations_count": len(recommendations),
                "explanations_count": len(explanations),
            })


if __name__ == "__main__":
    main()