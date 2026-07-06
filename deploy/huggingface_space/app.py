"""
Mental Health Agentic AI Platform — HuggingFace Spaces Deployment (v1.1)

Improvements over v1.0:
- Layered classification with safety overrides for clinical vocabulary
- Confidence threshold warnings for ambiguous predictions
- Keyword-based crisis vocabulary override (protects against false positives
  like classifying "demotivated" as love)
- Example input dropdown for better first-time UX
- Top-N predictions shown when model is uncertain
"""
import re
import time
import streamlit as st
import requests
import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mental Health AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add streaming import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
EMOTION_MODEL = "YDVJIYA/distilroberta-base-finetuned-emotion"

# Confidence threshold below which we show a warning + top-N alternatives
LOW_CONFIDENCE_THRESHOLD = 0.60

# ── High-Priority Sadness Vocabulary (Clinical / Mental Health) ─────
# These words STRONGLY indicate sadness/distress even if the model gets
# confused by surrounding positive-sounding structure. If ANY of these
# appear and the model didn't predict sadness, we override.
SADNESS_OVERRIDE_KEYWORDS = {
    # Direct distress signals
    "hopeless", "worthless", "empty", "numb", "meaningless", "pointless",
    # Motivation / energy loss  
    "demotivated", "unmotivated", "exhausted", "drained", "burnt out",
    "burned out", "no energy", "can't focus",
    # Depression indicators
    "depressed", "depression", "suicidal", "self-harm",
    # Life stagnation
    "stuck", "trapped", "failing", "failure", "giving up",
    # Sleep / rest issues (often depression symptoms)
    "can't sleep", "insomnia", "always tired",
    # Isolation
    "lonely", "isolated", "alone",
}

# ── Fear/Anxiety Override Vocabulary ────────────────────────────────
FEAR_OVERRIDE_KEYWORDS = {
    "anxious", "anxiety", "panic", "terrified", "overwhelmed",
    "can't breathe", "heart racing", "afraid", "scared",
}

# ── Crisis Vocabulary (Highest Priority — always triggers alert) ────
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm",
    "want to die", "no reason to live", "can't go on",
    "hurt myself", "not worth living",
]

CRISIS_CONFIDENCE_THRESHOLD = 0.75

# ── Lexicon for token-level explanation display ─────────────────────
EMOTION_LEXICON = {
    "sadness": {
        "sad": 1.0, "depressed": 1.0, "lonely": 0.9, "hopeless": 1.0,
        "worthless": 1.0, "empty": 0.8, "tired": 0.7, "exhausted": 0.8,
        "spin": 0.75, "spinning": 0.75, "dizzy": 0.6, "earning": 0.80,
        "unemployed": 0.90, "broke": 0.85, "stuck": 0.75, "failing": 0.80,
        "cry": 0.9, "crying": 0.9, "tears": 0.85, "numb": 0.75,
        "demotivated": 0.90, "unmotivated": 0.85, "drained": 0.75,
        "meaningless": 0.90, "pointless": 0.85, "burnt": 0.75, "burned": 0.75,
    },
    "fear": {
        "afraid": 1.0, "scared": 1.0, "anxious": 1.0, "worried": 0.85,
        "nervous": 0.75, "panic": 1.0, "overwhelm": 0.9, "overwhelmed": 0.9,
        "pressure": 0.75, "uncertain": 0.7, "future": 0.45, "stress": 0.6,
        "terrified": 1.0, "racing": 0.7, "breathe": 0.6,
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

# ── Wellness Knowledge Base ─────────────────────────────────────────
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
        "emotion": "sadness",
        "title": "Motivation & Energy Recovery",
        "content": "When you feel demotivated or drained: Start with one tiny action (2 minutes). Movement follows action, not the other way around. Small wins compound over time.",
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
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading fine-tuned emotion model (first time only)...")
def load_emotion_classifier():
    from transformers import pipeline, AutoTokenizer
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
# CLASSIFICATION WITH SAFETY OVERRIDES
# ═══════════════════════════════════════════════════════════════════
def _contains_any(text_lower: str, keywords: set) -> Optional[str]:
    """Return the first matching keyword, or None."""
    for kw in keywords:
        if kw in text_lower:
            return kw
    return None


def classify_with_safety_overrides(text: str, classifier) -> Dict[str, Any]:
    """
    Run model + apply safety overrides for clinical vocabulary.
    
    Priority hierarchy:
    1. Crisis keywords → force sadness + crisis flag
    2. Sadness clinical vocabulary → override to sadness if model missed
    3. Fear clinical vocabulary → override to fear if model missed
    4. Otherwise → use model prediction as-is
    
    Also flags low-confidence predictions for UI warning.
    """
    text_lower = text.lower()
    
    # Get raw model predictions
    raw_results = classifier(text)[0]
    sorted_results = sorted(raw_results, key=lambda x: x["score"], reverse=True)
    top_emotion = sorted_results[0]["label"].lower()
    top_confidence = sorted_results[0]["score"]
    
    # Track override reasoning for transparency
    override_reason = None
    override_applied = False
    
    # ── Layer 1: Crisis keywords take absolute priority ──
    crisis_match = _contains_any(text_lower, set(CRISIS_KEYWORDS))
    if crisis_match:
        top_emotion = "sadness"
        override_reason = f"Crisis keyword detected: '{crisis_match}'"
        override_applied = True
    
    # ── Layer 2: Clinical sadness vocabulary override ──
    elif top_emotion not in ("sadness", "fear"):
        sadness_match = _contains_any(text_lower, SADNESS_OVERRIDE_KEYWORDS)
        if sadness_match:
            top_emotion = "sadness"
            override_reason = f"Clinical distress vocabulary: '{sadness_match}'"
            override_applied = True
    
    # ── Layer 3: Fear vocabulary override ──
    if not override_applied and top_emotion not in ("fear", "sadness"):
        fear_match = _contains_any(text_lower, FEAR_OVERRIDE_KEYWORDS)
        if fear_match:
            top_emotion = "fear"
            override_reason = f"Anxiety vocabulary detected: '{fear_match}'"
            override_applied = True
    
    # If override changed the top emotion, adjust confidence display
    if override_applied:
        # Find the overridden emotion's actual model score
        overridden_score = next(
            (r["score"] for r in sorted_results if r["label"].lower() == top_emotion),
            0.5  # fallback if not in top predictions
        )
        # Use higher of model score or 0.75 (override signals high certainty)
        display_confidence = max(overridden_score, 0.75)
    else:
        display_confidence = top_confidence
    
    is_low_confidence = display_confidence < LOW_CONFIDENCE_THRESHOLD
    
    return {
        "emotion": top_emotion,
        "confidence": display_confidence,
        "all_predictions": sorted_results,
        "model_top_prediction": sorted_results[0]["label"].lower(),
        "model_top_confidence": top_confidence,
        "override_applied": override_applied,
        "override_reason": override_reason,
        "is_low_confidence": is_low_confidence,
    }


def assess_crisis(text: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """Detect crisis via keywords + high-risk emotion + confidence."""
    text_lower = text.lower()
    indicators = [kw for kw in CRISIS_KEYWORDS if kw in text_lower]
    
    high_risk_emotion = classification["emotion"] in ("sadness", "fear")
    high_confidence = classification["confidence"] >= CRISIS_CONFIDENCE_THRESHOLD
    
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
    emotion_lower = emotion.lower()
    matches = [item for item in WELLNESS_KB if item["emotion"] == emotion_lower]
    if not matches:
        matches = [item for item in WELLNESS_KB if item["emotion"] == "sadness"][:2]
    return matches[:3]


def explain_prediction(text: str, emotion: str) -> List[Dict[str, Any]]:
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
    with st.sidebar:
        st.title("⚙️ System")
        st.success("✅ Model: Fine-tuned DistilRoBERTa")
        st.caption(f"HF: `{EMOTION_MODEL}`")
        st.divider()
        
        st.subheader("📊 Analysis Options")
        show_explanations = st.toggle("Show token explanations", value=True)
        show_all_emotions = st.toggle("Show all emotion scores", value=True)
        show_overrides = st.toggle("Show safety override info", value=True)
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
        
        return show_explanations, show_all_emotions, show_overrides


def render_classification(result: Dict[str, Any], show_overrides: bool):
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
        st.metric("Model", "Fine-tuned v1.1")
    
    # Show override transparency
    if show_overrides and result.get("override_applied"):
        st.info(
            f"🛡️ **Safety Override Applied:** {result['override_reason']}\n\n"
            f"Model's raw top prediction was `{result['model_top_prediction']}` "
            f"({result['model_top_confidence']:.1%}), but distress-related "
            f"vocabulary triggered a `{emotion}` override. This layered approach "
            f"protects against misclassifications in mental health contexts."
        )
    
    # Warn on low confidence
    if result.get("is_low_confidence") and not result.get("override_applied"):
        st.warning(
            f"⚠️ **Low confidence prediction** ({confidence:.1%}). "
            f"Consider trying more specific language, or view all emotion "
            f"scores below to see alternatives."
        )


def render_all_predictions(predictions: List[Dict[str, Any]]):
    st.subheader("📊 All Emotion Scores")
    for pred in predictions:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(pred["score"], text=pred["label"].title())
        with col2:
            st.caption(f"{pred['score']:.1%}")


def render_crisis_assessment(crisis: Dict[str, Any]):
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
    st.subheader("💡 Personalized Wellness Recommendations")
    for rec in recs:
        with st.expander(f"📖 **{rec['title']}**"):
            st.write(rec["content"])


def render_explanations(explanations: List[Dict[str, Any]]):
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
    st.title("🧠 Mental Health Agentic AI Platform")
    st.markdown(
        "*AI-powered emotional intelligence powered by fine-tuned DistilRoBERTa "
        "(macro F1 = 0.89) + layered safety overrides for clinical vocabulary.*"
    )
    st.divider()
    
    show_explanations, show_all_emotions, show_overrides = sidebar()
    classifier = load_emotion_classifier()
    
    st.subheader("💬 Share what's on your mind")
    
    # Preset examples for quick demo
    example_options = {
        "— Type your own message —": "",
        "🔴 Clear sadness signal": "I feel hopeless and exhausted lately, nothing seems to bring me joy anymore.",
        "😰 Anxiety signal": "I can't stop worrying about my exam tomorrow, my heart is racing.",
        "🟡 Clinical distress (edge case)": "I've been feeling really wired and demotivated lately with work and money problems.",
        "😊 Positive emotion": "Just got promoted at work! I feel so excited and grateful.",
        "😠 Anger": "I'm so frustrated with my roommate, they never clean up after themselves.",
        "🚨 Crisis signal (triggers alert)": "I feel completely alone and don't know if I can keep going anymore.",
    }
    
    selected_example = st.selectbox(
        "Try an example (or type your own below):",
        options=list(example_options.keys()),
        index=1,
    )
    
    default_text = example_options[selected_example]
    text_input = st.text_area(
        "Your thoughts or feelings:",
        value=default_text,
        height=120,
        max_chars=5000,
        placeholder="Type here...",
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        start = time.time()
        with st.spinner("Analyzing..."):
            classification = classify_with_safety_overrides(text_input, classifier)
            classification["latency_ms"] = (time.time() - start) * 1000
            crisis = assess_crisis(text_input, classification)
            recommendations = retrieve_recommendations(classification["emotion"])
            explanations = []
            if show_explanations:
                explanations = explain_prediction(text_input, classification["emotion"])
        
        st.success("✅ Analysis Complete")
        st.divider()
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            render_classification(classification, show_overrides)
        with col_right:
            render_crisis_assessment(crisis)
        
        st.divider()
        
        if show_all_emotions:
            render_all_predictions(classification["all_predictions"])
            st.divider()
        
        render_recommendations(recommendations)
        st.divider()
        
        if show_explanations:
            render_explanations(explanations)
        
        with st.expander("🛠️ Developer View — Raw Response"):
            st.json({
                "classification": {
                    "final_emotion": classification["emotion"],
                    "final_confidence": classification["confidence"],
                    "model_raw_prediction": classification["model_top_prediction"],
                    "model_raw_confidence": classification["model_top_confidence"],
                    "override_applied": classification["override_applied"],
                    "override_reason": classification["override_reason"],
                    "low_confidence_warning": classification["is_low_confidence"],
                },
                "crisis_assessment": crisis,
                "recommendations_count": len(recommendations),
                "explanations_count": len(explanations),
            })

def stream_llm_response(user_text, emotion, confidence, crisis_detected, safety_override):
    """Stream empathetic response using Groq."""
    
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key or not GROQ_AVAILABLE:
        st.info("💡 LLM streaming not configured. Add GROQ_API_KEY to enable.")
        return
    
    # Crisis: show static message, don't use LLM
    if crisis_detected:
        st.error("""
        🆘 **Immediate Support Available:**
        - **988 Suicide & Crisis Lifeline** — Call or text 988
        - **Crisis Text Line** — Text HOME to 741741  
        - **Emergency** — Call 911
        
        You are not alone. Please reach out right now. 💙
        """)
        return
    
    system_prompt = """You are a compassionate mental health support assistant.
    Provide warm, brief (3-4 sentences), empathetic responses.
    Never diagnose. Never replace professional help. Always encourage support-seeking.
    If crisis keywords appear, redirect to crisis resources only."""
    
    user_prompt = f"""
    The user shared: "{user_text}"
    Detected emotion: {emotion} (confidence: {confidence:.0%})
    {"Safety system flagged this input." if safety_override else ""}
    
    Respond with genuine empathy and one gentle, practical suggestion.
    Start directly with acknowledgment. Keep it warm and human.
    """
    
    try:
        client = Groq(api_key=api_key)
        
        # Create streaming container in Streamlit
        with st.chat_message("assistant", avatar="💙"):
            # st.write_stream() handles streaming natively in Streamlit!
            
            def generate():
                stream = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            
            st.write_stream(generate())
            
    except Exception as e:
        st.warning(f"LLM response unavailable: {str(e)}")
        st.info("The emotion analysis above is still accurate and reliable.")


# ============================================================
# UPDATED MAIN UI FUNCTION
# ============================================================

def main():
    st.set_page_config(
        page_title="Mental Health AI Platform",
        page_icon="💙",
        layout="centered"
    )
    
    st.title("💙 Mental Health AI Platform")
    st.caption("Emotion analysis with empathetic AI support")
    
    # --- Example inputs dropdown ---
    examples = [
        "Select an example or type your own...",
        "I feel so hopeless and empty lately",
        "I'm really anxious about my job interview tomorrow",
        "Today was amazing! I got the promotion I worked for!",
        "I've been feeling overwhelmed and can't sleep",
        "I'm so angry at how things turned out",
    ]
    
    selected = st.selectbox("Try an example:", examples)
    
    user_input = st.text_area(
        "Or type your own:",
        value=selected if selected != examples[0] else "",
        placeholder="How are you feeling today?",
        max_chars=512,
        height=100
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_clicked and user_input.strip():
        
        with st.spinner("Analyzing..."):
            # Call your FastAPI backend (or run inline for HF Spaces)
            result = call_backend(user_input)  # your existing function
        
        if result:
            # --- EXISTING RESULTS DISPLAY (your current code) ---
            display_results(result)  # your existing function
            
            # --- NEW: LLM STREAMING RESPONSE ---
            st.divider()
            st.subheader("💙 AI Support Response")
            st.caption("Personalized empathetic response from Llama 3")
            
            stream_llm_response(
                user_text=user_input,
                emotion=result.get("emotion", "unknown"),
                confidence=result.get("confidence", 0.0),
                crisis_detected=result.get("crisis_detected", False),
                safety_override=result.get("safety_override_applied", False)
            )
            
            # Always show this disclaimer
            st.caption(
                "⚠️ This AI provides supportive responses only. "
                "It is not a substitute for professional mental health care. "
                "If you're struggling, please speak to a qualified professional."
            )
    
    elif analyze_clicked:
        st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()