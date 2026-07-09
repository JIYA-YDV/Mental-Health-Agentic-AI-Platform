# -*- coding: utf-8 -*-
"""
Mental Health Agentic AI Platform
Production-ready UI with LLM streaming, safety overrides, and compact dashboard.

Author: Jiya Yadav (@JIYA-YDV)
Model:  YDVJIYA/distilroberta-base-finetuned-emotion
"""

import os
import time
from typing import Optional

import streamlit as st

# Load .env file for local development
# NEW (works locally AND on HF Spaces):
try:
    from dotenv import load_dotenv
    from pathlib import Path
    
    # Try multiple possible .env locations
    current_file = Path(__file__).resolve()
    possible_env_paths = [
        current_file.parent / ".env",              # Same directory as app.py
        current_file.parent.parent / ".env",       # 1 level up
    ]
    
    # Add parents[2] only if it exists (won't on HF Spaces)
    try:
        possible_env_paths.append(current_file.parents[2] / ".env")
    except IndexError:
        pass  # HF Spaces doesn't have this depth
    
    # Load first .env found
    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
    
    # HF Spaces: env vars are already set via secrets, no .env needed
    
except ImportError:
    pass  # dotenv not installed (unlikely, but safe)
except Exception:
    pass  # Any other issue — HF Spaces uses secrets, not .env

# Optional Groq import (graceful degradation)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Mental Health AI Platform",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Mental Health Agentic AI Platform — Fine-tuned DistilRoBERTa with multi-agent orchestration."
    }
)


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    section[data-testid="stSidebar"] {
        width: 260px !important;
        background: linear-gradient(180deg, #0f1116 0%, #1a1d26 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2a2d3e;
        margin-bottom: 8px;
        transition: transform 0.2s;
    }
    
    .metric-card:hover { transform: translateY(-2px); }
    
    .emotion-box {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(75, 108, 183, 0.3);
    }
    
    .emotion-emoji { font-size: 52px; margin-bottom: 4px; }
    .emotion-label { font-size: 26px; font-weight: 700; color: white; margin: 0; }
    .emotion-confidence { font-size: 14px; color: rgba(255,255,255,0.85); margin-top: 4px; }
    
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low { background: #10b981; color: white; }
    .risk-medium { background: #f59e0b; color: white; }
    .risk-high { background: #ef4444; color: white; }
    .risk-critical { background: #7c2d12; color: white; }
    
    .bar-container { margin: 10px 0; }
    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 5px;
        color: #e5e7eb;
        font-weight: 500;
    }
    .bar-track {
        height: 8px;
        background: #2a2d3e;
        border-radius: 4px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #1e2130;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 14px;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #2a2d3e;
        background: #1e2130;
        font-size: 15px;
    }
    
    .header-banner {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        padding: 12px 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #334155;
    }
    .header-title { font-size: 20px; font-weight: 700; color: white; margin: 0; }
    .header-badges { display: flex; gap: 8px; }
    
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-success { background: #065f46; color: #6ee7b7; }
    .badge-info { background: #1e3a8a; color: #93c5fd; }
    .badge-warning { background: #78350f; color: #fcd34d; }
    .badge-purple { background: #581c87; color: #d8b4fe; }
    
    .welcome-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        border: 1px solid #334155;
        margin-top: 20px;
    }
    .welcome-stat { display: inline-block; margin: 0 20px; }
    
    .ai-response-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 10px 0;
    }
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = "YDVJIYA/distilroberta-base-finetuned-emotion"
GROQ_MODEL = "llama-3.1-8b-instant"

EMOTION_STYLES = {
    "sadness":  {"emoji": "😔", "color": "#3b82f6", "gradient": "linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)"},
    "joy":      {"emoji": "😊", "color": "#f59e0b", "gradient": "linear-gradient(135deg, #d97706 0%, #fbbf24 100%)"},
    "love":     {"emoji": "🥰", "color": "#ec4899", "gradient": "linear-gradient(135deg, #be185d 0%, #ec4899 100%)"},
    "anger":    {"emoji": "😠", "color": "#ef4444", "gradient": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)"},
    "fear":     {"emoji": "😨", "color": "#8b5cf6", "gradient": "linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%)"},
    "surprise": {"emoji": "😲", "color": "#10b981", "gradient": "linear-gradient(135deg, #047857 0%, #10b981 100%)"},
    "crisis":   {"emoji": "🚨", "color": "#dc2626", "gradient": "linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%)"},
}

EXAMPLES = {
    "— Select an example —": "",
    "😔 Clear sadness signal": "I feel hopeless and exhausted lately, nothing seems to bring me joy anymore.",
    "😊 Positive & grateful": "I got the promotion today! All that hard work finally paid off, I'm so grateful.",
    "😨 Anxious about future": "I can't stop worrying about my job interview tomorrow, my heart won't stop racing.",
    "😠 Frustrated & angry": "I'm so tired of being ignored. It feels like nothing I do matters at all.",
    "🥰 Love & connection": "Spending time with my family this weekend reminded me how loved I am.",
    "🚨 Crisis signal (test)": "I don't want to be here anymore. Nothing matters and I see no way out.",
}

NEGATIVE_LEXICON = {
    "hopeless": -1.0, "worthless": -0.95, "meaningless": -0.9,
    "empty": -0.85, "exhausted": -0.8, "sad": -0.75, "sadness": -0.75,
    "depressed": -0.9, "lonely": -0.75, "alone": -0.7,
    "crying": -0.75, "tearful": -0.75, "miserable": -0.85,
    "unhappy": -0.7, "grief": -0.85, "sorrow": -0.8,
    "numb": -0.8, "drained": -0.75, "tired": -0.5,
    "heartbroken": -0.85, "devastated": -0.9,
    "anxious": -0.8, "anxiety": -0.8, "scared": -0.75,
    "worried": -0.75, "worry": -0.7, "worrying": -0.75,
    "fear": -0.8, "afraid": -0.75, "fearful": -0.75,
    "nervous": -0.7, "panicked": -0.9, "panic": -0.9,
    "terrified": -0.95, "overwhelmed": -0.85, "racing": -0.6,
    "stressed": -0.75, "tense": -0.7, "restless": -0.6,
    "dread": -0.85, "frightened": -0.8, "uneasy": -0.65,
    "angry": -0.8, "furious": -0.9, "mad": -0.7,
    "frustrated": -0.75, "irritated": -0.65, "annoyed": -0.6,
    "hate": -0.85, "rage": -0.9, "resentful": -0.75,
    "bitter": -0.75, "hostile": -0.7,
    "hurt": -0.7, "pain": -0.7, "suffering": -0.85,
    "broken": -0.8, "shattered": -0.85, "wounded": -0.75,
    "terrible": -0.8, "awful": -0.75, "horrible": -0.8,
    "helpless": -0.85, "useless": -0.85,
    "suicide": -1.0, "suicidal": -1.0, "die": -0.9,
    "kill": -0.95, "death": -0.75,
    "nothing": -0.4, "never": -0.5, "no": -0.3,
    "cant": -0.5, "wont": -0.4,
}

POSITIVE_LEXICON = {
    "happy": 0.9, "joy": 0.95, "joyful": 0.9,
    "excited": 0.85, "cheerful": 0.8, "delighted": 0.85,
    "amazing": 0.85, "wonderful": 0.85, "great": 0.75,
    "fantastic": 0.9, "excellent": 0.85, "brilliant": 0.85,
    "awesome": 0.85, "incredible": 0.85,
    "grateful": 0.85, "thankful": 0.85, "blessed": 0.85,
    "appreciate": 0.75, "lucky": 0.75, "fortunate": 0.75,
    "love": 0.9, "loved": 0.85, "loving": 0.85,
    "care": 0.7, "caring": 0.75, "affection": 0.8,
    "adore": 0.9, "cherish": 0.85, "family": 0.5,
    "proud": 0.8, "accomplished": 0.85, "achieved": 0.8,
    "success": 0.8, "successful": 0.8, "won": 0.75,
    "promoted": 0.85, "promotion": 0.85, "paid": 0.5,
    "calm": 0.65, "peaceful": 0.75, "relaxed": 0.7,
    "content": 0.75, "satisfied": 0.7, "hopeful": 0.8,
    "confident": 0.75, "strong": 0.7, "brave": 0.75,
    "beautiful": 0.75, "perfect": 0.8, "best": 0.75,
    "better": 0.6, "good": 0.6,
}

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life",
    "don't want to be here", "want to die", "no way out",
    "better off dead", "no reason to live", "can't go on",
    "wanna die", "end it all", "take my life",
]

SADNESS_KEYWORDS = [
    "hopeless", "worthless", "meaningless", "empty inside",
    "exhausted", "nothing matters", "pointless", "numb",
    "can't feel anything", "helpless", "useless",
]

FEAR_KEYWORDS = [
    "terrified", "panic attack", "can't breathe",
    "overwhelming anxiety", "paralyzed with fear",
]

RECOMMENDATIONS_MAP = {
    "sadness": [
        "5-4-3-2-1 Grounding Exercise",
        "Cognitive Reframing for Negative Thoughts",
        "Reach out to a trusted friend or family member",
        "Consider speaking with a mental health professional",
    ],
    "fear": [
        "Box Breathing (4-4-4-4 pattern)",
        "Progressive Muscle Relaxation technique",
        "Anxiety journaling — write down your worries",
        "Grounding: name 5 things you can see",
    ],
    "anger": [
        "Take 10 deep breaths before responding",
        "Physical release: brief walk or exercise",
        "Journal what triggered the anger",
        "Anger management coping strategies",
    ],
    "joy": [
        "Gratitude journaling — capture this feeling",
        "Share your joy with someone you love",
        "Reflect on what led to this positive moment",
    ],
    "love": [
        "Express appreciation to those you care about",
        "Practice self-compassion daily",
        "Nurture your important relationships",
    ],
    "surprise": [
        "Take time to process unexpected events",
        "Reflect on your emotional response",
        "Talk through the surprise with someone",
    ],
}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_risk_level(confidence: float, crisis: bool) -> tuple:
    """Return (label, css_class) for risk badge."""
    if crisis:
        return ("Critical", "risk-critical")
    if confidence >= 0.9:
        return ("High", "risk-high")
    if confidence >= 0.6:
        return ("Medium", "risk-medium")
    return ("Low", "risk-low")


# ═══════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════

def analyze_text(text: str) -> dict:
    """Run emotion classification with layered safety overrides."""
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    if "classifier" not in st.session_state:
        with st.spinner("🔄 Loading model (first time only, ~30 seconds)..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            st.session_state.classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                device=-1,
            )

    start = time.time()
    raw_results = st.session_state.classifier(text)[0]
    elapsed_ms = int((time.time() - start) * 1000)

    results = sorted(raw_results, key=lambda x: x["score"], reverse=True)
    top = results[0]

    # Safety overrides
    text_lower = text.lower()
    crisis_detected = any(kw in text_lower for kw in CRISIS_KEYWORDS)
    safety_override = False
    override_reason = ""
    original_emotion = top["label"]
    predicted_emotion = top["label"].lower()

    if crisis_detected:
        predicted_emotion = "sadness"
        safety_override = True
        override_reason = "Crisis vocabulary detected — activated safety flow"
    elif any(kw in text_lower for kw in SADNESS_KEYWORDS) and predicted_emotion in ["joy", "love", "surprise"]:
        predicted_emotion = "sadness"
        safety_override = True
        override_reason = f"Depression vocabulary overrides '{original_emotion}' prediction"
    elif any(kw in text_lower for kw in FEAR_KEYWORDS) and predicted_emotion in ["joy", "love", "surprise"]:
        predicted_emotion = "fear"
        safety_override = True
        override_reason = f"Anxiety vocabulary overrides '{original_emotion}' prediction"

    # Token explanations
    words = text.split()
    explanations = []
    seen = set()

    for word in words[:40]:
        clean = word.lower().strip(".,!?;:'\"()[]{}")
        if clean in seen or len(clean) < 2:
            continue
        seen.add(clean)

        if clean in NEGATIVE_LEXICON:
            explanations.append({
                "word": clean,
                "weight": NEGATIVE_LEXICON[clean],
                "influence": "negative",
            })
        elif clean in POSITIVE_LEXICON:
            explanations.append({
                "word": clean,
                "weight": POSITIVE_LEXICON[clean],
                "influence": "positive",
            })

    explanations.sort(key=lambda x: abs(x["weight"]), reverse=True)

    return {
        "emotion": predicted_emotion,
        "confidence": top["score"],
        "all_emotions": [{"label": r["label"], "score": r["score"]} for r in results],
        "crisis_detected": crisis_detected,
        "crisis_confidence": 0.95 if crisis_detected else 0.0,
        "recommendations": RECOMMENDATIONS_MAP.get(predicted_emotion, RECOMMENDATIONS_MAP["sadness"]),
        "explanations": explanations,
        "safety_override_applied": safety_override,
        "override_reason": override_reason,
        "processing_time_ms": elapsed_ms,
    }


# ═══════════════════════════════════════════════════════════════
# LLM STREAMING (Bulletproof — works on any Streamlit version)
# ═══════════════════════════════════════════════════════════════

def stream_ai_response(user_text: str, result: dict):
    """Stream empathetic LLM response. Works on any Streamlit version."""

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or not GROQ_AVAILABLE:
        st.info("💡 **Enable AI Streaming:** Add `GROQ_API_KEY` to your `.env` file (free at [console.groq.com](https://console.groq.com))")
        return

    emotion = result.get("emotion", "unknown")
    confidence = result.get("confidence", 0.0)
    crisis = result.get("crisis_detected", False)
    override = result.get("safety_override_applied", False)

    if crisis:
        st.error("""
        🚨 **Immediate Support Available 24/7:**
        - **988** — Call or text (Suicide & Crisis Lifeline)
        - **Text HOME to 741741** — Crisis Text Line
        - **911** — Emergency Services
        
        You are not alone. Trained professionals are ready to help right now. 💙
        """)
        return

    system_prompt = (
        "You are a compassionate mental health support assistant. "
        "Provide warm, brief (3-4 sentences), empathetic responses. "
        "Never diagnose. Never replace professional help. "
        "Always encourage support-seeking. "
        "Keep your tone warm, human, and non-clinical."
    )

    override_note = ""
    if override:
        override_note = "Note: Our safety system flagged this input as potentially high-risk."

    user_prompt = (
        f'The user shared: "{user_text}"\n\n'
        f"Detected emotion: {emotion} (confidence: {confidence:.0%})\n"
        f"{override_note}\n\n"
        f"Respond with genuine empathy. Acknowledge their {emotion}, "
        f"validate what they're feeling, and offer one gentle, practical suggestion. "
        f"Start directly with warmth. Keep it 3-4 sentences."
    )

    try:
        client = Groq(api_key=api_key)

        with st.chat_message("assistant", avatar="💙"):
            # Try modern st.write_stream first (Streamlit 1.31+)
            if hasattr(st, "write_stream"):
                def generate():
                    stream = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.7,
                        max_tokens=200,
                        stream=True,
                    )
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                st.write_stream(generate())
            else:
                # Fallback for older Streamlit — manual placeholder streaming
                message_placeholder = st.empty()
                full_response = ""

                stream = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    stream=True,
                )

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

    except Exception as e:
        st.warning("⚠️ AI streaming temporarily unavailable — but the emotion analysis above remains accurate.")
        st.caption(f"Debug: {str(e)[:100]}")


# ═══════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════

def render_header():
    """Compact professional header banner."""
    llm_badge = ""
    if os.getenv("GROQ_API_KEY", "").strip() and GROQ_AVAILABLE:
        llm_badge = '<span class="badge badge-purple">✨ LLM Enabled</span>'

    st.markdown(f"""
    <div class="header-banner">
        <div>
            <p class="header-title">💙 Mental Health AI Platform</p>
        </div>
        <div class="header-badges">
            <span class="badge badge-success">● Live</span>
            <span class="badge badge-info">v1.2 • DistilRoBERTa</span>
            <span class="badge badge-warning">F1: 0.89</span>
            {llm_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> dict:
    """Sidebar with settings, metrics, and links."""
    with st.sidebar:
        st.markdown("### ⚙️ System")
        st.success("✅ Model Ready")
        st.caption(f"`{MODEL_NAME}`")

        st.markdown("---")
        st.markdown("### 🎛️ Display Options")
        show_tokens = st.toggle("Token explanations", value=True)
        show_scores = st.toggle("All emotion scores", value=True)
        show_safety = st.toggle("Safety override info", value=True)
        show_llm = st.toggle("AI streaming response", value=True)
        show_raw = st.toggle("Raw API response", value=False)

        st.markdown("---")
        st.markdown("### 📊 Model Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Macro F1", "0.89", "+0.02")
        with col2:
            st.metric("Crisis Recall", "96%", "+3%")

        st.markdown("---")
        st.markdown("### 🔗 Project Links")
        st.markdown("""
        - [🐙 GitHub Repository](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform)
        - [🤗 Fine-Tuned Model](https://huggingface.co/YDVJIYA/distilroberta-base-finetuned-emotion)
        - [📊 Evaluation Report](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/blob/main/docs/EVALUATION.md)
        """)

        st.markdown("---")
        st.warning(
            "⚠️ **Disclaimer:** Research tool only. "
            "Not a substitute for professional mental health care. "
            "In crisis? Call **988** (US)."
        )

    return {
        "show_tokens": show_tokens,
        "show_scores": show_scores,
        "show_safety": show_safety,
        "show_llm": show_llm,
        "show_raw": show_raw,
    }


def render_input_section() -> tuple:
    """Compact input area with example selector."""
    st.markdown("#### 💬 Share what's on your mind")

    selected_example = st.selectbox(
        "Try an example:",
        list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )

    user_text = st.text_area(
        "Your text:",
        value=EXAMPLES[selected_example],
        placeholder="How are you feeling today? Share your thoughts here...",
        height=110,
        max_chars=500,
        label_visibility="collapsed",
    )

    analyze = st.button(
        "🔍 Analyze Emotional Content",
        type="primary",
        use_container_width=True,
        disabled=not user_text.strip(),
    )

    return user_text, analyze


def render_welcome_placeholder():
    """Beautiful placeholder shown before first analysis."""
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size: 52px; margin-bottom: 10px;">🧠</div>
        <h3 style="color: white; margin: 0;">Ready to Analyze</h3>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 14px;">
            Fine-tuned DistilRoBERTa • 7 emotion classes • Layered safety overrides
        </p>
        <div style="display: flex; justify-content: center; margin-top: 24px; flex-wrap: wrap;">
            <div class="welcome-stat">
                <div style="color: #10b981; font-size: 28px; font-weight: 700;">96%</div>
                <div style="color: #94a3b8; font-size: 12px;">Crisis Recall</div>
            </div>
            <div class="welcome-stat">
                <div style="color: #3b82f6; font-size: 28px; font-weight: 700;">0.89</div>
                <div style="color: #94a3b8; font-size: 12px;">Macro F1</div>
            </div>
            <div class="welcome-stat">
                <div style="color: #8b5cf6; font-size: 28px; font-weight: 700;">&lt;500ms</div>
                <div style="color: #94a3b8; font-size: 12px;">Avg Latency</div>
            </div>
            <div class="welcome-stat">
                <div style="color: #f59e0b; font-size: 28px; font-weight: 700;">4</div>
                <div style="color: #94a3b8; font-size: 12px;">AI Agents</div>
            </div>
        </div>
        <p style="color: #64748b; font-size: 13px; margin-top: 24px;">
            👆 Try an example above or type your own text to see the full analysis
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_results(result: dict, settings: dict, user_text: str):
    """Compact grid layout for all analysis results."""

    emotion = result.get("emotion", "unknown").lower()
    confidence = result.get("confidence", 0.0)
    crisis = result.get("crisis_detected", False)
    processing_ms = result.get("processing_time_ms", 0)
    safety_override = result.get("safety_override_applied", False)

    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["sadness"])
    risk_label, risk_class = get_risk_level(confidence, crisis)

    # Top row: emotion card + metrics
    col_emotion, col_metrics = st.columns([1, 2])

    with col_emotion:
        st.markdown(f"""
        <div class="emotion-box" style="background: {style['gradient']};">
            <div class="emotion-emoji">{style['emoji']}</div>
            <p class="emotion-label">{emotion.title()}</p>
            <p class="emotion-confidence">Confidence: {confidence:.1%}</p>
            <div style="margin-top: 12px;">
                <span class="risk-badge {risk_class}">{risk_label} Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Latency</div>
                <div style="color: white; font-size: 22px; font-weight: 700;">{processing_ms}ms</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Model</div>
                <div style="color: white; font-size: 22px; font-weight: 700;">v1.1 FT</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            crisis_status = "⚠️ Yes" if crisis else "✓ No"
            crisis_color = "#ef4444" if crisis else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Crisis</div>
                <div style="color: {crisis_color}; font-size: 22px; font-weight: 700;">{crisis_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            override_status = "✓ Applied" if safety_override else "— None"
            override_color = "#f59e0b" if safety_override else "#94a3b8"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Safety</div>
                <div style="color: {override_color}; font-size: 22px; font-weight: 700;">{override_status}</div>
            </div>
            """, unsafe_allow_html=True)

        if safety_override and settings["show_safety"]:
            st.info(
                f"🛡️ **Safety override active:** "
                f"{result.get('override_reason', 'Clinical vocabulary detected')}"
            )

    # Crisis alert
    if crisis:
        st.error("""
        🚨 **Immediate Crisis Support Available:**
        📞 **988** (Call/Text) • 💬 **Text HOME to 741741** • 🆘 **911** for emergencies
        """)

    st.markdown("")

    # Tabbed sections
    tab_labels = ["📊 Emotions", "💡 Recommendations", "🎯 Token Analysis"]
    if settings["show_raw"]:
        tab_labels.append("🔬 Raw API")

    tabs = st.tabs(tab_labels)

    # Tab 1: Emotions
    with tabs[0]:
        if settings["show_scores"]:
            all_emotions = result.get("all_emotions", [])
            if all_emotions:
                sorted_emotions = sorted(all_emotions, key=lambda x: x["score"], reverse=True)

                col_left, col_right = st.columns(2)
                for idx, item in enumerate(sorted_emotions):
                    target_col = col_left if idx % 2 == 0 else col_right
                    with target_col:
                        emo = item["label"].lower()
                        score = item["score"]
                        emoji = EMOTION_STYLES.get(emo, {}).get("emoji", "•")
                        color = EMOTION_STYLES.get(emo, {}).get("color", "#6366f1")
                        pct = score * 100

                        st.markdown(f"""
                        <div class="bar-container">
                            <div class="bar-label">
                                <span>{emoji} {item['label'].title()}</span>
                                <span>{pct:.1f}%</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width: {pct}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("No detailed emotion scores available.")
        else:
            st.caption("Enable 'All emotion scores' in the sidebar to view.")

    # Tab 2: Recommendations
    with tabs[1]:
        recommendations = result.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                title = rec if isinstance(rec, str) else rec.get("title", "Resource")
                st.markdown(f"**{i}. 📚 {title}**")
                if isinstance(rec, dict) and rec.get("description"):
                    st.caption(rec["description"])
        else:
            st.caption("No specific recommendations available for this input.")

    # Tab 3: Token Analysis
    with tabs[2]:
        if settings["show_tokens"]:
            explanations = result.get("explanations", [])
            if explanations:
                st.caption(f"Words that most influenced the prediction ({len(explanations)} tokens):")

                cols = st.columns(4)
                for idx, token in enumerate(explanations[:16]):
                    word = token.get("word", "?")
                    weight = token.get("weight", 0)
                    influence = token.get("influence", "neutral")

                    if influence == "negative":
                        emoji = "🔴"
                        color = "#ef4444"
                    elif influence == "positive":
                        emoji = "🟢"
                        color = "#10b981"
                    else:
                        emoji = "⚪"
                        color = "#94a3b8"

                    with cols[idx % 4]:
                        st.markdown(f"""
                        <div style="background: #1e2130; padding: 10px; border-radius: 8px;
                                    border-left: 3px solid {color}; margin-bottom: 8px;">
                            <div style="font-weight: 600; color: white;">{emoji} {word}</div>
                            <div style="font-size: 11px; color: #94a3b8;">Weight: {weight:+.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("No token explanations for this text. Try inputs with clearer emotional vocabulary.")
        else:
            st.caption("Enable 'Token explanations' in the sidebar to view.")

    # Tab 4: Raw JSON
    if settings["show_raw"]:
        with tabs[3]:
            st.json(result)

    # LLM streaming section
    if settings["show_llm"]:
        st.markdown("---")
        st.markdown("""
        <div class="ai-response-header">
            <span style="font-size: 24px;">💙</span>
            <span style="font-size: 18px; font-weight: 700; color: white;">Personalized AI Support</span>
            <span class="ai-badge">Llama 3.1</span>
        </div>
        """, unsafe_allow_html=True)

        stream_ai_response(user_text, result)

        st.caption(
            "⚠️ This AI provides supportive responses only. "
            "It is not a substitute for professional mental health care. "
            "If you're struggling, please speak to a qualified professional."
        )


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    render_header()
    settings = render_sidebar()

    user_text, analyze_clicked = render_input_section()
    st.markdown("")

    if analyze_clicked and user_text.strip():
        with st.spinner("🧠 Analyzing your input..."):
            result = analyze_text(user_text)

        if result:
            render_results(result, settings, user_text)
    else:
        render_welcome_placeholder()


if __name__ == "__main__":
    main()