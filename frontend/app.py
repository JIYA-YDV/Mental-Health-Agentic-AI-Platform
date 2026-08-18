# -*- coding: utf-8 -*-
"""
Mental Health Agentic AI Platform
DistilRoBERTa emotion analysis + Groq LLM supportive response (streaming).

Author: Jiya Yadav (@JIYA-YDV)
Emotion Model: YDVJIYA/distilroberta-base-finetuned-emotion
LLM Provider: Groq (OpenAI-compatible)

UPDATE:
- Fixed Groq chat model to: groq/compound-mini
- Removed model discovery + model picker to keep responses consistent
"""

import os
import time
from typing import Dict, List, Tuple

import streamlit as st

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit UI call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health AI Platform",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Mental Health AI Platform — DistilRoBERTa + Groq LLM supportive responses"},
)

# ──────────────────────────────────────────────────────────────
# Optional .env (local only; harmless on HF Spaces)
# ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    from pathlib import Path

    p = Path(__file__).resolve()
    for ep in [p.parent / ".env", p.parent.parent / ".env"]:
        if ep.exists():
            load_dotenv(ep)
            break
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# Groq import (graceful)
# ──────────────────────────────────────────────────────────────
GROQ_AVAILABLE = False
GROQ_IMPORT_ERROR = ""
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except Exception as e:
    GROQ_IMPORT_ERROR = str(e)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="collapsedControl"] {display:none;}
[data-testid="stToolbar"] {visibility:hidden; height:0px;}
[data-testid="stDecoration"] {display:none;}

.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { width: 260px !important; background: linear-gradient(180deg, #0f1116 0%, #1a1d26 100%); }

.pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:11px; font-weight:800;
        margin-left:6px; border:1px solid rgba(255,255,255,0.08); white-space:nowrap; }
.pill-live { background:#065f46; color:#6ee7b7; }
.pill-info { background:#1e3a8a; color:#93c5fd; }
.pill-warn { background:#78350f; color:#fcd34d; }
.pill-llm  { background:#1e40af; color:#bfdbfe; }
.pill-off  { background:#334155; color:#cbd5e1; }

.metric-card { background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
               border-radius: 12px; padding: 16px; border: 1px solid #2a2d3e; margin-bottom: 8px; }

.emotion-box { border-radius: 16px; padding: 24px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
.emotion-emoji { font-size: 52px; margin-bottom: 4px; }
.emotion-label { font-size: 26px; font-weight: 800; color: white; margin: 0; }
.emotion-confidence { font-size: 14px; color: rgba(255,255,255,0.85); margin-top: 4px; }

.risk-badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:800;
              text-transform:uppercase; letter-spacing:0.5px; }
.risk-low { background:#10b981; color:white; }
.risk-medium { background:#f59e0b; color:white; }
.risk-high { background:#ef4444; color:white; }
.risk-critical { background:#7c2d12; color:white; }

.bar-container { margin: 10px 0; }
.bar-label { display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; color:#e5e7eb; font-weight:600; }
.bar-track { height:8px; background:#2a2d3e; border-radius:4px; overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; transition: width 0.6s ease; }

.stTextArea textarea { border-radius: 10px; border: 1px solid #2a2d3e; background: #1e2130; font-size: 15px; }

.ai-response-header { display:flex; align-items:center; gap:10px; margin:18px 0 10px 0; }
.ai-badge { background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
            color:white; padding:4px 10px; border-radius:999px; font-size:11px; font-weight:800; text-transform:uppercase; }

.welcome-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 16px; padding: 32px; text-align: center; border: 1px solid #334155; margin-top: 20px; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "YDVJIYA/distilroberta-base-finetuned-emotion"

# Fixed Groq model (no picker, no discovery)
GROQ_CHAT_MODEL = "groq/compound-mini"

EMOTION_STYLES = {
    "sadness": {"emoji": "😔", "color": "#3b82f6", "gradient": "linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)"},
    "joy": {"emoji": "😊", "color": "#f59e0b", "gradient": "linear-gradient(135deg, #d97706 0%, #fbbf24 100%)"},
    "love": {"emoji": "🥰", "color": "#ec4899", "gradient": "linear-gradient(135deg, #be185d 0%, #ec4899 100%)"},
    "anger": {"emoji": "😠", "color": "#ef4444", "gradient": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)"},
    "fear": {"emoji": "😨", "color": "#8b5cf6", "gradient": "linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%)"},
    "surprise": {"emoji": "😲", "color": "#10b981", "gradient": "linear-gradient(135deg, #047857 0%, #10b981 100%)"},
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

CRISIS_KEYWORDS = [
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "don't want to be here",
    "dont want to be here",
    "want to die",
    "no way out",
    "better off dead",
    "no reason to live",
    "can't go on",
    "cant go on",
    "wanna die",
    "end it all",
    "take my life",
]

SADNESS_KEYWORDS = [
    "hopeless",
    "worthless",
    "meaningless",
    "empty inside",
    "exhausted",
    "nothing matters",
    "pointless",
    "numb",
    "can't feel anything",
    "cant feel anything",
    "helpless",
    "useless",
]

FEAR_KEYWORDS = [
    "terrified",
    "panic attack",
    "can't breathe",
    "cant breathe",
    "overwhelming anxiety",
    "paralyzed with fear",
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

NEGATIVE_LEXICON = {
    "hopeless": -1.0,
    "worthless": -0.95,
    "meaningless": -0.9,
    "empty": -0.85,
    "exhausted": -0.8,
    "depressed": -0.9,
    "lonely": -0.75,
    "anxious": -0.8,
    "panic": -0.9,
    "terrified": -0.95,
    "overwhelmed": -0.85,
    "angry": -0.8,
    "frustrated": -0.75,
    "suicide": -1.0,
    "suicidal": -1.0,
    "die": -0.9,
}
POSITIVE_LEXICON = {
    "happy": 0.9,
    "joy": 0.95,
    "grateful": 0.85,
    "love": 0.9,
    "proud": 0.8,
    "calm": 0.65,
    "hopeful": 0.8,
    "confident": 0.75,
}

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def get_secret(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def mask_key(k: str) -> str:
    if not k:
        return "NOT FOUND"
    if len(k) < 12:
        return "***"
    return f"{k[:4]}...{k[-4:]}"


def get_risk_level(confidence: float, crisis: bool) -> Tuple[str, str]:
    if crisis:
        return ("Critical", "risk-critical")
    if confidence >= 0.90:
        return ("High", "risk-high")
    if confidence >= 0.60:
        return ("Medium", "risk-medium")
    return ("Low", "risk-low")


def local_support_fallback(emotion: str) -> str:
    if emotion == "sadness":
        return (
            "I’m really sorry you’re feeling this way. Your feelings are valid, and you don’t have to carry them alone. "
            "If you can, try a small grounding step and consider reaching out to someone you trust."
        )
    if emotion == "fear":
        return (
            "That sounds really intense. Try a few slow breaths and gently bring your focus back to the present. "
            "If it keeps feeling overwhelming, consider talking with someone you trust or a professional."
        )
    if emotion == "anger":
        return (
            "It makes sense you’d feel frustrated. Try pausing for a few breaths before reacting. "
            "If you want, write down what triggered it and what you needed in that moment."
        )
    return (
        "Thanks for sharing. Whatever you’re feeling is valid. Try one gentle step—breathing slowly, journaling, "
        "or reaching out to someone you trust."
    )


# ──────────────────────────────────────────────────────────────
# EMOTION MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_classifier():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None, device=-1)


def analyze_text(text: str) -> dict:
    classifier = get_classifier()
    start = time.time()
    raw_results = classifier(text)[0]
    elapsed_ms = int((time.time() - start) * 1000)

    results = sorted(raw_results, key=lambda x: x["score"], reverse=True)
    top = results[0]

    tl = text.lower()
    crisis = any(k in tl for k in CRISIS_KEYWORDS)
    predicted = top["label"].lower()
    override = False
    reason = ""

    if crisis:
        # Safety flow: crisis vocabulary triggers "sadness" for conservative supportive messaging,
        # while the UI separately shows crisis resources.
        predicted = "sadness"
        override = True
        reason = "Crisis vocabulary detected — activated safety flow"
    elif any(k in tl for k in SADNESS_KEYWORDS) and predicted in ["joy", "love", "surprise"]:
        predicted = "sadness"
        override = True
        reason = "Depression vocabulary overrides positive prediction"
    elif any(k in tl for k in FEAR_KEYWORDS) and predicted in ["joy", "love", "surprise"]:
        predicted = "fear"
        override = True
        reason = "Anxiety vocabulary overrides positive prediction"

    explanations = []
    seen = set()
    for w in text.split()[:40]:
        c = w.lower().strip(".,!?;:'\"()[]{}")
        if c in seen or len(c) < 2:
            continue
        seen.add(c)
        if c in NEGATIVE_LEXICON:
            explanations.append({"word": c, "weight": NEGATIVE_LEXICON[c], "influence": "negative"})
        elif c in POSITIVE_LEXICON:
            explanations.append({"word": c, "weight": POSITIVE_LEXICON[c], "influence": "positive"})
    explanations.sort(key=lambda x: abs(x["weight"]), reverse=True)

    return {
        "emotion": predicted,
        "confidence": float(top["score"]),
        "all_emotions": [{"label": r["label"], "score": float(r["score"])} for r in results],
        "crisis_detected": bool(crisis),
        "recommendations": RECOMMENDATIONS_MAP.get(predicted, RECOMMENDATIONS_MAP["sadness"]),
        "explanations": explanations,
        "safety_override_applied": bool(override),
        "override_reason": reason,
        "processing_time_ms": elapsed_ms,
    }


# ──────────────────────────────────────────────────────────────
# GROQ: streaming + non-stream fallback (fixed model)
# ──────────────────────────────────────────────────────────────
def stream_groq_response(api_key: str, system_prompt: str, user_prompt: str):
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=350,  # Groq uses max_tokens (not max_completion_tokens)
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


def groq_nonstream(api_key: str, system_prompt: str, user_prompt: str) -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=350,
        stream=False,
    )
    return resp.choices[0].message.content or ""


def render_llm_section(user_text: str, result: dict):
    st.markdown("---")
    st.markdown(
        """
        <div class="ai-response-header">
            <span style="font-size: 22px;">💙</span>
            <span style="font-size: 18px; font-weight: 800; color: white;">Personalized AI Support</span>
            <span class="ai-badge">Groq</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("crisis_detected", False):
        st.error("🚨 **Immediate Crisis Support:** **988** (US) • **Text HOME to 741741** • **911**")
        return

    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ Add secret `GROQ_API_KEY` (from https://console.groq.com/keys) and restart.")
        with st.chat_message("assistant", avatar="💙"):
            st.write(local_support_fallback(result.get("emotion", "unknown")))
        return

    if not GROQ_AVAILABLE:
        st.error("⚠️ Groq SDK not installed. Add `groq` to requirements.txt.")
        st.caption(f"Import error: {GROQ_IMPORT_ERROR[:200]}")
        with st.chat_message("assistant", avatar="💙"):
            st.write(local_support_fallback(result.get("emotion", "unknown")))
        return

    emotion = result.get("emotion", "unknown")
    confidence = result.get("confidence", 0.0)
    override = result.get("safety_override_applied", False)

    system_prompt = (
        "You are a compassionate mental health support assistant. "
        "Provide warm, brief (3-4 sentences), supportive responses. "
        "Validate feelings and offer one gentle, practical suggestion. "
        "Never diagnose. Never claim certainty. "
        "Do not reveal hidden reasoning or write <think> blocks. "
        "You are not a replacement for professional mental health care."
    )

    override_note = "Safety note: input may be high-risk; respond carefully." if override else ""
    user_prompt = (
        f'The user shared: "{user_text}"\n\n'
        f"Detected emotion: {emotion} (confidence: {confidence:.0%})\n"
        f"{override_note}\n\n"
        "Write a supportive response: validate the feeling and offer one gentle practical suggestion. Keep it 3–4 sentences."
    )

    with st.chat_message("assistant", avatar="💙"):
        try:
            if hasattr(st, "write_stream"):
                st.write_stream(stream_groq_response(api_key, system_prompt, user_prompt))
            else:
                ph = st.empty()
                full = ""
                for piece in stream_groq_response(api_key, system_prompt, user_prompt):
                    full += piece
                    ph.markdown(full + "▌")
                ph.markdown(full)
        except Exception as e:
            st.warning("⚠️ Streaming failed; trying non-stream response...")
            st.caption(f"Debug: {str(e)[:220]}")
            try:
                txt = groq_nonstream(api_key, system_prompt, user_prompt)
                st.write(txt if txt.strip() else local_support_fallback(emotion))
            except Exception as e2:
                st.error("❌ Groq call failed.")
                st.caption(f"Debug: {str(e2)[:220]}")
                st.write(local_support_fallback(emotion))

    st.caption("⚠️ Supportive responses only. Not a substitute for professional mental health care.")


# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
def render_header():
    groq_key = get_secret("GROQ_API_KEY")
    pills = [
        '<span class="pill pill-live">● Live</span>',
        '<span class="pill pill-info">DistilRoBERTa</span>',
    ]
    if groq_key and GROQ_AVAILABLE:
        pills.append('<span class="pill pill-llm">✨ Groq Enabled</span>')
    elif groq_key and not GROQ_AVAILABLE:
        pills.append('<span class="pill pill-warn">Groq SDK Missing</span>')
    else:
        pills.append('<span class="pill pill-off">No GROQ_API_KEY</span>')

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("## 💙 Mental Health AI Platform")
        st.caption("Emotion detection + safety overrides + Groq supportive response (fixed model)")
    with c2:
        st.markdown("".join(pills), unsafe_allow_html=True)


def render_sidebar() -> Dict[str, object]:
    groq_key = get_secret("GROQ_API_KEY")

    with st.sidebar:
        st.markdown("### ⚙️ System")
        st.success("✅ Emotion Model Ready")
        st.caption(f"`{MODEL_NAME}`")

        st.markdown("---")
        st.markdown("### 🔑 Groq Status")
        st.write(f"**Key:** `{mask_key(groq_key)}`")
        st.write(f"**SDK:** `{'OK' if GROQ_AVAILABLE else 'MISSING'}`")

        st.markdown("---")
        st.markdown("### 🧠 LLM Model (Fixed)")
        st.code(GROQ_CHAT_MODEL, language="text")
        st.caption("Locked to one model for consistent behavior (no model switching).")

        st.markdown("---")
        st.markdown("### 🎛️ Display Options")
        show_tokens = st.toggle("Token explanations", value=True)
        show_scores = st.toggle("All emotion scores", value=True)
        show_safety = st.toggle("Safety override info", value=True)
        show_llm = st.toggle("AI supportive response (Groq)", value=True)
        show_raw = st.toggle("Raw API response", value=False)

        st.markdown("---")
        st.warning("⚠️ Disclaimer: Research tool only. In crisis? Call **988** (US).")

    return {
        "show_tokens": show_tokens,
        "show_scores": show_scores,
        "show_safety": show_safety,
        "show_llm": show_llm,
        "show_raw": show_raw,
    }


def render_input_section() -> Tuple[str, bool]:
    st.markdown("#### 💬 Share what's on your mind")
    selected = st.selectbox("Try an example:", list(EXAMPLES.keys()), label_visibility="collapsed")
    user_text = st.text_area(
        "Your text:",
        value=EXAMPLES[selected],
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
    st.markdown(
        """
        <div class="welcome-card">
            <div style="font-size: 52px; margin-bottom: 10px;">🧠</div>
            <h3 style="color: white; margin: 0;">Ready to Analyze</h3>
            <p style="color: #94a3b8; margin-top: 8px; font-size: 14px;">
                Fine-tuned DistilRoBERTa • Safety overrides • Optional Groq support
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result: dict, settings: dict, user_text: str):
    emotion = result.get("emotion", "sadness").lower()
    confidence = result.get("confidence", 0.0)
    crisis = result.get("crisis_detected", False)
    ms = result.get("processing_time_ms", 0)
    override = result.get("safety_override_applied", False)

    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["sadness"])
    risk_label, risk_class = get_risk_level(confidence, crisis)

    col_emotion, col_metrics = st.columns([1, 2])

    with col_emotion:
        st.markdown(
            f"""
            <div class="emotion-box" style="background: {style['gradient']};">
                <div class="emotion-emoji">{style['emoji']}</div>
                <p class="emotion-label">{emotion.title()}</p>
                <p class="emotion-confidence">Confidence: {confidence:.1%}</p>
                <div style="margin-top: 12px;">
                    <span class="risk-badge {risk_class}">{risk_label} Risk</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Latency</div>
                    <div style="color:white;font-size:22px;font-weight:800;">{ms}ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="metric-card">
                    <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Model</div>
                    <div style="color:white;font-size:22px;font-weight:800;">FT</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            crisis_status = "⚠️ Yes" if crisis else "✓ No"
            crisis_color = "#ef4444" if crisis else "#10b981"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Crisis</div>
                    <div style="color:{crisis_color};font-size:22px;font-weight:800;">{crisis_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            override_status = "✓ Applied" if override else "— None"
            override_color = "#f59e0b" if override else "#94a3b8"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Safety</div>
                    <div style="color:{override_color};font-size:22px;font-weight:800;">{override_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if override and settings["show_safety"]:
            st.info(f"🛡️ **Safety override active:** {result.get('override_reason', '')}")

    if crisis:
        st.error("🚨 **Immediate Crisis Support:** **988** (US) • **Text HOME to 741741** • **911**")

    tabs = st.tabs(
        ["📊 Emotions", "💡 Recommendations", "🎯 Token Analysis"]
        + (["🔬 Raw API"] if settings["show_raw"] else [])
    )

    with tabs[0]:
        if settings["show_scores"]:
            all_emotions = result.get("all_emotions", [])
            if all_emotions:
                sorted_emotions = sorted(all_emotions, key=lambda x: x["score"], reverse=True)
                cL, cR = st.columns(2)
                for i, item in enumerate(sorted_emotions):
                    target = cL if i % 2 == 0 else cR
                    with target:
                        emo = item["label"].lower()
                        score = float(item["score"])
                        emoji = EMOTION_STYLES.get(emo, {}).get("emoji", "•")
                        color = EMOTION_STYLES.get(emo, {}).get("color", "#6366f1")
                        pct = score * 100
                        st.markdown(
                            f"""
                            <div class="bar-container">
                                <div class="bar-label">
                                    <span>{emoji} {item['label'].title()}</span>
                                    <span>{pct:.1f}%</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill" style="width:{pct}%; background:{color};"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No detailed emotion scores available.")
        else:
            st.caption("Enable 'All emotion scores' in sidebar.")

    with tabs[1]:
        recs = result.get("recommendations", [])
        for i, r in enumerate(recs, 1):
            st.markdown(f"**{i}. 📚 {r}**")

    with tabs[2]:
        if settings["show_tokens"]:
            exps = result.get("explanations", [])
            if exps:
                st.caption(f"Influential tokens ({len(exps)}):")
                cols = st.columns(4)
                for i, tok in enumerate(exps[:16]):
                    word = tok["word"]
                    weight = tok["weight"]
                    infl = tok["influence"]
                    emoji, color = ("🔴", "#ef4444") if infl == "negative" else ("🟢", "#10b981")
                    with cols[i % 4]:
                        st.markdown(
                            f"""
                            <div style="background:#1e2130;padding:10px;border-radius:8px;border-left:3px solid {color};margin-bottom:8px;">
                                <div style="font-weight:700;color:white;">{emoji} {word}</div>
                                <div style="font-size:11px;color:#94a3b8;">Weight: {weight:+.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No token explanations for this input.")
        else:
            st.caption("Enable 'Token explanations' in sidebar.")

    if settings["show_raw"]:
        with tabs[3]:
            st.json(result)

    if settings["show_llm"]:
        render_llm_section(user_text, result)


def main():
    render_header()
    settings = render_sidebar()

    user_text, analyze_clicked = render_input_section()
    st.markdown("")

    if analyze_clicked and user_text.strip():
        with st.spinner("🧠 Analyzing your input..."):
            result = analyze_text(user_text.strip())
        render_results(result, settings, user_text.strip())
    else:
        render_welcome_placeholder()


if __name__ == "__main__":
    main()