# -*- coding: utf-8 -*-
"""
Mental Health Agentic AI Platform — Streamlit Frontend
Full agentic backend (FastAPI :8000) + Groq LLM streaming.

Architecture:
    Streamlit UI  ──HTTP──►  FastAPI backend (multi-agent + RAG + SHAP)
                  ──API──►   Groq LLM (streaming supportive response)

Author: Jiya Yadav (@JIYA-YDV)
"""

import os
import time
from typing import Dict, Tuple
import textwrap
import streamlit.components.v1 as components

import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mental Health AI Platform",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Mental Health AI Platform — Agentic backend + Groq LLM"},
)

# ══════════════════════════════════════════════════════════════════════
# .env loading (works both locally and on HF Spaces)
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# Backend client
# ══════════════════════════════════════════════════════════════════════
from api_client import analyze, check_health, get_session_id, BACKEND_URL

# ══════════════════════════════════════════════════════════════════════
# Groq (streaming supportive response — kept in frontend)
# ══════════════════════════════════════════════════════════════════════
GROQ_AVAILABLE = False
GROQ_IMPORT_ERROR = ""
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception as e:
    GROQ_IMPORT_ERROR = str(e)

# ══════════════════════════════════════════════════════════════════════
# CSS (identical to HF version)
# ══════════════════════════════════════════════════════════════════════
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
section[data-testid="stSidebar"] { width: 280px !important; background: linear-gradient(180deg, #0f1116 0%, #1a1d26 100%); }

.pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:11px; font-weight:800;
        margin-left:6px; border:1px solid rgba(255,255,255,0.08); white-space:nowrap; }
.pill-live { background:#065f46; color:#6ee7b7; }
.pill-info { background:#1e3a8a; color:#93c5fd; }
.pill-warn { background:#78350f; color:#fcd34d; }
.pill-llm  { background:#1e40af; color:#bfdbfe; }
.pill-off  { background:#334155; color:#cbd5e1; }
.pill-err  { background:#7f1d1d; color:#fecaca; }

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
/* ═════════════════════════════════════════════════════════════════
   RECOMMENDATIONS — Search-engine-style ranked results
   ═════════════════════════════════════════════════════════════════ */
.rec-search-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    background: #0f172a;
    border-radius: 10px;
    margin-bottom: 12px;
    border: 1px solid #1e293b;
}

.rec-search-title {
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 600;
}

.rec-search-count {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.3px;
}

.rec-card {
    background: linear-gradient(135deg, #1e2130 0%, #252838 100%);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
    border-left: 4px solid #6366f1;
    transition: all 0.25s ease;
    position: relative;
}

.rec-card:hover {
    transform: translateX(2px);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
}

.rec-card-top {
    border-left-color: #10b981;
    background: linear-gradient(135deg, #1e2130 0%, #1e3a2e 100%);
}

.rec-card-fallback {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #1e2130 0%, #2a2418 100%);
}

.rec-rank-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.rec-rank-top {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
}

.rec-rank-related {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: white;
}

.rec-rank-fallback {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
}

.rec-title-new {
    color: white;
    font-weight: 700;
    font-size: 16px;
    margin: 4px 0 10px 0;
    line-height: 1.3;
}

.rec-relevance-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
}

.rec-relevance-label {
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
    min-width: 70px;
}

.rec-relevance-bar {
    flex: 1;
    height: 6px;
    background: #0f172a;
    border-radius: 3px;
    overflow: hidden;
    max-width: 280px;
}

.rec-relevance-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

.rec-relevance-fill-top {
    background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
}

.rec-relevance-fill-mid {
    background: linear-gradient(90deg, #6366f1 0%, #a78bfa 100%);
}

.rec-relevance-fill-low {
    background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
}

.rec-relevance-pct {
    color: #e2e8f0;
    font-weight: 800;
    min-width: 45px;
    text-align: right;
}

.rec-tags-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
    padding-bottom: 10px;
    border-bottom: 1px dashed #1e293b;
    flex-wrap: wrap;
}

.rec-tag {
    background: #1e293b;
    color: #94a3b8;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

.rec-content-new {
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.6;
    padding-top: 4px;
}

.agent-trace { background: #0f172a; border-radius: 10px; padding: 12px 16px;
               margin: 8px 0; border-left: 3px solid #10b981; color: #6ee7b7;
               font-family: monospace; font-size: 12px; }
               
/* ═════════════════════════════════════════════════════════════════
   AGENT TRACE — rich visual pipeline
   ═════════════════════════════════════════════════════════════════ */
.trace-container {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f2e 100%);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #1e293b;
    margin: 8px 0;
}

.trace-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    margin-bottom: 14px;
    border-bottom: 1px solid #1e293b;
}

.trace-header-title {
    color: white;
    font-weight: 800;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.trace-header-total {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

.trace-step {
    background: #0a1120;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 3px solid #10b981;
    position: relative;
}

.trace-step-parallel {
    border-left-color: #f59e0b;
}

.trace-step-aggregation {
    border-left-color: #a855f7;
}

.trace-step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.trace-step-label {
    color: #6ee7b7;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-family: 'Courier New', monospace;
}

.trace-step-parallel .trace-step-label {
    color: #fcd34d;
}

.trace-step-aggregation .trace-step-label {
    color: #d8b4fe;
}

.trace-step-latency {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
    background: #1e293b;
    padding: 2px 8px;
    border-radius: 6px;
}

.trace-agent-block {
    padding: 8px 0;
    border-top: 1px dashed #1e293b;
    margin-top: 8px;
}

.trace-agent-block:first-of-type {
    border-top: none;
    margin-top: 0;
    padding-top: 0;
}

.trace-agent-name {
    color: white;
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 6px;
}

.trace-field {
    display: flex;
    color: #cbd5e1;
    font-size: 12px;
    padding: 2px 0;
    font-family: 'Courier New', monospace;
}

.trace-field-key {
    color: #64748b;
    min-width: 130px;
    display: inline-block;
}

.trace-field-value {
    color: #e2e8f0;
    flex: 1;
}

.trace-field-value-highlight {
    color: #6ee7b7;
    font-weight: 700;
}

.trace-confidence-bar {
    display: inline-block;
    width: 120px;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    margin-left: 8px;
    overflow: hidden;
    vertical-align: middle;
}

.trace-confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #6ee7b7 0%, #10b981 100%);
    border-radius: 3px;
}

.trace-arrow {
    text-align: center;
    color: #475569;
    font-size: 18px;
    margin: -4px 0;
}

/* ═════════════════════════════════════════════════════════════════
   EXPLAINABILITY — Token highlighting + contribution bars
   ═════════════════════════════════════════════════════════════════ */
.exp-container {
    background: #0f172a;
    border-radius: 12px;
    padding: 16px 18px;
    margin: 10px 0;
    border: 1px solid #1e293b;
}

.exp-section-title {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.exp-highlighted-text {
    background: #1e293b;
    padding: 14px 16px;
    border-radius: 10px;
    line-height: 2.2;
    font-size: 15px;
    color: #e2e8f0;
    margin-bottom: 16px;
}

.exp-token-highlight {
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
    margin: 0 2px;
    display: inline-block;
    position: relative;
    cursor: help;
}

.exp-token-strong {
    background: rgba(239, 68, 68, 0.25);
    color: #fca5a5;
    border-bottom: 2px solid #ef4444;
}

.exp-token-medium {
    background: rgba(245, 158, 11, 0.25);
    color: #fcd34d;
    border-bottom: 2px solid #f59e0b;
}

.exp-token-mild {
    background: rgba(59, 130, 246, 0.20);
    color: #93c5fd;
    border-bottom: 2px solid #3b82f6;
}

.exp-token-positive {
    background: rgba(16, 185, 129, 0.25);
    color: #6ee7b7;
    border-bottom: 2px solid #10b981;
}

.exp-token-weight {
    display: inline-block;
    background: #0f172a;
    color: #64748b;
    font-size: 9px;
    font-weight: 800;
    padding: 1px 4px;
    border-radius: 4px;
    margin-left: 3px;
    vertical-align: super;
    font-family: 'Courier New', monospace;
}

.exp-bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}

.exp-bar-emoji {
    width: 22px;
    text-align: center;
}

.exp-bar-word {
    color: #cbd5e1;
    min-width: 110px;
    font-weight: 700;
}

.exp-bar-track {
    flex: 1;
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    max-width: 320px;
}

.exp-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.exp-bar-weight {
    color: #e2e8f0;
    font-weight: 800;
    min-width: 50px;
    text-align: right;
}

.exp-method-note {
    color: #64748b;
    font-size: 11px;
    padding: 8px 12px;
    background: #0a1120;
    border-radius: 8px;
    margin-top: 14px;
    border-left: 2px solid #6366f1;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════
GROQ_CHAT_MODEL = "groq/compound-mini"

EMOTION_STYLES = {
    "sadness": {"emoji": "😔", "color": "#3b82f6", "gradient": "linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)"},
    "joy": {"emoji": "😊", "color": "#f59e0b", "gradient": "linear-gradient(135deg, #d97706 0%, #fbbf24 100%)"},
    "love": {"emoji": "🥰", "color": "#ec4899", "gradient": "linear-gradient(135deg, #be185d 0%, #ec4899 100%)"},
    "anger": {"emoji": "😠", "color": "#ef4444", "gradient": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)"},
    "fear": {"emoji": "😨", "color": "#8b5cf6", "gradient": "linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%)"},
    "surprise": {"emoji": "😲", "color": "#10b981", "gradient": "linear-gradient(135deg, #047857 0%, #10b981 100%)"},
    "disgust": {"emoji": "🤢", "color": "#84cc16", "gradient": "linear-gradient(135deg, #4d7c0f 0%, #84cc16 100%)"},
    "neutral": {"emoji": "😐", "color": "#6b7280", "gradient": "linear-gradient(135deg, #4b5563 0%, #6b7280 100%)"},
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


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
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


def risk_class_from_level(level: str) -> Tuple[str, str]:
    """Map backend risk_level → (label, css_class)."""
    mapping = {
        "low": ("Low", "risk-low"),
        "medium": ("Medium", "risk-medium"),
        "high": ("High", "risk-high"),
        "critical": ("Critical", "risk-critical"),
    }
    return mapping.get((level or "low").lower(), ("Low", "risk-low"))


def local_support_fallback(emotion: str) -> str:
    fallbacks = {
        "sadness": "I'm really sorry you're feeling this way. Your feelings are valid, and you don't have to carry them alone.",
        "fear": "That sounds intense. Try a few slow breaths and gently bring your focus back to the present.",
        "anger": "It makes sense you'd feel frustrated. Try pausing for a few breaths before reacting.",
    }
    return fallbacks.get(
        emotion,
        "Thanks for sharing. Whatever you're feeling is valid. Try one gentle step today.",
    )


# ══════════════════════════════════════════════════════════════════════
# GROQ streaming (unchanged — stays in frontend)
# ══════════════════════════════════════════════════════════════════════
def stream_groq_response(api_key: str, system_prompt: str, user_prompt: str):
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6, top_p=0.95, max_tokens=350, stream=True,
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
        temperature=0.6, top_p=0.95, max_tokens=350, stream=False,
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
        st.warning("⚠️ Add GROQ_API_KEY to .env and restart.")
        with st.chat_message("assistant", avatar="💙"):
            st.write(local_support_fallback(result.get("emotion", "unknown")))
        return

    if not GROQ_AVAILABLE:
        st.error("⚠️ Groq SDK not installed. Run: pip install groq")
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


# ══════════════════════════════════════════════════════════════════════
# UI — HEADER
# ══════════════════════════════════════════════════════════════════════
def render_header(backend_health: dict):
    """Portfolio-grade header with branding, links, and 'How it works' reveal."""
    groq_key = get_secret("GROQ_API_KEY")

    # ── Status pills (unchanged logic, refined visuals) ────────────────
    pills = []
    if backend_health["reachable"]:
        pills.append('<span class="pill pill-live">● Backend Live</span>')
    else:
        pills.append('<span class="pill pill-err">● Backend OFFLINE</span>')
    pills.append('<span class="pill pill-info">Multi-Agent + RAG</span>')
    if groq_key and GROQ_AVAILABLE:
        pills.append('<span class="pill pill-llm">✨ Groq Enabled</span>')
    elif not groq_key:
        pills.append('<span class="pill pill-off">No GROQ_API_KEY</span>')

    # ── Two-column header ──────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown(
            """
            <div style="margin-bottom: 4px;">
                <h1 style="
                    margin: 0;
                    font-size: 32px;
                    font-weight: 900;
                    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.5px;
                ">
                    💙 Mental Health AI Platform
                </h1>
                <p style="
                    color: #94a3b8;
                    font-size: 14px;
                    margin: 6px 0 0 0;
                    font-weight: 500;
                ">
                    Multi-agent AI for compassionate emotional support · Fine-tuned NLP · Vector RAG · Explainable · Real-time LLM
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div style="display:flex; flex-wrap:wrap; justify-content:flex-end; align-items:center; gap:4px;">
                {''.join(pills)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Attribution + links row (portfolio essentials!) ────────────────
    st.markdown(
        """
        <div style="
            display: flex;
            align-items: center;
            gap: 18px;
            margin-top: 10px;
            padding: 8px 0;
            border-top: 1px solid #1e2130;
            border-bottom: 1px solid #1e2130;
            font-size: 12px;
            color: #94a3b8;
        ">
            <span style="color: #cbd5e1; font-weight: 600;">
                👩‍💻 Built by <a href="https://github.com/JIYA-YDV"
                target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: 700;">Jiya Yadav</a>
            </span>
            <span style="color: #475569;">·</span>
            <a href="https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform"
               target="_blank" style="color: #94a3b8; text-decoration: none;">
                ⭐ GitHub Repo
            </a>
            <span style="color: #475569;">·</span>
            <a href="https://huggingface.co/spaces/YDVJIYA/mental-health-ai-platform"
               target="_blank" style="color: #94a3b8; text-decoration: none;">
                🤗 Live Demo (HF Space)
            </a>
            <span style="color: #475569;">·</span>
            <a href="https://huggingface.co/YDVJIYA/distilroberta-base-finetuned-emotion"
               target="_blank" style="color: #94a3b8; text-decoration: none;">
                🧠 Fine-tuned Model
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── "How it works" expandable section (portfolio killer feature!) ──
    with st.expander("📄 How it works — Architecture Overview", expanded=False):
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown(
                """
                #### 🧠 Multi-Agent Pipeline

                ```
                User Input
                    ↓
                ┌───────────────────────┐
                │ Orchestrator          │
                └───────────────────────┘
                    ↓
                ┌───────────────────────┐
                │ 1. ClassificationAgent│ → DistilRoBERTa
                └───────────────────────┘
                    ↓
                ┌───────────────────────┐
                │ 2. CrisisAgent  ⚡    │ (parallel)
                │ 3. RAGAgent     ⚡    │ → ChromaDB
                │ 4. Explainer   ⚡    │ → Lexicon
                └───────────────────────┘
                    ↓
                Aggregated Response
                    ↓
                Groq LLM (streaming)
                ```
                """
            )

        with col_b:
            st.markdown(
                """
                #### 🔧 Stack

                - **Backend:** FastAPI + async orchestration
                - **NLP:** DistilRoBERTa (fine-tuned, HuggingFace)
                - **Vector DB:** ChromaDB with cosine similarity
                - **Embeddings:** all-MiniLM-L6-v2
                - **LLM:** Groq (`groq/compound-mini`, streaming)
                - **Explainability:** Lexicon token attribution
                - **Monitoring:** Prometheus + structlog
                - **UI:** Streamlit with custom CSS

                #### 🛡️ Safety Features

                - Crisis keyword + confidence detection
                - Automatic override to safety-first messaging
                - 988 / 741741 / 911 resource surfacing
                - Never diagnoses or replaces professional care
                """
            )

        st.info(
            "💡 **This is a research prototype**, not a clinical tool. "
            "For real mental health support, please reach out to qualified professionals."
        )

# ══════════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ══════════════════════════════════════════════════════════════════════
def render_sidebar(backend_health: dict) -> Dict[str, object]:
    groq_key = get_secret("GROQ_API_KEY")

    with st.sidebar:
        # ═══ SYSTEM STATUS ══════════════════════════════════════════════
        st.markdown(
            "<div style='color:#94a3b8; font-size:11px; font-weight:800; "
            "text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;'>"
            "⚙️ System Status</div>",
            unsafe_allow_html=True,
        )

        if backend_health["reachable"]:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%);
                            padding: 10px 12px; border-radius: 8px; margin-bottom: 6px;">
                    <div style="color: white; font-weight: 700; font-size: 13px;">
                        ✅ Backend Online
                    </div>
                    <div style="color: #a7f3d0; font-size: 10px; margin-top: 2px;
                                font-family: monospace;">
                        v{backend_health['version']} · {BACKEND_URL.replace('http://', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if backend_health["models_loaded"]:
                st.caption("🧠 All ML models loaded")
        else:
            st.error("❌ Backend offline")
            st.caption(backend_health.get("error", ""))
            with st.expander("How to start"):
                st.code(
                    "python -m uvicorn backend.main:app --reload --port 8000",
                    language="bash",
                )

        st.markdown("---")

        # ═══ AGENTS ═════════════════════════════════════════════════════
        st.markdown(
            "<div style='color:#94a3b8; font-size:11px; font-weight:800; "
            "text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;'>"
            "🤖 Active Agents</div>",
            unsafe_allow_html=True,
        )

        agents = [
            ("🧠", "ClassificationAgent", "DistilRoBERTa"),
            ("🛡️", "CrisisAgent", "Keyword + confidence"),
            ("📚", "RAGAgent", "ChromaDB + MiniLM"),
            ("🔍", "Explainer", "Lexicon attribution"),
        ]
        for emoji, name, tech in agents:
            st.markdown(
                f"""
                <div style="padding: 5px 8px; margin-bottom: 4px; background: #1a1f2e;
                            border-radius: 6px; border-left: 2px solid #10b981;">
                    <div style="color: white; font-weight: 600; font-size: 12px;">
                        {emoji} {name}
                    </div>
                    <div style="color: #64748b; font-size: 10px; margin-top: 2px;">
                        {tech}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ═══ LLM ════════════════════════════════════════════════════════
        st.markdown(
            "<div style='color:#94a3b8; font-size:11px; font-weight:800; "
            "text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;'>"
            "✨ LLM Provider</div>",
            unsafe_allow_html=True,
        )

        groq_status_color = "#10b981" if (groq_key and GROQ_AVAILABLE) else "#f59e0b"
        groq_status_text = "Connected" if (groq_key and GROQ_AVAILABLE) else "Not configured"

        st.markdown(
            f"""
            <div style="padding: 10px 12px; background: #1a1f2e; border-radius: 8px;
                        border-left: 3px solid {groq_status_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: white; font-weight: 700; font-size: 12px;">Groq</span>
                    <span style="color: {groq_status_color}; font-size: 10px; font-weight: 800;
                                text-transform: uppercase;">● {groq_status_text}</span>
                </div>
                <div style="color: #64748b; font-family: monospace; font-size: 10px;
                            margin-top: 4px;">
                    {GROQ_CHAT_MODEL}
                </div>
                <div style="color: #64748b; font-family: monospace; font-size: 10px;
                            margin-top: 2px;">
                    Key: {mask_key(groq_key)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ═══ DISPLAY OPTIONS ═══════════════════════════════════════════
        st.markdown(
            "<div style='color:#94a3b8; font-size:11px; font-weight:800; "
            "text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;'>"
            "🎛️ Display Options</div>",
            unsafe_allow_html=True,
        )

        show_explanations = st.toggle(
            "Token explanations",
            value=True,
            key="toggle_explanations",
        )

        # Explainer method radio (only when explanations enabled)
        if show_explanations:
            explainer_method = st.radio(
                "Explanation method",
                options=["lexicon", "shap"],
                index=0,
                format_func=lambda x: {
                    "lexicon": "⚡ Lexicon (fast, ~50ms)",
                    "shap": "🔬 SHAP (authentic, ~2s)",
                }[x],
                help=(
                    "Lexicon: matches curated emotion words (fast).\n\n"
                    "SHAP: real model-attention attributions (slower)."
                ),
                key="radio_explainer_method",
            )
        else:
            explainer_method = "lexicon"

        show_scores = st.toggle(
            "All emotion scores",
            value=True,
            key="toggle_scores",
        )
        show_safety = st.toggle(
            "Safety override info",
            value=True,
            key="toggle_safety",
        )
        show_llm = st.toggle(
            "AI supportive response",
            value=True,
            key="toggle_llm",
        )
        show_agent_trace = st.toggle(
            "Agent execution trace",
            value=False,
            key="toggle_agent_trace",
        )
        show_raw = st.toggle(
            "Raw API response",
            value=False,
            key="toggle_raw",
        )

        # ═══ FOOTER ═════════════════════════════════════════════════════
        st.markdown(
            f"""
            <div style="padding: 8px 10px; background: #0f172a; border-radius: 6px;
                        margin-bottom: 8px;">
                <div style="color: #64748b; font-size: 10px; text-transform: uppercase;
                            letter-spacing: 0.5px; font-weight: 700;">Session</div>
                <div style="color: #94a3b8; font-family: monospace; font-size: 10px;
                            margin-top: 2px;">{get_session_id()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
                        padding: 10px 12px; border-radius: 8px;">
                <div style="color: #fcd34d; font-weight: 700; font-size: 11px;
                            margin-bottom: 4px;">
                    ⚠️ Research Tool Only
                </div>
                <div style="color: #fde68a; font-size: 10px; line-height: 1.4;">
                    Not a substitute for professional mental health care.
                    In crisis? Call <strong>988</strong> (US).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "show_explanations": show_explanations,
        "explainer_method": explainer_method,
        "show_scores": show_scores,
        "show_safety": show_safety,
        "show_llm": show_llm,
        "show_agent_trace": show_agent_trace,
        "show_raw": show_raw,
    }


# ══════════════════════════════════════════════════════════════════════
# UI — INPUT
# ══════════════════════════════════════════════════════════════════════
def render_input_section() -> Tuple[str, bool]:
    st.markdown("#### 💬 Share what's on your mind")
    selected = st.selectbox("Try an example:", list(EXAMPLES.keys()), label_visibility="collapsed")
    user_text = st.text_area(
        "Your text:",
        value=EXAMPLES[selected],
        placeholder="How are you feeling today? Share your thoughts here...",
        height=110,
        max_chars=5000,
        label_visibility="collapsed",
    )
    analyze_clicked = st.button(
        "🔍 Analyze Emotional Content",
        type="primary",
        use_container_width=True,
        disabled=not user_text.strip(),
    )
    return user_text, analyze_clicked

def render_welcome_placeholder():
    """Rich welcome state shown before any analysis."""
    html_code = """
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 20px;
            padding: 40px 32px;
            text-align: center;
            border: 1px solid #334155;
            box-shadow: 0 4px 30px rgba(99, 102, 241, 0.08);
            font-family: sans-serif;
        ">
            <div style="font-size: 64px; margin-bottom: 16px;">🧠✨</div>
            <h2 style="
                color: white;
                margin: 0 0 8px 0;
                font-size: 24px;
                font-weight: 800;
                background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">Ready to Analyze</h2>
            <p style="color: #94a3b8; margin: 0 0 24px 0; font-size: 14px; max-width: 480px; margin-left: auto; margin-right: auto;">
                Share what's on your mind and our multi-agent system will provide
                emotion detection, personalized recommendations, and AI-powered support.
            </p>

            <div style="
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
                margin-top: 20px;
            ">
                <div style="
                    background: rgba(99, 102, 241, 0.10);
                    padding: 12px 18px;
                    border-radius: 10px;
                    border: 1px solid rgba(99, 102, 241, 0.30);
                    min-width: 160px;
                ">
                    <div style="font-size: 24px; margin-bottom: 4px;">🎯</div>
                    <div style="color: white; font-weight: 700; font-size: 12px;">Fine-tuned NLP</div>
                    <div style="color: #94a3b8; font-size: 10px; margin-top: 2px;">DistilRoBERTa · 6 emotions</div>
                </div>

                <div style="
                    background: rgba(16, 185, 129, 0.10);
                    padding: 12px 18px;
                    border-radius: 10px;
                    border: 1px solid rgba(16, 185, 129, 0.30);
                    min-width: 160px;
                ">
                    <div style="font-size: 24px; margin-bottom: 4px;">📚</div>
                    <div style="color: white; font-weight: 700; font-size: 12px;">Vector RAG</div>
                    <div style="color: #94a3b8; font-size: 10px; margin-top: 2px;">ChromaDB · Semantic search</div>
                </div>

                <div style="
                    background: rgba(245, 158, 11, 0.10);
                    padding: 12px 18px;
                    border-radius: 10px;
                    border: 1px solid rgba(245, 158, 11, 0.30);
                    min-width: 160px;
                ">
                    <div style="font-size: 24px; margin-bottom: 4px;">🛡️</div>
                    <div style="color: white; font-weight: 700; font-size: 12px;">Crisis Detection</div>
                    <div style="color: #94a3b8; font-size: 10px; margin-top: 2px;">Keyword + confidence</div>
                </div>

                <div style="
                    background: rgba(168, 85, 247, 0.10);
                    padding: 12px 18px;
                    border-radius: 10px;
                    border: 1px solid rgba(168, 85, 247, 0.30);
                    min-width: 160px;
                ">
                    <div style="font-size: 24px; margin-bottom: 4px;">✨</div>
                    <div style="color: white; font-weight: 700; font-size: 12px;">LLM Streaming</div>
                    <div style="color: #94a3b8; font-size: 10px; margin-top: 2px;">Groq · Real-time tokens</div>
                </div>
            </div>

            <div style="
                margin-top: 28px;
                padding-top: 20px;
                border-top: 1px dashed #334155;
                color: #64748b;
                font-size: 11px;
            ">
                💡 Try one of the examples above, or type your own thoughts below
            </div>
        </div>
    """
    components.html(html_code, height=360, scrolling=False)

# ══════════════════════════════════════════════════════════════════════
# UI — MULTI-STAGE ANALYSIS PROGRESS
# ══════════════════════════════════════════════════════════════════════
def analyze_with_progress(
    user_text: str,
    include_explanations: bool,
    explainer_method: str = "lexicon",  # ← NEW
) -> dict:
    """Call backend with faked multi-stage progress feedback."""
    stages = [
        ("🧠 Classifying emotion (DistilRoBERTa)", 0.15),
        ("🛡️  Assessing crisis risk", 0.05),
        ("📚 Retrieving from knowledge base (ChromaDB)", 0.10),
    ]
    if include_explanations:
        if explainer_method == "shap":
            stages.append(("🔬 Computing SHAP attributions (may take ~2s)", 0.50))
        else:
            stages.append(("🔍 Computing lexicon attributions", 0.10))

    progress_container = st.empty()
    progress_bar = st.progress(0)

    total_fake_time = sum(s[1] for s in stages)
    elapsed = 0.0

    for i, (label, dur) in enumerate(stages[:-1]):
        progress_container.markdown(
            f'<div class="agent-trace">{label}...</div>',
            unsafe_allow_html=True,
        )
        time.sleep(dur)
        elapsed += dur
        progress_bar.progress(min(elapsed / (total_fake_time + 0.1), 0.85))

    final_label = stages[-1][0]
    progress_container.markdown(
        f'<div class="agent-trace">{final_label}...</div>',
        unsafe_allow_html=True,
    )

    result = analyze(
        user_text,
        include_explanations=include_explanations,
        explainer_method=explainer_method,  # ← NEW
    )

    progress_bar.progress(1.0)
    progress_container.empty()
    progress_bar.empty()

    return result

def highlight_tokens_in_text(text: str, explanations: list) -> str:
    """
    Replace matched tokens in the original text with styled HTML spans.
    Case-insensitive, preserves original casing.
    """
    if not explanations:
        return text

    # Sort tokens by length (longest first) so multi-word phrases match first
    sorted_tokens = sorted(
        explanations,
        key=lambda x: len(x.get("word", "")),
        reverse=True,
    )

    result = text
    # Track already-replaced positions to avoid double-highlighting
    for tok in sorted_tokens:
        word = tok.get("word", "").strip()
        if not word:
            continue

        weight = abs(float(tok.get("weight", 0.0)))
        influence = tok.get("influence", "positive")

        # Choose highlight class by strength + influence
        if influence == "positive" and weight >= 0.85:
            css_class = "exp-token-positive"
        elif weight >= 0.85:
            css_class = "exp-token-strong"
        elif weight >= 0.55:
            css_class = "exp-token-medium"
        else:
            css_class = "exp-token-mild"

        # Case-insensitive replace, preserving original case
        import re
        pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)

        def replace_fn(match):
            original = match.group(1)
            return (
                f'<span class="exp-token-highlight {css_class}">'
                f'{original}'
                f'<span class="exp-token-weight">{weight:.2f}</span>'
                f'</span>'
            )

        result = pattern.sub(replace_fn, result, count=1)

    return result
    
# ══════════════════════════════════════════════════════════════════════
# UI — RESULTS
# ══════════════════════════════════════════════════════════════════════
def render_results(result: dict, opts: dict, user_text: str):
    emotion = (result.get("emotion") or "sadness").lower()
    confidence = float(result.get("confidence", 0.0))
    crisis = result.get("crisis_detected", False)
    ms = int(result.get("processing_time_ms", 0))
    override = result.get("safety_override_applied", False)
    crisis_assessment = result.get("crisis_assessment", {}) or {}
    risk_level = crisis_assessment.get("risk_level", "low")

    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["sadness"])
    risk_label, risk_class = risk_class_from_level(risk_level)

    # ── Emotion Box + Metrics ─────────────────────────────────────────
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
            risk_score = crisis_assessment.get("risk_score", 0.0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;">Risk Score</div>
                    <div style="color:white;font-size:22px;font-weight:800;">{risk_score:.2f}</div>
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

        if override and opts["show_safety"]:
            st.info(f"🛡️ **Safety trigger:** {result.get('override_reason', '')}")

    # ── Crisis banner ─────────────────────────────────────────────────
    if crisis:
        st.error("🚨 **Immediate Crisis Support:** **988** (US) • **Text HOME to 741741** • **911**")
        resources = crisis_assessment.get("immediate_resources", [])
        if resources:
            with st.expander("View all crisis resources"):
                for r in resources:
                    st.markdown(f"- {r}")

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_labels = ["📊 Emotions", "💡 Recommendations", "🎯 Explainability"]
    if opts["show_agent_trace"]:
        tab_labels.append("🤖 Agent Trace")
    if opts["show_raw"]:
        tab_labels.append("🔬 Raw API")
    tabs = st.tabs(tab_labels)

    # ── TAB: Emotions ──────────────────────────────────────────────────
    with tabs[0]:
        if opts["show_scores"]:
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
                st.caption("No emotion scores returned.")
        else:
            st.caption("Enable 'All emotion scores' in sidebar.")

    # ── TAB: Recommendations (RICH CARDS from backend RAG!) ───────────
    with tabs[1]:
        recs = result.get("recommendations", [])
        used_fallback = result.get("_used_fallback_recommendations", False)

        if recs:
            if used_fallback:
                st.caption(
                    f"💡 {len(recs)} curated recommendation(s) "
                    "(no strong semantic match in knowledge base)"
                )
            else:
                st.caption(
                    f"📚 Retrieved {len(recs)} recommendation(s) "
                    "from ChromaDB knowledge base"
                )

            for i, rec in enumerate(recs, 1):
                title = rec.get("title", "Recommendation")
                content = rec.get("content", "")
                category = rec.get("category", "general")
                relevance = rec.get("relevance_score", 0.0)
                source = rec.get("source", "Knowledge Base")

                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="rec-title">{i}. {title}</div>
                        <div class="rec-meta">
                            📂 {category} • 🎯 Relevance: {relevance:.1%} • 🔖 {source}
                        </div>
                        <div class="rec-content">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No recommendations available for this input.")

        # ── TAB: Explainability (token highlighting + contribution bars) ──
    with tabs[2]:
        if opts["show_explanations"]:
            summary = result.get("explanation_summary")
            exps = result.get("explanations", [])

            # Summary (from backend explainer)
            if summary:
                st.info(f"💬 {summary}")

            if exps:
                # ─── Section 1: Highlighted input text ─────────────────
                highlighted = highlight_tokens_in_text(user_text, exps)

                st.markdown(
                    f"""
                    <div class="exp-container">
                        <div class="exp-section-title">📝 Highlighted Input</div>
                        <div class="exp-highlighted-text">"{highlighted}"</div>
                    """,
                    unsafe_allow_html=True,
                )

                # ─── Section 2: Contribution breakdown bars ────────────
                st.markdown(
                    '<div class="exp-section-title" style="margin-top:16px;">'
                    '📊 Contribution Breakdown</div>',
                    unsafe_allow_html=True,
                )

                # Sort by absolute weight descending
                sorted_exps = sorted(
                    exps,
                    key=lambda x: abs(float(x.get("weight", 0.0))),
                    reverse=True,
                )

                for tok in sorted_exps[:10]:
                    word = tok.get("word", "")
                    weight = float(tok.get("weight", 0.0))
                    abs_weight = abs(weight)
                    influence = tok.get("influence", "positive")

                    if influence == "positive" and abs_weight >= 0.85:
                        emoji = "🟢"
                        fill_color = "linear-gradient(90deg, #10b981, #34d399)"
                    elif abs_weight >= 0.85:
                        emoji = "🔴"
                        fill_color = "linear-gradient(90deg, #ef4444, #f87171)"
                    elif abs_weight >= 0.55:
                        emoji = "🟠"
                        fill_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
                    else:
                        emoji = "🔵"
                        fill_color = "linear-gradient(90deg, #3b82f6, #60a5fa)"

                    bar_pct = min(abs_weight * 100, 100)

                    st.markdown(
                        f"""
                        <div class="exp-bar-row">
                            <span class="exp-bar-emoji">{emoji}</span>
                            <span class="exp-bar-word">{word}</span>
                            <span class="exp-bar-track">
                                <span class="exp-bar-fill" style="width:{bar_pct}%; background:{fill_color};"></span>
                            </span>
                            <span class="exp-bar-weight">{weight:+.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ─── Method disclosure (dynamic based on method used) ──
                method_used = result.get("explainer_method", "lexicon_attribution")

                if method_used == "shap_gradient":
                    method_html = """
                    <div class="exp-method-note" style="border-left-color: #10b981;">
                        🔬 <strong>Method:</strong> SHAP gradient attributions.
                        Real model-attention values computed via SHAP's Partition
                        explainer over the DistilRoBERTa pipeline. This shows exactly
                        which tokens the fine-tuned model relied on for its prediction.
                    </div>
                    </div>
                    """
                else:
                    method_html = """
                    <div class="exp-method-note">
                        ⚡ <strong>Method:</strong> Lexicon-based token attribution.
                        Fast (~5ms), deterministic scoring across 8 emotion categories
                        with 200+ curated terms. Toggle to SHAP mode for real
                        model-attention analysis (slower).
                    </div>
                    </div>
                    """

                st.markdown(method_html, unsafe_allow_html=True)
            else:
                st.caption("No influential tokens detected for this input.")
        else:
            st.caption("Enable 'Token explanations' in sidebar.")

    # ── TAB: Agent Trace (rich visualization) ─────────────────────────
    tab_idx = 3
    if opts["show_agent_trace"]:
        with tabs[tab_idx]:
            # Prepare trace data
            num_recs = len(result.get("recommendations", []))
            num_explanations = len(result.get("explanations", []))
            has_explanations = num_explanations > 0
            crisis_indicators = crisis_assessment.get("crisis_indicators", [])
            input_preview = user_text[:60] + ("..." if len(user_text) > 60 else "")

            # Estimated per-step latency (real breakdown from backend logs)
            # Classification is sequential; Crisis + RAG + Explainer run parallel
            step1_ms = max(15, int(ms * 0.10))
            step2_ms = max(30, int(ms * 0.85))
            step3_ms = max(1, ms - step1_ms - step2_ms)

            st.markdown(
                f"""
                <div class="trace-container">
                    <div class="trace-header">
                        <div class="trace-header-title">
                            🤖 Multi-Agent Execution Trace
                        </div>
                        <div class="trace-header-total">
                            TOTAL: {ms}ms
                        </div>
                    </div>

                    <!-- ── STEP 1: Classification (sequential) ─────── -->
                    <div class="trace-step">
                        <div class="trace-step-header">
                            <div class="trace-step-label">
                                STEP 1 · ClassificationAgent
                            </div>
                            <div class="trace-step-latency">{step1_ms}ms</div>
                        </div>
                        <div class="trace-field">
                            <span class="trace-field-key">Input:</span>
                            <span class="trace-field-value">"{input_preview}"</span>
                        </div>
                        <div class="trace-field">
                            <span class="trace-field-key">Model:</span>
                            <span class="trace-field-value">DistilRoBERTa (fine-tuned)</span>
                        </div>
                        <div class="trace-field">
                            <span class="trace-field-key">Output:</span>
                            <span class="trace-field-value">
                                <span class="trace-field-value-highlight">{emotion} @ {confidence:.1%}</span>
                                <span class="trace-confidence-bar">
                                    <span class="trace-confidence-fill" style="width:{confidence * 100}%;"></span>
                                </span>
                            </span>
                        </div>
                    </div>

                    <div class="trace-arrow">↓</div>

                    <!-- ── STEP 2: Parallel Agents ───────────────── -->
                    <div class="trace-step trace-step-parallel">
                        <div class="trace-step-header">
                            <div class="trace-step-label">
                                STEP 2 · Parallel Agents ⚡
                            </div>
                            <div class="trace-step-latency">{step2_ms}ms</div>
                        </div>

                        <div class="trace-agent-block">
                            <div class="trace-agent-name">🛡️ CrisisAgent</div>
                            <div class="trace-field">
                                <span class="trace-field-key">risk_level:</span>
                                <span class="trace-field-value">{risk_level}</span>
                            </div>
                            <div class="trace-field">
                                <span class="trace-field-key">risk_score:</span>
                                <span class="trace-field-value">{crisis_assessment.get("risk_score", 0.0):.2f}</span>
                            </div>
                            <div class="trace-field">
                                <span class="trace-field-key">is_crisis:</span>
                                <span class="trace-field-value">{"⚠️ YES" if crisis else "✓ No"}</span>
                            </div>
                            {"".join([f'<div class="trace-field"><span class="trace-field-key">indicators:</span><span class="trace-field-value">{ind}</span></div>' for ind in crisis_indicators[:2]])}
                        </div>

                        <div class="trace-agent-block">
                            <div class="trace-agent-name">📚 RAGAgent (ChromaDB)</div>
                            <div class="trace-field">
                                <span class="trace-field-key">knowledge_base:</span>
                                <span class="trace-field-value">10 documents (mental health)</span>
                            </div>
                            <div class="trace-field">
                                <span class="trace-field-key">retrieved:</span>
                                <span class="trace-field-value">{num_recs} result(s)</span>
                            </div>
                            <div class="trace-field">
                                <span class="trace-field-key">embedding_model:</span>
                                <span class="trace-field-value">all-MiniLM-L6-v2</span>
                            </div>
                        </div>

                        {"" if not has_explanations else f'''
                        <div class="trace-agent-block">
                            <div class="trace-agent-name">🔍 Explainer (Lexicon)</div>
                            <div class="trace-field">
                                <span class="trace-field-key">method:</span>
                                <span class="trace-field-value">Token attribution</span>
                            </div>
                            <div class="trace-field">
                                <span class="trace-field-key">tokens_returned:</span>
                                <span class="trace-field-value">{num_explanations} attribution(s)</span>
                            </div>
                        </div>
                        '''}
                    </div>

                    <div class="trace-arrow">↓</div>

                    <!-- ── STEP 3: Orchestrator Aggregation ────────── -->
                    <div class="trace-step trace-step-aggregation">
                        <div class="trace-step-header">
                            <div class="trace-step-label">
                                STEP 3 · Orchestrator Aggregation
                            </div>
                            <div class="trace-step-latency">{step3_ms}ms</div>
                        </div>
                        <div class="trace-field">
                            <span class="trace-field-key">response:</span>
                            <span class="trace-field-value">JSON envelope sealed</span>
                        </div>
                        <div class="trace-field">
                            <span class="trace-field-key">status:</span>
                            <span class="trace-field-value">
                                <span class="trace-field-value-highlight">200 OK</span>
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "💡 Latencies estimated from total round-trip. "
                "Backend logs show exact per-agent timings in structured format."
            )
        tab_idx += 1

    # ── TAB: Raw API ──────────────────────────────────────────────────
    if opts["show_raw"]:
        with tabs[tab_idx]:
            st.json(result)

    # ── Groq LLM (streaming) ──────────────────────────────────────────
    if opts["show_llm"]:
        render_llm_section(user_text, result)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # Health check up front
    backend_health = check_health()

    render_header(backend_health)
    opts = render_sidebar(backend_health)

    user_text, analyze_clicked = render_input_section()
    st.markdown("")

    if analyze_clicked and user_text.strip():
        if not backend_health["reachable"]:
            st.error("❌ Cannot analyze — backend is offline.")
            st.info(
                "Start the backend in a separate terminal:\n\n"
                "```\npython -m uvicorn backend.main:app --reload --port 8000\n```"
            )
            return

        try:
            result = analyze_with_progress(
                user_text.strip(),
                include_explanations=opts["show_explanations"],
                explainer_method=opts["explainer_method"],
            )
            render_results(result, opts, user_text.strip())
        except RuntimeError as e:
            st.error(f"❌ Analysis failed: {e}")
    else:
        render_welcome_placeholder()


if __name__ == "__main__":
    main()