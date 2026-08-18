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

.rec-card { background: linear-gradient(135deg, #1e2130 0%, #252838 100%);
            border-radius: 12px; padding: 16px; margin-bottom: 12px;
            border-left: 4px solid #6366f1; }
.rec-title { color: white; font-weight: 700; font-size: 15px; margin-bottom: 4px; }
.rec-meta { color: #94a3b8; font-size: 11px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.4px; }
.rec-content { color: #cbd5e1; font-size: 13px; line-height: 1.55; }

.agent-trace { background: #0f172a; border-radius: 10px; padding: 12px 16px;
               margin: 8px 0; border-left: 3px solid #10b981; color: #6ee7b7;
               font-family: monospace; font-size: 12px; }
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
    groq_key = get_secret("GROQ_API_KEY")
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

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("## 💙 Mental Health AI Platform")
        st.caption("Agentic backend (FastAPI) + ChromaDB RAG + SHAP explainability + Groq LLM")
    with c2:
        st.markdown("".join(pills), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ══════════════════════════════════════════════════════════════════════
def render_sidebar(backend_health: dict) -> Dict[str, object]:
    groq_key = get_secret("GROQ_API_KEY")

    with st.sidebar:
        st.markdown("### ⚙️ System")

        if backend_health["reachable"]:
            st.success(f"✅ Backend v{backend_health['version']}")
            st.caption(f"`{BACKEND_URL}`")
            if backend_health["models_loaded"]:
                st.caption("🧠 ML models loaded")
            else:
                st.warning("⚠️ Models not fully loaded")
        else:
            st.error("❌ Backend offline")
            st.caption(backend_health.get("error", ""))
            st.code(
                "python -m uvicorn backend.main:app --reload --port 8000",
                language="bash",
            )

        st.markdown("---")
        st.markdown("### 🤖 Agents")
        st.caption("• Classification (DistilRoBERTa)")
        st.caption("• Crisis Detection")
        st.caption("• RAG (ChromaDB)")
        st.caption("• Explainer (SHAP)")

        st.markdown("---")
        st.markdown("### 🔑 Groq Status")
        st.write(f"**Key:** `{mask_key(groq_key)}`")
        st.write(f"**SDK:** `{'OK' if GROQ_AVAILABLE else 'MISSING'}`")

        st.markdown("---")
        st.markdown("### 🧠 LLM Model")
        st.code(GROQ_CHAT_MODEL, language="text")

        st.markdown("---")
        st.markdown("### 🎛️ Display Options")
        show_explanations = st.toggle("SHAP token explanations", value=True)
        show_scores = st.toggle("All emotion scores", value=True)
        show_safety = st.toggle("Safety override info", value=True)
        show_llm = st.toggle("AI supportive response (Groq)", value=True)
        show_agent_trace = st.toggle("Agent execution trace", value=False)
        show_raw = st.toggle("Raw API response", value=False)

        st.markdown("---")
        st.caption(f"**Session:** `{get_session_id()}`")
        st.warning("⚠️ Research tool only. In crisis? Call **988** (US).")

    return {
        "show_explanations": show_explanations,
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
    st.markdown(
        """
        <div class="welcome-card">
            <div style="font-size: 52px; margin-bottom: 10px;">🧠</div>
            <h3 style="color: white; margin: 0;">Ready to Analyze</h3>
            <p style="color: #94a3b8; margin-top: 8px; font-size: 14px;">
                Multi-Agent Orchestration • ChromaDB RAG • SHAP Explainability • Groq LLM
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# UI — MULTI-STAGE ANALYSIS PROGRESS
# ══════════════════════════════════════════════════════════════════════
def analyze_with_progress(user_text: str, include_explanations: bool) -> dict:
    """
    Call backend with faked multi-stage progress feedback.
    Reveals the agentic architecture to the user visually.
    """
    stages = [
        ("🧠 Classifying emotion (DistilRoBERTa)", 0.15),
        ("🛡️  Assessing crisis risk", 0.05),
        ("📚 Retrieving from knowledge base (ChromaDB)", 0.10),
    ]
    if include_explanations:
        stages.append(("🔍 Computing SHAP explanations", 0.30))

    progress_container = st.empty()
    progress_bar = st.progress(0)

    total_fake_time = sum(s[1] for s in stages)
    elapsed = 0.0

    # Start the actual backend call in the foreground.
    # We show fake progress up to ~90% while it runs, then jump to 100%.
    for i, (label, dur) in enumerate(stages[:-1]):
        progress_container.markdown(f'<div class="agent-trace">{label}...</div>', unsafe_allow_html=True)
        time.sleep(dur)
        elapsed += dur
        progress_bar.progress(min(elapsed / (total_fake_time + 0.1), 0.85))

    # Final stage: actual backend call
    final_label = stages[-1][0]
    progress_container.markdown(f'<div class="agent-trace">{final_label}...</div>', unsafe_allow_html=True)
    result = analyze(user_text, include_explanations=include_explanations)

    progress_bar.progress(1.0)
    progress_container.empty()
    progress_bar.empty()

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

    # ── TAB: Explainability (SHAP) ─────────────────────────────────────
    with tabs[2]:
        if opts["show_explanations"]:
            summary = result.get("explanation_summary")
            if summary:
                st.info(f"💬 {summary}")

            exps = result.get("explanations", [])
            if exps:
                st.caption(f"Influential tokens ({len(exps)}):")
                cols = st.columns(4)
                for i, tok in enumerate(exps[:16]):
                    word = tok.get("word", "")
                    weight = float(tok.get("weight", 0.0))
                    infl = tok.get("influence", "positive")
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
                st.caption("Enable 'Include explanations' when analyzing to see SHAP output.")
        else:
            st.caption("Enable 'SHAP token explanations' in sidebar.")

    # ── TAB: Agent Trace ──────────────────────────────────────────────
    tab_idx = 3
    if opts["show_agent_trace"]:
        with tabs[tab_idx]:
            st.markdown(f'<div class="agent-trace">✅ ClassificationAgent → {emotion} ({confidence:.1%})</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-trace">✅ CrisisAgent → risk_level={risk_level}, is_crisis={crisis}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-trace">✅ RAGAgent → {len(result.get("recommendations", []))} recommendation(s)</div>', unsafe_allow_html=True)
            if result.get("explanations"):
                st.markdown(f'<div class="agent-trace">✅ ExplainerAgent (SHAP) → {len(result["explanations"])} tokens</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-trace">✅ Orchestrator complete → {ms}ms</div>', unsafe_allow_html=True)
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
    # Health check up front (also drives header pill + sidebar)
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
            )
            render_results(result, opts, user_text.strip())
        except RuntimeError as e:
            st.error(f"❌ Analysis failed: {e}")
    else:
        render_welcome_placeholder()


if __name__ == "__main__":
    main()