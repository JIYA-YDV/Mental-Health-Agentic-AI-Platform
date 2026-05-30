import streamlit as st
import requests
from datetime import datetime

# ── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mental Health Agentic AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/classify"
HEALTH_URL = "http://127.0.0.1:8000/health"

# ── Emotion Color Mapping ──────────────────────────────────────────────────

EMOTION_COLORS = {
    "Joy / Happiness": "#2ecc71",
    "Neutral": "#95a5a6",
    "Sadness / Depression": "#3498db",
    "Fear / Anxiety": "#9b59b6",
    "Anger": "#e74c3c",
    "Disgust": "#e67e22",
    "Surprise": "#f1c40f"
}

EMOTION_ICONS = {
    "Joy / Happiness": "😊",
    "Neutral": "😐",
    "Sadness / Depression": "😢",
    "Fear / Anxiety": "😰",
    "Anger": "😠",
    "Disgust": "🤢",
    "Surprise": "😲"
}

RISK_COLORS = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e74c3c",
    "critical": "#8e44ad"
}


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar with settings and system status."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # System health check
        st.subheader("🔌 System Status")
        try:
            health = requests.get(HEALTH_URL, timeout=3).json()
            st.success(f"✅ API Online | v{health.get('version', 'N/A')}")
            st.info(
                f"🤖 Models: {'Loaded' if health.get('models_loaded') else 'Loading'}"
            )
        except Exception:
            st.error("❌ API Offline - Start the backend server")

        st.divider()

        # Options
        st.subheader("🛠️ Analysis Options")
        include_explanations = st.toggle(
            "Include Token Explanations",
            value=False,
            help="Show which words influenced the emotion detection"
        )

        show_all_predictions = st.toggle(
            "Show All Emotion Scores",
            value=True
        )

        st.divider()

        # Disclaimer
        st.subheader("⚠️ Important Notice")
        st.caption(
            "This platform is an AI research tool and does NOT provide "
            "medical advice. If you are in crisis, please contact: "
            "**988 Suicide & Crisis Lifeline** or text HOME to **741741**"
        )

    return include_explanations, show_all_predictions


# ── Main Content ───────────────────────────────────────────────────────────

def render_crisis_banner(crisis_assessment: dict):
    """Render crisis alert banner if risk level is high."""
    if crisis_assessment["is_crisis"]:
        risk = crisis_assessment["risk_level"]
        color = RISK_COLORS.get(risk, "#e74c3c")

        st.error(
            f"🚨 **Crisis Alert - {risk.upper()} Risk Detected**\n\n"
            "Please reach out for immediate support:"
        )
        for resource in crisis_assessment["immediate_resources"]:
            st.markdown(f"- {resource}")
    else:
        st.info(
            "💙 **Wellness Resources Available** - "
            "See recommendations below for personalized support"
        )


def render_emotion_card(emotion: str, confidence: float):
    """Render primary emotion detection card."""
    color = EMOTION_COLORS.get(emotion, "#7f8c8d")
    icon = EMOTION_ICONS.get(emotion, "🔍")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}44);
            border-left: 5px solid {color};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        ">
            <h2 style="margin:0; color: {color};">
                {icon} {emotion}
            </h2>
            <p style="margin:5px 0; font-size: 18px;">
                Confidence: <strong>{round(confidence * 100, 1)}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_recommendations(recommendations: list):
    """Render wellness recommendations from RAG pipeline."""
    if not recommendations:
        st.info("No specific recommendations retrieved for this input.")
        return

    st.subheader("💡 Personalized Wellness Recommendations")
    st.caption("Retrieved from mental health knowledge base via RAG pipeline")

    for i, rec in enumerate(recommendations):
        category_icons = {
            "breathing": "🫁",
            "mindfulness": "🧘",
            "coping": "🛡️",
            "crisis": "🆘",
            "wellness": "🌿"
        }
        icon = category_icons.get(rec["category"], "📋")

        with st.expander(
            f"{icon} {rec['title']} "
            f"(Relevance: {round(rec['relevance_score'] * 100, 1)}%)",
            expanded=(i == 0)
        ):
            st.write(rec["content"])
            st.caption(f"Category: {rec['category'].title()} | Source: {rec['source']}")


def render_predictions_chart(all_predictions: list, show_all: bool):
    """Render emotion score bar chart."""
    if not show_all:
        return

    st.subheader("📊 All Emotion Scores")

    for pred in all_predictions:
        label = pred["label"]
        score = pred["score"]
        color = EMOTION_COLORS.get(label, "#7f8c8d")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(score)
        with col2:
            st.write(f"**{round(score * 100, 1)}%**")

        st.caption(f"{EMOTION_ICONS.get(label, '🔍')} {label}")


def render_explanations(explanations: list):
    """Render token-level explanations."""
    if not explanations:
        return

    st.subheader("🔍 Token-Level Explanations")
    st.caption("Words that influenced the emotion detection")

    important_tokens = [
        e for e in explanations
        if e["importance"] > 0.5
    ]

    if important_tokens:
        st.write("**Key words detected:**")
        cols = st.columns(min(len(important_tokens), 4))
        for i, token in enumerate(important_tokens):
            with cols[i % 4]:
                st.metric(
                    label=token["token"],
                    value=f"{round(token['importance'] * 100)}%",
                    delta=token["sentiment_direction"]
                )
    else:
        st.info("No strongly influential individual tokens found.")


# ── Main App ───────────────────────────────────────────────────────────────

def main():
    # Sidebar
    include_explanations, show_all_predictions = render_sidebar()

    # Header
    st.title("🧠 Mental Health Agentic AI Platform")
    st.markdown(
        "*AI-powered emotional intelligence and wellness insights "
        "powered by multi-agent orchestration*"
    )
    st.divider()

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            "💬 Share your thoughts or feelings:",
            height=150,
            placeholder=(
                "I've been feeling really anxious about work lately "
                "and can't seem to calm down..."
            )
        )

    with col2:
        st.write("")
        st.write("")
        session_id = st.text_input(
            "Session ID (optional)",
            placeholder="session_001"
        )
        analyze_button = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True
        )

    # Analysis
    if analyze_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
            return

        with st.spinner("🤖 Running multi-agent analysis..."):
            try:
                payload = {
                    "text": user_input.strip(),
                    "include_explanations": include_explanations,
                    "session_id": session_id if session_id else None
                }

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Analysis Complete")
                    st.divider()

                    # Crisis banner (always first)
                    render_crisis_banner(result["crisis_assessment"])
                    st.divider()

                    # Emotion card + metadata
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        render_emotion_card(
                            result["emotion"],
                            result["confidence"]
                        )
                    with col_b:
                        st.metric(
                            "⚡ Processing Time",
                            f"{result['processing_time_ms']} ms"
                        )
                        risk = result["crisis_assessment"]["risk_level"]
                        st.metric(
                            "🛡️ Risk Level",
                            risk.upper(),
                            delta_color="inverse"
                        )
                    with col_c:
                        risk_score = result["crisis_assessment"]["risk_score"]
                        st.metric(
                            "📈 Risk Score",
                            f"{round(risk_score * 100, 1)}%"
                        )
                        st.metric(
                            "🎯 Confidence",
                            f"{round(result['confidence'] * 100, 1)}%"
                        )

                    st.divider()

                    # Predictions chart
                    render_predictions_chart(
                        result["all_predictions"],
                        show_all_predictions
                    )

                    # Wellness recommendations
                    render_recommendations(result["recommendations"])

                    # Explanations
                    if include_explanations and result.get("explanations"):
                        render_explanations(result["explanations"])

                    # Raw JSON for developers
                    with st.expander("🛠️ Raw API Response (Developer View)"):
                        st.json(result)

                elif response.status_code == 422:
                    st.error(f"❌ Validation Error: {response.json()['detail']}")
                elif response.status_code == 503:
                    st.error("❌ Service Unavailable - Models may still be loading")
                else:
                    st.error(f"❌ API Error {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to backend API. "
                    "Please run: `python -m backend.main`"
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The model may be loading.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()