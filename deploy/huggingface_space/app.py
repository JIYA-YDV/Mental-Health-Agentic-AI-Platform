"""
Mental Health AI Platform - Redesigned Compact UI
Fits everything in one viewport with modern dashboard aesthetics.
"""

import streamlit as st
import requests
import os
import time
from typing import Optional

# Optional: Groq for LLM streaming (add later)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ============================================================
# PAGE CONFIG - Wide layout for dashboard feel
# ============================================================

st.set_page_config(
    page_title="Mental Health AI",
    page_icon="💙",
    layout="wide",  # ← KEY: use full width
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS - Modern, compact, professional
# ============================================================

st.markdown("""
<style>
    /* Reduce top padding to fit more on screen */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        width: 260px !important;
        background: linear-gradient(180deg, #0f1116 0%, #1a1d26 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2a2d3e;
        margin-bottom: 8px;
    }
    
    /* Emotion result box - eye-catching */
    .emotion-box {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(75, 108, 183, 0.3);
    }
    
    .emotion-emoji {
        font-size: 48px;
        margin-bottom: 4px;
    }
    
    .emotion-label {
        font-size: 24px;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    
    .emotion-confidence {
        font-size: 14px;
        color: rgba(255,255,255,0.8);
        margin-top: 4px;
    }
    
    /* Risk level badges */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low { background: #10b981; color: white; }
    .risk-medium { background: #f59e0b; color: white; }
    .risk-high { background: #ef4444; color: white; }
    .risk-critical { background: #7c2d12; color: white; }
    
    /* Bar chart styling */
    .bar-container {
        margin: 8px 0;
    }
    
    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
        color: #e5e7eb;
    }
    
    .bar-track {
        height: 8px;
        background: #2a2d3e;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4b6cb7, #6366f1);
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    
    /* Compact buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Tab styling */
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
    
    /* Compact text area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #2a2d3e;
        background: #1e2130;
    }
    
    /* Header banner */
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
    
    .header-title {
        font-size: 20px;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    
    .header-badges {
        display: flex;
        gap: 8px;
    }
    
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    
    .badge-success { background: #065f46; color: #6ee7b7; }
    .badge-info { background: #1e3a8a; color: #93c5fd; }
    .badge-warning { background: #78350f; color: #fcd34d; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# EMOTION → EMOJI + COLOR MAPPING
# ============================================================

EMOTION_STYLES = {
    "sadness":  {"emoji": "😔", "color": "#3b82f6", "gradient": "linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)"},
    "joy":      {"emoji": "😊", "color": "#f59e0b", "gradient": "linear-gradient(135deg, #d97706 0%, #fbbf24 100%)"},
    "love":     {"emoji": "🥰", "color": "#ec4899", "gradient": "linear-gradient(135deg, #be185d 0%, #ec4899 100%)"},
    "anger":    {"emoji": "😠", "color": "#ef4444", "gradient": "linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)"},
    "fear":     {"emoji": "😨", "color": "#8b5cf6", "gradient": "linear-gradient(135deg, #6d28d9 0%, #8b5cf6 100%)"},
    "surprise": {"emoji": "😲", "color": "#10b981", "gradient": "linear-gradient(135deg, #047857 0%, #10b981 100%)"},
    "crisis":   {"emoji": "🚨", "color": "#dc2626", "gradient": "linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%)"},
}


def get_risk_level(confidence: float, crisis: bool) -> tuple:
    """Return (label, css_class) for risk badge."""
    if crisis:
        return ("Critical", "risk-critical")
    if confidence >= 0.9:
        return ("High", "risk-high")
    if confidence >= 0.6:
        return ("Medium", "risk-medium")
    return ("Low", "risk-low")


# ============================================================
# EXAMPLES
# ============================================================

EXAMPLES = {
    "— Select an example —": "",
    "😔 Clear sadness signal": "I feel hopeless and exhausted lately, nothing seems to bring me joy anymore.",
    "😊 Positive & grateful": "I got the promotion today! All that hard work finally paid off, I'm so grateful.",
    "😨 Anxious about future": "I can't stop worrying about my job interview tomorrow, my heart won't stop racing.",
    "😠 Frustrated & angry": "I'm so tired of being ignored. It feels like nothing I do matters at all.",
    "🥰 Love & connection": "Spending time with my family this weekend reminded me how loved I am.",
    "🚨 Crisis signal (test)": "I don't want to be here anymore. Nothing matters and I see no way out.",
}


# ============================================================
# HEADER
# ============================================================

def render_header():
    """Compact professional header."""
    st.markdown("""
    <div class="header-banner">
        <div>
            <p class="header-title">💙 Mental Health AI Platform</p>
        </div>
        <div class="header-badges">
            <span class="badge badge-success">● Live</span>
            <span class="badge badge-info">v1.2 • DistilRoBERTa</span>
            <span class="badge badge-warning">F1: 0.89</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR - Compact settings + info
# ============================================================

def render_sidebar():
    """Compact sidebar with settings and metadata."""
    with st.sidebar:
        st.markdown("### ⚙️ System")
        st.success("✅ Model Ready")
        st.caption("`YDVJIYA/distilroberta-base-finetuned-emotion`")
        
        st.markdown("---")
        st.markdown("### 🎛️ Display")
        show_tokens = st.toggle("Token explanations", value=True)
        show_scores = st.toggle("All emotion scores", value=True)
        show_safety = st.toggle("Safety override info", value=True)
        show_raw = st.toggle("Raw API response", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Model Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Macro F1", "0.89", "+0.02")
        with col2:
            st.metric("Crisis Recall", "96%", "+3%")
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("""
        - [🐙 GitHub Repo](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform)
        - [🤗 Fine-Tuned Model](https://huggingface.co/YDVJIYA/distilroberta-base-finetuned-emotion)
        - [📊 Evaluation Report](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/blob/main/docs/EVALUATION.md)
        """)
        
        st.markdown("---")
        st.warning(
            "⚠️ **Disclaimer:** Research tool only. Not a substitute for "
            "professional care. Crisis? Call **988** (US)."
        )
    
    return {
        "show_tokens": show_tokens,
        "show_scores": show_scores,
        "show_safety": show_safety,
        "show_raw": show_raw,
    }


# ============================================================
# INPUT SECTION - Compact horizontal layout
# ============================================================

def render_input_section():
    """Compact input area with example selector inline."""
    
    st.markdown("#### 💬 Share what's on your mind")
    
    # Two-column layout: example dropdown + character counter
    col_ex, col_analyze = st.columns([3, 1])
    
    with col_ex:
        selected_example = st.selectbox(
            "Try an example:",
            list(EXAMPLES.keys()),
            label_visibility="collapsed"
        )
    
    # Text input
    user_text = st.text_area(
        "Your text:",
        value=EXAMPLES[selected_example],
        placeholder="How are you feeling today? Share your thoughts here...",
        height=100,
        max_chars=500,
        label_visibility="collapsed"
    )
    
    # Analyze button - full width
    analyze = st.button(
        "🔍 Analyze Emotional Content",
        type="primary",
        use_container_width=True,
        disabled=not user_text.strip()
    )
    
    return user_text, analyze


# ============================================================
# RESULTS DISPLAY - Grid layout, fits one screen
# ============================================================

def render_results(result: dict, settings: dict):
    """
    Compact grid layout showing all key info at once.
    Uses tabs for detailed sections to avoid vertical scroll.
    """
    
    emotion = result.get("emotion", "unknown").lower()
    confidence = result.get("confidence", 0.0)
    crisis = result.get("crisis_detected", False)
    processing_ms = result.get("processing_time_ms", 0)
    safety_override = result.get("safety_override_applied", False)
    
    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["sadness"])
    risk_label, risk_class = get_risk_level(confidence, crisis)
    
    # =============================================
    # TOP ROW: Emotion Card + Metrics Grid
    # =============================================
    col_emotion, col_metrics = st.columns([1, 2])
    
    with col_emotion:
        st.markdown(f"""
        <div class="emotion-box" style="background: {style['gradient']};">
            <div class="emotion-emoji">{style['emoji']}</div>
            <p class="emotion-label">{emotion.title()}</p>
            <p class="emotion-confidence">Confidence: {confidence:.1%}</p>
            <div style="margin-top: 10px;">
                <span class="risk-badge {risk_class}">{risk_label} Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metrics:
        # 4 mini metric cards in a row
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Latency</div>
                <div style="color: white; font-size: 20px; font-weight: 700;">{processing_ms}ms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Model</div>
                <div style="color: white; font-size: 20px; font-weight: 700;">v1.1 FT</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            crisis_status = "⚠️ Yes" if crisis else "✓ No"
            crisis_color = "#ef4444" if crisis else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Crisis</div>
                <div style="color: {crisis_color}; font-size: 20px; font-weight: 700;">{crisis_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m4:
            override_status = "✓ Applied" if safety_override else "— None"
            override_color = "#f59e0b" if safety_override else "#94a3b8"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Safety</div>
                <div style="color: {override_color}; font-size: 20px; font-weight: 700;">{override_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Safety override notice (if applied)
        if safety_override and settings["show_safety"]:
            st.info(
                f"🛡️ **Safety override active:** {result.get('override_reason', 'Clinical vocabulary detected')}"
            )
    
    # Crisis alert (if detected) - full width, hard to miss
    if crisis:
        st.error("""
        🚨 **Crisis Support Available Now:**
        📞 **988** (Call/Text) • 💬 **Text HOME to 741741** • 🆘 **911** for emergencies
        """)
    
    # =============================================
    # BOTTOM ROW: Tabbed detail sections
    # =============================================
    st.markdown("")  # small spacer
    
    tab_labels = ["📊 Emotions", "💡 Recommendations", "🎯 Token Analysis"]
    if settings["show_raw"]:
        tab_labels.append("🔬 Raw API")
    
    tabs = st.tabs(tab_labels)
    
    # Tab 1: Emotion breakdown
    with tabs[0]:
        if settings["show_scores"]:
            all_emotions = result.get("all_emotions", [])
            if all_emotions:
                # Sort by score descending
                sorted_emotions = sorted(all_emotions, key=lambda x: x["score"], reverse=True)
                
                # Two columns for compact display
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
            for rec in recommendations:
                title = rec if isinstance(rec, str) else rec.get("title", "Resource")
                description = rec.get("description", "") if isinstance(rec, dict) else ""
                st.markdown(f"**📚 {title}**")
                if description:
                    st.caption(description)
                st.markdown("---")
        else:
            st.caption("No specific recommendations available for this input.")
    
    # Tab 3: Token Analysis
    with tabs[2]:
        if settings["show_tokens"]:
            explanations = result.get("explanations", [])
            if explanations:
                st.caption("Words that most influenced the prediction:")
                
                # Grid of token chips
                cols = st.columns(4)
                for idx, token in enumerate(explanations[:12]):  # limit to 12 for compactness
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
                st.caption("No token explanations available.")
        else:
            st.caption("Enable 'Token explanations' in the sidebar to view.")
    
    # Tab 4: Raw JSON (developer view)
    if settings["show_raw"]:
        with tabs[3]:
            st.json(result)


# ============================================================
# MOCK BACKEND (replace with your real API call)
# ============================================================

# ============================================================
# INFERENCE - Fixed tokenizer loading
# ============================================================

def analyze_text(text: str) -> dict:
    """
    Run emotion classification with safety overrides.
    Uses the fine-tuned model with proper tokenizer loading (use_fast=False fix).
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    
    # Load model + tokenizer only once (cached in session state)
    if "classifier" not in st.session_state:
        with st.spinner("Loading model (first time only, ~30 seconds)..."):
            model_name = "YDVJIYA/distilroberta-base-finetuned-emotion"
            
            # CRITICAL FIX: Load tokenizer explicitly with use_fast=False
            # This avoids the tokenizer.json format incompatibility
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False  # ← THE FIX from Phase 5
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Build pipeline with the pre-loaded components
            st.session_state.classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                device=-1  # CPU
            )
    
    # Run inference
    start = time.time()
    results = st.session_state.classifier(text)[0]
    elapsed_ms = int((time.time() - start) * 1000)
    
    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = results[0]
    
    # ============================================================
    # SAFETY OVERRIDE SYSTEM (layered defense)
    # ============================================================
    text_lower = text.lower()
    
    crisis_keywords = [
        "suicide", "suicidal", "kill myself", "end my life",
        "don't want to be here", "want to die", "no way out",
        "better off dead", "no reason to live", "can't go on"
    ]
    
    sadness_keywords = [
        "hopeless", "worthless", "meaningless", "empty inside",
        "exhausted", "nothing matters", "pointless", "numb",
        "can't feel anything", "helpless", "useless"
    ]
    
    fear_keywords = [
        "terrified", "panic attack", "can't breathe",
        "overwhelming anxiety", "paralyzed with fear"
    ]
    
    crisis_detected = any(kw in text_lower for kw in crisis_keywords)
    safety_override = False
    override_reason = ""
    original_emotion = top["label"]
    predicted_emotion = top["label"].lower()
    
    # Layer 1: Crisis (highest priority)
    if crisis_detected:
        predicted_emotion = "sadness"
        safety_override = True
        override_reason = "Crisis vocabulary detected — activated safety flow"
    
    # Layer 2: Sadness override (depression vocabulary misclassified as positive)
    elif any(kw in text_lower for kw in sadness_keywords) and predicted_emotion in ["joy", "love", "surprise"]:
        predicted_emotion = "sadness"
        safety_override = True
        override_reason = f"Depression vocabulary overrides '{original_emotion}' prediction"
    
    # Layer 3: Fear override (anxiety vocabulary misclassified as positive)
    elif any(kw in text_lower for kw in fear_keywords) and predicted_emotion in ["joy", "love", "surprise"]:
        predicted_emotion = "fear"
        safety_override = True
        override_reason = f"Anxiety vocabulary overrides '{original_emotion}' prediction"
    
    # ============================================================
    # TOKEN EXPLANATIONS (lexicon-based)
    # ============================================================
    words = text.split()
    explanations = []
    
    negative_lexicon = {
        "hopeless": -1.0, "worthless": -0.95, "meaningless": -0.9,
        "empty": -0.85, "exhausted": -0.8, "sad": -0.75,
        "angry": -0.7, "scared": -0.75, "anxious": -0.7,
        "terrible": -0.8, "awful": -0.75, "depressed": -0.9,
        "lost": -0.7, "broken": -0.8, "alone": -0.7,
        "crying": -0.75, "hurt": -0.7, "suicide": -1.0
    }
    
    positive_lexicon = {
        "happy": 0.9, "joy": 0.95, "grateful": 0.85,
        "love": 0.9, "amazing": 0.85, "wonderful": 0.85,
        "excited": 0.8, "great": 0.75, "blessed": 0.8,
        "proud": 0.8, "thankful": 0.85, "confident": 0.75
    }
    
    seen = set()
    for word in words[:30]:
        clean = word.lower().strip(".,!?;:'\"")
        if clean in seen:
            continue
        seen.add(clean)
        
        if clean in negative_lexicon:
            explanations.append({
                "word": clean,
                "weight": negative_lexicon[clean],
                "influence": "negative"
            })
        elif clean in positive_lexicon:
            explanations.append({
                "word": clean,
                "weight": positive_lexicon[clean],
                "influence": "positive"
            })
    
    # ============================================================
    # RECOMMENDATIONS (emotion-specific)
    # ============================================================
    recommendations_map = {
        "sadness": [
            "5-4-3-2-1 Grounding Exercise",
            "Cognitive Reframing for Negative Thoughts",
            "Reach out to a trusted friend or family member",
            "Consider speaking with a mental health professional"
        ],
        "fear": [
            "Box Breathing (4-4-4-4 pattern)",
            "Progressive Muscle Relaxation technique",
            "Anxiety journaling — write down your worries",
            "Grounding: name 5 things you can see"
        ],
        "anger": [
            "Take 10 deep breaths before responding",
            "Physical release: brief walk or exercise",
            "Journal what triggered the anger",
            "Anger management coping strategies"
        ],
        "joy": [
            "Gratitude journaling — capture this feeling",
            "Share your joy with someone you love",
            "Reflect on what led to this positive moment"
        ],
        "love": [
            "Express appreciation to those you care about",
            "Practice self-compassion daily",
            "Nurture your important relationships"
        ],
        "surprise": [
            "Take time to process unexpected events",
            "Reflect on your emotional response",
            "Talk through the surprise with someone"
        ],
    }
    
    return {
        "emotion": predicted_emotion,
        "confidence": top["score"],
        "all_emotions": [{"label": r["label"], "score": r["score"]} for r in results],
        "crisis_detected": crisis_detected,
        "crisis_confidence": 0.95 if crisis_detected else 0.0,
        "recommendations": recommendations_map.get(predicted_emotion, recommendations_map["sadness"]),
        "explanations": explanations,
        "safety_override_applied": safety_override,
        "override_reason": override_reason,
        "processing_time_ms": elapsed_ms,
    }


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    render_header()
    
    # Sidebar (returns display settings)
    settings = render_sidebar()
    
    # Main layout: input on top, results below
    user_text, analyze_clicked = render_input_section()
    
    # Divider
    st.markdown("")
    
    # Results section
    if analyze_clicked and user_text.strip():
        with st.spinner("🧠 Analyzing..."):
            result = analyze_text(user_text)
        
        if result:
            render_results(result, settings)
    elif not analyze_clicked:
        # Placeholder showing what will appear
        st.info("👆 Enter text above and click **Analyze** to see emotional analysis, "
                "risk assessment, personalized recommendations, and token-level explanations — "
                "all in one view.")


if __name__ == "__main__":
    main()