# -*- coding: utf-8 -*-
"""
HTTP client for the Mental Health Agentic AI Platform backend.

Wraps all calls to the FastAPI backend (localhost:8000) so the
Streamlit UI stays clean and testable. Handles:
- Health checks
- Analysis requests (POST /classify)
- Retries, timeouts, and error normalization
- Session ID management
"""

import os
import uuid
from typing import Any, Dict, Optional

import requests
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30
HEALTH_TIMEOUT_SECONDS = 3


# ═══════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════
def get_session_id() -> str:
    """
    Get or create a unique session ID for this Streamlit session.
    Persists across reruns within the same browser tab.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:12]}"
    return st.session_state.session_id


# ═══════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════
def check_health() -> Dict[str, Any]:
    """
    Ping the backend /health endpoint.
    Returns dict with status info, or error dict if unreachable.
    """
    try:
        resp = requests.get(
            f"{BACKEND_URL}/health",
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "reachable": True,
            "status": data.get("status", "unknown"),
            "version": data.get("version", "?"),
            "models_loaded": data.get("models_loaded", False),
            "error": None,
        }
    except requests.exceptions.ConnectionError:
        return {
            "reachable": False,
            "status": "offline",
            "version": None,
            "models_loaded": False,
            "error": "Backend server not running on " + BACKEND_URL,
        }
    except requests.exceptions.Timeout:
        return {
            "reachable": False,
            "status": "timeout",
            "version": None,
            "models_loaded": False,
            "error": "Backend did not respond within 3 seconds",
        }
    except Exception as e:
        return {
            "reachable": False,
            "status": "error",
            "version": None,
            "models_loaded": False,
            "error": str(e)[:200],
        }


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def analyze(
    text: str,
    include_explanations: bool = True,
    session_id: Optional[str] = None,
    explainer_method: str = "lexicon",  # ← NEW PARAM
) -> Dict[str, Any]:
    """
    Call POST /classify on the backend.

    Args:
        text: User input text (validated by backend, 1-5000 chars).
        include_explanations: Whether to run SHAP explainer.
        session_id: Optional session ID (auto-generated if omitted).

    Returns:
        Dict with backend response fields, PLUS extra derived fields
        the UI needs:
            - crisis_detected (bool)
            - safety_override_applied (bool)
            - override_reason (str)
            - all_emotions (list) — alias for all_predictions
            - explanations (list) — with `word` key aliased from `token`

    Raises:
        RuntimeError: If backend returns non-200 or is unreachable.
    """
    if session_id is None:
        session_id = get_session_id()

    payload = {
        "text": text.strip(),
        "include_explanations": include_explanations,
        "explainer_method": explainer_method,  # ← NEW
        "session_id": session_id,
    }

    try:
        resp = requests.post(
            f"{BACKEND_URL}/classify",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach backend at {BACKEND_URL}. "
            "Is uvicorn running? Start it with:\n"
            "    python -m uvicorn backend.main:app --reload --port 8000"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Backend timed out after {REQUEST_TIMEOUT_SECONDS}s. "
            "The model may be overloaded."
        )

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(
            f"Backend error {resp.status_code}: {str(detail)[:300]}"
        )

    data = resp.json()
    return _normalize_response(data)


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE NORMALIZATION
# Add UI-friendly derived fields and aliases without losing backend data
# ═══════════════════════════════════════════════════════════════════════
def _normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add derived fields the UI expects.
    Original backend fields remain untouched.
    """
    crisis_assessment = data.get("crisis_assessment", {}) or {}
    is_crisis = bool(crisis_assessment.get("is_crisis", False))
    indicators = crisis_assessment.get("crisis_indicators", []) or []

    # Derived: safety override info
    data["crisis_detected"] = is_crisis
    data["safety_override_applied"] = is_crisis or bool(indicators)
    data["override_reason"] = (
        f"Crisis indicators detected: {', '.join(indicators)}"
        if indicators
        else ""
    )

    # Alias: all_predictions → all_emotions
    data["all_emotions"] = data.get("all_predictions", [])

    # Alias: explanation tokens (token → word)
    raw_explanations = data.get("explanations") or []
    data["explanations"] = [
        {
            "word": tok.get("token", ""),
            "weight": float(tok.get("weight", 0.0)),
            "influence": tok.get("influence", "positive"),
        }
        for tok in raw_explanations
    ]

    # Ensure recommendations is a list
    data["recommendations"] = data.get("recommendations") or []

    # ═══ NEW: Fallback recommendations by emotion ═══════════════════════
    # If backend RAG returned nothing (low similarity), use curated defaults
    # so the UI always shows helpful, emotion-appropriate suggestions.
    if not data["recommendations"]:
        data["recommendations"] = _fallback_recommendations_for(
            data.get("emotion", "").lower()
        )
        data["_used_fallback_recommendations"] = True
    else:
        data["_used_fallback_recommendations"] = False

    return data


# ═══════════════════════════════════════════════════════════════════════
# FALLBACK RECOMMENDATIONS (used when RAG has no strong match)
# ═══════════════════════════════════════════════════════════════════════
_FALLBACK_RECS = {
    "joy": [
        {
            "title": "Savor the Moment",
            "content": "Positive experiences are worth pausing on. Take 60 seconds to fully absorb this feeling — notice sensations, thoughts, and gratitude. This 'savoring' practice has been shown to extend positive mood.",
            "category": "positive psychology",
            "relevance_score": 0.85,
            "source": "Curated Guidance",
        },
        {
            "title": "Share the Good News",
            "content": "Research shows that sharing positive experiences with others (called 'capitalization') amplifies happiness for both people. Text someone who'd celebrate with you.",
            "category": "connection",
            "relevance_score": 0.80,
            "source": "Curated Guidance",
        },
        {
            "title": "Gratitude Journaling",
            "content": "Write down 3 specific things that made this moment possible — your effort, others' support, timing, etc. This trains the brain to notice positive patterns.",
            "category": "mindfulness",
            "relevance_score": 0.78,
            "source": "Curated Guidance",
        },
    ],
    "love": [
        {
            "title": "Express Appreciation",
            "content": "Tell the person(s) how you feel — specifically what they did and how it affected you. This deepens connection and reinforces the bond you're feeling.",
            "category": "connection",
            "relevance_score": 0.85,
            "source": "Curated Guidance",
        },
        {
            "title": "Practice Loving-Kindness Meditation",
            "content": "Silently offer well-wishes: 'May you be happy, may you be safe, may you be at peace.' Start with yourself, then extend to loved ones. 5 minutes daily builds emotional warmth.",
            "category": "mindfulness",
            "relevance_score": 0.80,
            "source": "Curated Guidance",
        },
        {
            "title": "Nurture Your Relationships",
            "content": "Small acts of care compound over time. A brief message, a shared meal, or listening without fixing — these are the foundations of lasting connection.",
            "category": "relationships",
            "relevance_score": 0.75,
            "source": "Curated Guidance",
        },
    ],
    "surprise": [
        {
            "title": "Process Before Reacting",
            "content": "Unexpected events can activate the nervous system. Take three slow breaths before deciding how to respond. This gives your prefrontal cortex time to catch up.",
            "category": "regulation",
            "relevance_score": 0.80,
            "source": "Curated Guidance",
        },
        {
            "title": "Reflect on Your Emotional Response",
            "content": "Surprise often masks another underlying emotion — joy, fear, relief, or shock. Ask: 'Beneath the surprise, what am I feeling?'",
            "category": "self-awareness",
            "relevance_score": 0.75,
            "source": "Curated Guidance",
        },
    ],
    "anger": [
        {
            "title": "Physiological Reset",
            "content": "Anger activates the sympathetic nervous system. Cold water on your wrists, a brief walk, or 10 deep breaths (exhale longer than inhale) rapidly calms the body.",
            "category": "regulation",
            "relevance_score": 0.85,
            "source": "Curated Guidance",
        },
        {
            "title": "Name the Underlying Need",
            "content": "Anger often signals an unmet need — respect, fairness, safety, autonomy. Ask: 'What do I actually need right now?' Naming it makes it addressable.",
            "category": "self-awareness",
            "relevance_score": 0.80,
            "source": "Curated Guidance",
        },
        {
            "title": "Journal the Trigger",
            "content": "Write freely for 5 minutes about what happened. Don't edit or judge. This externalizes the emotion and often reveals patterns you can address.",
            "category": "reflection",
            "relevance_score": 0.75,
            "source": "Curated Guidance",
        },
    ],
    "fear": [
        {
            "title": "Box Breathing (4-4-4-4)",
            "content": "Inhale 4 seconds, hold 4, exhale 4, hold 4. Repeat 4 times. This regulates the autonomic nervous system and is used by Navy SEALs before high-stress events.",
            "category": "anxiety",
            "relevance_score": 0.90,
            "source": "Curated Guidance",
        },
        {
            "title": "5-4-3-2-1 Grounding",
            "content": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. This anchors you in the present when anxiety is future-focused.",
            "category": "grounding",
            "relevance_score": 0.85,
            "source": "Curated Guidance",
        },
        {
            "title": "Worry Journaling",
            "content": "Write down every worry — no matter how small. Then for each, mark: (a) can I influence this? (b) what's one small action? This shifts anxiety into agency.",
            "category": "cognitive",
            "relevance_score": 0.78,
            "source": "Curated Guidance",
        },
    ],
    "sadness": [
        {
            "title": "Self-Compassion Break",
            "content": "Place a hand on your heart and say: 'This is a moment of suffering. Suffering is part of being human. May I be kind to myself.' Research shows this reduces the sting of difficult emotions.",
            "category": "self-compassion",
            "relevance_score": 0.85,
            "source": "Curated Guidance",
        },
        {
            "title": "Gentle Movement",
            "content": "A 10-minute walk outdoors — even in dim light — meaningfully improves mood in most people. You don't need to feel motivated; the action itself shifts the state.",
            "category": "behavioral activation",
            "relevance_score": 0.80,
            "source": "Curated Guidance",
        },
        {
            "title": "Reach Out",
            "content": "Sadness often isolates. Send one message to someone you trust — even just 'thinking of you today.' Connection is a proven mood regulator.",
            "category": "connection",
            "relevance_score": 0.78,
            "source": "Curated Guidance",
        },
    ],
    "disgust": [
        {
            "title": "Pause and Distance",
            "content": "Physical or mental distance from the source of disgust can help. Step away, close the tab, or shift environments. Then process what happened.",
            "category": "regulation",
            "relevance_score": 0.75,
            "source": "Curated Guidance",
        },
    ],
    "neutral": [
        {
            "title": "Body Scan Check-In",
            "content": "Even in neutral states, brief self-awareness helps. Scan your body from head to toe — notice any tension, hunger, fatigue. Small adjustments compound.",
            "category": "mindfulness",
            "relevance_score": 0.70,
            "source": "Curated Guidance",
        },
    ],
}


def _fallback_recommendations_for(emotion: str) -> list:
    """
    Return curated recommendations when RAG has no strong semantic match.
    Ensures the UI always shows emotion-appropriate guidance.
    """
    return _FALLBACK_RECS.get(emotion, _FALLBACK_RECS.get("neutral", []))