# -*- coding: utf-8 -*-
"""
Real-Time Monitoring Dashboard for Mental Health AI Platform.

Pulls Prometheus metrics from http://localhost:8001/metrics and visualizes:
- Total Requests & Crisis Alerts
- Average Latency (ms)
- Emotion Classification Distribution
- Model Confidence Score Distribution
- Risk Level Breakdown
"""

import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Monitoring · Mental Health AI",
    page_icon="📊",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
METRICS_URL = os.getenv("METRICS_URL", "http://localhost:8001/metrics")
REFRESH_INTERVAL = 5  # seconds

# ══════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}

    .metric-tile {
        background: linear-gradient(135deg, #1e2130 0%, #2a2d3e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2d3e;
        margin-bottom: 12px;
    }

    .metric-tile-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-tile-value {
        color: white;
        font-size: 32px;
        font-weight: 900;
        line-height: 1;
    }

    .metric-tile-delta {
        color: #10b981;
        font-size: 12px;
        font-weight: 700;
        margin-top: 6px;
    }

    .section-header {
        color: white;
        font-size: 18px;
        font-weight: 800;
        margin: 24px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .live-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════
# PROMETHEUS PARSER
# ══════════════════════════════════════════════════════════════════════
def fetch_metrics() -> str:
    """Fetch raw Prometheus text from backend metrics endpoint."""
    try:
        resp = requests.get(METRICS_URL, timeout=3)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"# ERROR: {e}"


def parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    """Parse Prometheus text format into structured dict."""
    results = defaultdict(list)
    line_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([\-0-9.eE+]+)')
    label_re = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        name, labels_str, value = m.groups()
        labels = dict(label_re.findall(labels_str or ""))
        try:
            results[name].append((labels, float(value)))
        except ValueError:
            continue

    return dict(results)


def sum_counter(metrics: Dict, name: str) -> float:
    """Sum counter values across all label combinations."""
    return sum(v for _, v in metrics.get(name, []))


def group_counter_by_label(metrics: Dict, name: str, label: str) -> Dict[str, float]:
    """Group counter by a specific label value."""
    result = defaultdict(float)
    for labels, value in metrics.get(name, []):
        key = labels.get(label, "unknown")
        result[key] += value
    return dict(result)


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns([3, 1])
with col_a:
    st.markdown(
        """
        <h1 style="
            margin: 0;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">
            📊 Real-Time Monitoring
        </h1>
        <p style="color:#94a3b8; font-size:13px; margin-top:4px;">
            Live Prometheus metrics from the FastAPI backend (port 8001)
        </p>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""
        <div style="text-align:right; padding-top:8px;">
            <span class="live-badge">
                <span class="live-dot"></span>LIVE
            </span>
            <div style="color:#64748b; font-size:11px; margin-top:6px; font-family:monospace;">
                {METRICS_URL}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════
# FETCH & PARSE METRICS
# ══════════════════════════════════════════════════════════════════════
raw_text = fetch_metrics()

if raw_text.startswith("# ERROR"):
    st.error(f"❌ Cannot reach metrics endpoint: {METRICS_URL}")
    st.info("Ensure the backend is running: `python -m uvicorn backend.main:app --reload --port 8000`")
    st.stop()

parsed = parse_prometheus(raw_text)

# Exact metric names from backend/monitoring/metrics.py
TOTAL_REQUESTS = sum_counter(parsed, "mh_platform_requests_total")
CRISIS_ALERTS = sum_counter(parsed, "mh_platform_crisis_alerts_total")

# Calculate average latency using exact metric names
LATENCY_SUM = sum_counter(parsed, "mh_platform_request_duration_ms_sum")
LATENCY_COUNT = sum_counter(parsed, "mh_platform_request_duration_ms_count")
AVG_LATENCY = (LATENCY_SUM / LATENCY_COUNT) if LATENCY_COUNT > 0 else 0.0

# Calculate average confidence
CONF_SUM = sum_counter(parsed, "mh_platform_confidence_score_sum")
CONF_COUNT = sum_counter(parsed, "mh_platform_confidence_score_count")
AVG_CONF = (CONF_SUM / CONF_COUNT) if CONF_COUNT > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════
# TOP-LEVEL TILES
# ══════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Total Requests</div>
            <div class="metric-tile-value">{int(TOTAL_REQUESTS):,}</div>
            <div class="metric-tile-delta">Recorded via Prometheus</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Crisis Alerts</div>
            <div class="metric-tile-value">{int(CRISIS_ALERTS):,}</div>
            <div class="metric-tile-delta" style="color:{'#ef4444' if CRISIS_ALERTS > 0 else '#10b981'};">
                {'Safety triggered' if CRISIS_ALERTS > 0 else 'All clear'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Avg Latency</div>
            <div class="metric-tile-value">{AVG_LATENCY:.0f}ms</div>
            <div class="metric-tile-delta">End-to-End Pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Avg Confidence</div>
            <div class="metric-tile-value">{AVG_CONF:.1%}</div>
            <div class="metric-tile-delta">Model Certainty</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════
# CHARTS & BREAKDOWNS
# ══════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">🎭 Emotion Breakdown</div>', unsafe_allow_html=True)
    emotion_counts = group_counter_by_label(parsed, "mh_platform_requests_total", "emotion")
    if emotion_counts:
        df_emo = pd.DataFrame([
            {"Emotion": k.title(), "Count": int(v)}
            for k, v in emotion_counts.items()
        ]).sort_values("Count", ascending=False)
        st.bar_chart(df_emo.set_index("Emotion"), color="#6366f1", height=260)
    else:
        st.info("No classification data recorded yet. Send a request from the main app!")

with col_right:
    st.markdown('<div class="section-header">🛡️ Risk Level Breakdown</div>', unsafe_allow_html=True)
    risk_counts = group_counter_by_label(parsed, "mh_platform_requests_total", "risk_level")
    if risk_counts:
        df_risk = pd.DataFrame([
            {"Risk Level": k.title(), "Count": int(v)}
            for k, v in risk_counts.items()
        ])
        st.bar_chart(df_risk.set_index("Risk Level"), color="#ef4444", height=260)
    else:
        st.info("No risk level data recorded yet.")

# ══════════════════════════════════════════════════════════════════════
# RAW METRICS
# ══════════════════════════════════════════════════════════════════════
with st.expander("🔬 Raw Prometheus Endpoint Text"):
    st.code(raw_text, language="text")

# ══════════════════════════════════════════════════════════════════════
# REFRESH FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
rf_col1, rf_col2 = st.columns([4, 1])
with rf_col1:
    st.caption(f"Metrics collected live from `{METRICS_URL}` • Last refreshed: {time.strftime('%H:%M:%S')}")
with rf_col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()