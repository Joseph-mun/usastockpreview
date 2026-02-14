# -*- coding: utf-8 -*-
"""Streamlit dashboard for US Market Predictor.

Run locally: streamlit run dashboard.py --server.port 8501
Streamlit Cloud: Set S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in secrets.
"""

import io
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==================== Config ====================

DATA_DIR = Path(__file__).resolve().parent / "data"
SIGNAL_PATH = DATA_DIR / "prediction_signal.json"
HISTORY_PATH = DATA_DIR / "probability_history.csv"

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "ap-northeast-2")

TIER_COLORS = {
    "Aggressive": "#ef4444",
    "Growth": "#f97316",
    "Moderate": "#eab308",
    "Cautious": "#22c55e",
    "Defensive": "#3b82f6",
}

# ==================== S3 Helpers ====================


def _get_s3_client():
    """Create S3 client using env vars or Streamlit secrets."""
    import boto3
    return boto3.client("s3", region_name=S3_REGION)


def _load_s3_json(key: str) -> dict | None:
    try:
        s3 = _get_s3_client()
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return None


def _load_s3_csv(key: str) -> pd.DataFrame:
    try:
        s3 = _get_s3_client()
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(resp["Body"].read()), parse_dates=["date"])
    except Exception:
        return pd.DataFrame()


# ==================== Data Loading ====================


@st.cache_data(ttl=300)
def load_signal() -> dict | None:
    """Load latest prediction signal JSON (S3 or local)."""
    if S3_BUCKET:
        return _load_s3_json("prediction_signal.json")
    if not SIGNAL_PATH.exists():
        return None
    try:
        return json.loads(SIGNAL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    """Load probability history CSV (S3 or local)."""
    if S3_BUCKET:
        df = _load_s3_csv("probability_history.csv")
        if not df.empty:
            return df.sort_values("date").reset_index(drop=True)
        return df
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ==================== Page Config ====================

st.set_page_config(
    page_title="US Market Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 US Market Predictor Dashboard")

# ==================== Load Data ====================

signal = load_signal()
history = load_history()

if signal is None:
    st.warning(
        "예측 데이터가 없습니다. `python -m src.pipelines.daily_pipeline` 을 먼저 실행하세요."
    )
    st.stop()

# ==================== Row 1: Key Metrics ====================

prob = signal.get("probability", 0.5)
signal_text = signal.get("signal", "N/A")
tier = signal.get("tier", "N/A")
alloc = signal.get("allocation", {})
date_str = signal.get("date", "N/A")
prev_prob = signal.get("prev_probability")
indicators = signal.get("indicators", {})

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta = None
    if prev_prob is not None:
        delta = f"{(prob - prev_prob) * 100:+.1f}%p"
    st.metric("상승 확률", f"{prob * 100:.1f}%", delta=delta)

with col2:
    st.metric("시그널", signal_text)

with col3:
    st.metric("Tier", tier)

with col4:
    tqqq_pct = alloc.get("tqqq", 0) * 100
    spy_pct = alloc.get("splg", 0) * 100
    cash_pct = alloc.get("cash", 0) * 100
    st.metric("TQQQ / SPY / Cash", f"{tqqq_pct:.0f}% / {spy_pct:.0f}% / {cash_pct:.0f}%")

with col5:
    st.metric("기준일", date_str)

st.divider()

# ==================== Row 2: Gauge + Allocation Pie ====================

col_gauge, col_pie = st.columns(2)

with col_gauge:
    st.subheader("확률 게이지")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={
                "reference": (prev_prob or prob) * 100,
                "suffix": "%p",
                "increasing": {"color": "#22c55e"},
                "decreasing": {"color": "#ef4444"},
            },
            number={"suffix": "%", "font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 45], "color": "#dbeafe"},
                    {"range": [45, 50], "color": "#fef9c3"},
                    {"range": [50, 55], "color": "#fef3c7"},
                    {"range": [55, 60], "color": "#fed7aa"},
                    {"range": [60, 100], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "NASDAQ 20일 상승 확률"},
        )
    )
    fig_gauge.update_layout(height=300, margin=dict(t=40, b=20, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_pie:
    st.subheader("포트폴리오 배분")
    labels = ["TQQQ", "SPY", "Cash"]
    values = [alloc.get("tqqq", 0), alloc.get("splg", 0), alloc.get("cash", 0)]
    colors = ["#ef4444", "#3b82f6", "#94a3b8"]

    fig_pie = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=14),
        )
    )
    tier_color = TIER_COLORS.get(tier, "#6b7280")
    fig_pie.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=30, r=30),
        annotations=[
            dict(text=tier, x=0.5, y=0.5, font_size=16, font_color=tier_color, showarrow=False)
        ],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ==================== Row 3: Probability Trend Chart ====================

st.subheader("확률 트렌드 (최근 60일)")

if not history.empty and len(history) >= 2:
    trend_days = st.slider("표시 기간 (일)", min_value=14, max_value=min(len(history), 500), value=60)
    trend_df = history.tail(trend_days)

    fig_trend = go.Figure()

    # Probability line
    fig_trend.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["probability"] * 100,
            mode="lines+markers",
            name="상승 확률",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=4),
        )
    )

    # 50% reference line
    fig_trend.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")

    # Tier thresholds
    fig_trend.add_hline(y=60, line_dash="dot", line_color="#ef4444", annotation_text="Aggressive (60%)")
    fig_trend.add_hline(y=45, line_dash="dot", line_color="#3b82f6", annotation_text="Defensive (45%)")

    fig_trend.update_layout(
        height=350,
        xaxis_title="날짜",
        yaxis_title="확률 (%)",
        yaxis=dict(range=[20, 80]),
        margin=dict(t=20, b=40, l=50, r=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("확률 히스토리 데이터가 부족합니다. 일일 파이프라인이 2일 이상 실행되면 트렌드가 표시됩니다.")

st.divider()

# ==================== Row 4: Recent History Table ====================

st.subheader("최근 5일 히스토리")

if not history.empty:
    recent = history.tail(5).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d") if hasattr(recent["date"].dtype, "tz") or pd.api.types.is_datetime64_any_dtype(recent["date"]) else recent["date"]
    recent["probability"] = (recent["probability"] * 100).round(1).astype(str) + "%"

    display_cols = {
        "date": "날짜",
        "probability": "확률",
        "signal": "시그널",
        "tier": "Tier",
        "tqqq_weight": "TQQQ",
        "spy_weight": "SPY",
        "cash_weight": "Cash",
    }
    available = [c for c in display_cols if c in recent.columns]
    recent_display = recent[available].rename(columns=display_cols)
    st.dataframe(recent_display, use_container_width=True, hide_index=True)
else:
    st.info("히스토리 데이터가 없습니다.")

st.divider()

# ==================== Row 5: Indicator Cards ====================

st.subheader("시장 지표")

ind_col1, ind_col2, ind_col3, ind_col4, ind_col5, ind_col6 = st.columns(6)

vix = indicators.get("vix", 0)
rsi = indicators.get("rsi", 0)
adx = indicators.get("adx", 0)
sma50 = indicators.get("sma50_ratio", 0)

with ind_col1:
    vix_color = "🟢" if vix < 15 else "🟡" if vix < 25 else "🔴"
    st.metric("VIX", f"{vix:.1f}", help="변동성 지수 (낮을수록 안정)")
    st.caption(f"{vix_color} {'Low' if vix < 15 else 'Mid' if vix < 25 else 'High'} Vol")

with ind_col2:
    rsi_label = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
    st.metric("RSI", f"{rsi:.1f}", help="상대강도지수 (30 이하 과매도, 70 이상 과매수)")
    st.caption(rsi_label)

with ind_col3:
    adx_label = "추세" if adx > 25 else "횡보" if adx < 20 else "약추세"
    st.metric("ADX", f"{adx:.1f}", help="추세강도 (25 이상 = 강한 추세)")
    st.caption(adx_label)

with ind_col4:
    sma_pct = sma50 * 100
    st.metric("SMA50 비율", f"{sma_pct:.1f}%", help="S&P500 중 50일선 위 종목 비율")

with ind_col5:
    vix_filter = signal.get("vix_filter", "N/A")
    st.metric("VIX 필터", vix_filter or "N/A", help="VIX 기반 TQQQ 노출 조절")

with ind_col6:
    regime = signal.get("regime", "N/A")
    st.metric("시장 체제", regime or "N/A", help="ADX 기반 추세/횡보 판단")

st.divider()

# ==================== Footer ====================

meta_info = []
if signal.get("ensemble_active"):
    meta_info.append("앙상블 모델 사용 중")
if signal.get("meta_learner_adjusted"):
    meta_info.append("메타러너 보정 적용")
timestamp = signal.get("timestamp", "")

st.caption(
    f"마지막 업데이트: {timestamp} | "
    + " | ".join(meta_info) if meta_info else f"마지막 업데이트: {timestamp}"
)
