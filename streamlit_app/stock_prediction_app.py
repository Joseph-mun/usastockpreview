# -*- coding: utf-8 -*-
"""
주가 예측 확률 확인 스트림릿 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import sys
import os
import importlib
import io
import json
import zipfile
import pickle
import base64
import html as html_lib
import urllib.request as urllib_request

# 일부 환경에서 잘못된 프록시(예: 127.0.0.1:9)가 설정되어
# Yahoo/FinanceDataReader/requests 호출이 실패하는 경우가 있어, 해당 케이스만 자동 해제합니다.
def _disable_bad_local_proxy_env():
    bad_markers = ("127.0.0.1:9", "localhost:9")
    keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    removed = False
    for k in keys:
        v = os.environ.get(k)
        if v and any(m in str(v) for m in bad_markers):
            os.environ.pop(k, None)
            removed = True
    if removed:
        # requests/urllib 계열이 프록시를 우회하도록 설정
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

_disable_bad_local_proxy_env()

def _force_requests_no_proxy_if_bad_local_proxy():
    """
    requests는 환경변수뿐 아니라 Windows 시스템 프록시도 읽을 수 있어,
    env를 지워도 127.0.0.1:9 같은 '죽은 프록시'를 계속 탈 수 있습니다.
    이 경우에만 requests 호출을 세션(trust_env=False, proxies={})로 강제합니다.
    """
    bad_markers = ("127.0.0.1:9", "localhost:9")

    def _has_bad_proxy() -> bool:
        # env
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            v = os.environ.get(k)
            if v and any(m in str(v) for m in bad_markers):
                return True
        # system proxy (urllib)
        try:
            px = urllib_request.getproxies() or {}
            for v in px.values():
                if v and any(m in str(v) for m in bad_markers):
                    return True
        except Exception:
            pass
        return False

    if not _has_bad_proxy():
        return

    try:
        import requests
    except Exception:
        return

    _orig_request = requests.request

    def _request_no_proxy(method, url, **kwargs):
        # 외부에서 proxies를 명시한 경우는 존중
        if "proxies" not in kwargs:
            kwargs["proxies"] = {}
        timeout = kwargs.get("timeout", None)
        if timeout is None:
            kwargs["timeout"] = 20
        with requests.Session() as s:
            s.trust_env = False
            return s.request(method=method, url=url, **kwargs)

    # requests.get/post/... 는 requests.request를 사용하므로 이것만 갈아끼우면 대부분 커버됩니다.
    requests.request = _request_no_proxy
    requests.api.request = _request_no_proxy
    # 보수적으로 get도 직접 바꿔둠(FinanceDataReader가 requests.get 직접 호출)
    requests.get = lambda url, **kwargs: _request_no_proxy("GET", url, **kwargs)
    requests.post = lambda url, **kwargs: _request_no_proxy("POST", url, **kwargs)

_force_requests_no_proxy_if_bad_local_proxy()

# 현재 파일의 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import stock_analysis_refactored as sar
    sar = importlib.reload(sar)  # 스트림릿 재실행 시 최신 코드 반영

    StockDataCollector = sar.StockDataCollector
    StockPredictor = sar.StockPredictor
    get_sp500_tickers = sar.get_sp500_tickers
    get_bond_data = sar.get_bond_data
    get_vix_data = sar.get_vix_data
    calculate_rsi = sar.calculate_rsi
    calculate_macd = sar.calculate_macd
    calculate_moving_averages = sar.calculate_moving_averages
    prepare_prediction_data = sar.prepare_prediction_data
    build_feature_matrix = sar.build_feature_matrix
except ImportError as e:
    import streamlit as st
    st.error(f"모듈 import 오류: {str(e)}\n\nstock_analysis_refactored.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="주가 예측 확률 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Shinhan Theme (UI/Charts)
# =========================

SHINHAN_BLUE = "#0046ff"
SHINHAN_NAVY = "#00236e"
SHINHAN_SKY = "#4baff5"
SHINHAN_LIGHT = "#8cd2f5"
TEXT_DARK = "#0b1220"
BG_SOFT = "#f5f8ff"

def build_screen_analysis_report_html_from_session_state() -> str | None:
    """
    화면(분석 결과) HTML 리포트 생성.

    - Plotly figure는 가능한 경우 PNG로 변환해 <img>로 저장(=진짜 '이미지' 형태)
    - PNG 변환이 불가하면(예: kaleido 미설치) Plotly interactive HTML로 fallback
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    up_prob = st.session_state.get("report_current_up_prob")
    down_prob = st.session_state.get("report_current_down_prob")
    up_recent5 = st.session_state.get("report_recent5_up")
    down_recent5 = st.session_state.get("report_recent5_down")

    sma_asof = st.session_state.get("sma50_asof")
    sma_above_df = st.session_state.get("sma50_above_df")
    sma_below_df = st.session_state.get("sma50_below_df")
    sma_view = st.session_state.get("report_sma50_view")  # 최신 사용자가 지정한 정렬/표시 설정

    ret_stats = st.session_state.get("report_return_stats")
    ret_stats_fig_json = st.session_state.get("report_return_stats_fig_json")
    sma_diffpct = st.session_state.get("report_sma50_diffpct")
    sma_diffpct_fig_json = st.session_state.get("report_sma50_diffpct_fig_json")

    has_sma_tables = (
        isinstance(sma_above_df, pd.DataFrame)
        and isinstance(sma_below_df, pd.DataFrame)
        and (not sma_above_df.empty or not sma_below_df.empty)
    )
    has_recent5 = bool(up_recent5) or bool(down_recent5)
    has_return_stats = bool(ret_stats) and isinstance(ret_stats, dict) and bool(ret_stats.get("rows"))
    has_sma_diffpct = bool(sma_diffpct) and isinstance(sma_diffpct, dict) and bool(sma_diffpct.get("rows"))

    fig_items: list[tuple[str, str]] = []
    # (title, session_key_of_json)
    for title, key in [
        ("상승 확률 추이", "report_fig_prob_trend_json"),
        ("최근 5년 지수 비교 vs 상승 확률", "report_fig_index_compare_json"),
        ("하락 확률 추이", "report_fig_down_trend_json"),
        ("최근 5년 지수 비교 vs 하락 확률", "report_fig_down_index_compare_json"),
    ]:
        if st.session_state.get(key):
            fig_items.append((title, key))

    if up_prob is None and down_prob is None and not fig_items and not has_sma_tables and not has_recent5 and not has_return_stats and not has_sma_diffpct:
        return None

    def _fmt_pct(p):
        try:
            return f"{float(p) * 100:.2f}%"
        except Exception:
            return "-"

    def _df_to_html_table(df: pd.DataFrame) -> str:
        try:
            tmp = df.copy()

            # 보기 좋은 포맷(가능한 컬럼만)
            if "MarketCap" in tmp.columns:
                def _fmt_mcap(x):
                    try:
                        return f"{float(x):,.0f}"
                    except Exception:
                        return ""
                tmp["MarketCap"] = tmp["MarketCap"].apply(lambda x: _fmt_mcap(x) if pd.notna(x) else "")

            for col in ["Adj Close", "SMA50"]:
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce").round(2)

            if "diff_pct" in tmp.columns:
                tmp["diff_pct"] = pd.to_numeric(tmp["diff_pct"], errors="coerce").round(2)

            return tmp.to_html(index=False, escape=True, classes="tbl")
        except Exception:
            return ""

    def _df_to_html_table_highlight_row(df: pd.DataFrame, highlight_col: str, highlight_value: str | None) -> str:
        """
        간단한 HTML 테이블 생성 + 특정 행 하이라이트.
        (pandas Styler를 HTML로 넣는 건 환경/버전 의존이 커서 직접 렌더링)
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return ""
        cols = list(df.columns)
        hv = str(highlight_value) if highlight_value is not None else None

        def esc(x):
            return html_lib.escape("" if x is None else str(x))

        out = []
        out.append("<table class='tbl' style='width:100%;'>")
        out.append("<thead><tr>" + "".join([f"<th>{esc(c)}</th>" for c in cols]) + "</tr></thead>")
        out.append("<tbody>")
        for _, row in df.iterrows():
            is_hl = False
            try:
                if hv is not None and highlight_col in df.columns:
                    is_hl = (str(row.get(highlight_col)) == hv)
            except Exception:
                is_hl = False
            style = " style='background:#fff3bf;'" if is_hl else ""
            out.append("<tr" + style + ">" + "".join([f"<td>{esc(row.get(c))}</td>" for c in cols]) + "</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    def _apply_sma_view(df: pd.DataFrame) -> pd.DataFrame:
        """
        HTML 저장 시에도 사용자가 마지막으로 선택한 정렬/표시개수 설정을 적용합니다.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        if not isinstance(sma_view, dict):
            return df

        sort_by = sma_view.get("sort_by")
        sort_dir = sma_view.get("sort_dir")  # "내림차순"/"오름차순"
        top_n = sma_view.get("top_n")

        tmp = df.copy()
        try:
            if sort_by and sort_by in tmp.columns:
                if sort_by in {"diff_pct", "MarketCap", "Adj Close", "SMA50"}:
                    tmp[sort_by] = pd.to_numeric(tmp[sort_by], errors="coerce")
                ascending = (sort_dir == "오름차순")
                tmp = tmp.sort_values(sort_by, ascending=ascending, na_position="last")
        except Exception:
            pass

        try:
            n = int(top_n) if top_n is not None else None
            if n is not None and n > 0:
                n = min(n, len(tmp))
                tmp = tmp.head(n)
        except Exception:
            pass

        return tmp.reset_index(drop=True)

    def _try_plotly_png_base64(fig) -> str | None:
        try:
            img_bytes = pio.to_image(fig, format="png", scale=2)
            return base64.b64encode(img_bytes).decode("ascii")
        except Exception:
            return None

    plotlyjs_included = False
    body_parts: list[str] = []

    # 요약 영역(텍스트)
    body_parts.append(f"<h2 style='margin:0 0 8px 0;'>화면 분석 결과</h2>")
    body_parts.append(f"<div style='color:#555;margin:0 0 16px 0;'>생성 시각: {generated_at}</div>")

    # 확률 요약(있으면)
    if up_prob is not None or down_prob is not None:
        body_parts.append("<div style='display:flex;gap:12px;flex-wrap:wrap;margin:0 0 18px 0;'>")
        if up_prob is not None:
            body_parts.append(
                f"<div style='border:1px solid #e6e8ef;border-radius:12px;padding:12px 14px;min-width:220px;'>"
                f"<div style='font-size:13px;color:#555;'>현재 상승 확률</div>"
                f"<div style='font-size:22px;font-weight:800;color:#0b1220;margin-top:6px;'>{_fmt_pct(up_prob)}</div>"
                f"</div>"
            )
        if down_prob is not None:
            body_parts.append(
                f"<div style='border:1px solid #e6e8ef;border-radius:12px;padding:12px 14px;min-width:220px;'>"
                f"<div style='font-size:13px;color:#555;'>현재 하락 확률</div>"
                f"<div style='font-size:22px;font-weight:800;color:#0b1220;margin-top:6px;'>{_fmt_pct(down_prob)}</div>"
                f"</div>"
            )
        body_parts.append("</div>")

    # 최근 5일 상승/하락 확률 요약
    if has_recent5:
        body_parts.append("<h3 style='margin:18px 0 10px 0;'>📅 최근 5일 확률 요약</h3>")

        rows_by_date: dict[str, dict] = {}
        for rows, k in [(up_recent5, "up"), (down_recent5, "down")]:
            if not rows:
                continue
            if isinstance(rows, list):
                for r in rows:
                    try:
                        d = str(r.get("date", ""))
                        p = r.get("prob", None)
                        if not d:
                            continue
                        rows_by_date.setdefault(d, {})
                        rows_by_date[d][k] = p
                    except Exception:
                        continue

        dates = sorted(rows_by_date.keys())
        body_parts.append(
            "<div style='border:1px solid #e6e8ef;border-radius:14px;overflow:hidden;background:#fff;'>"
            "<table class='tbl' style='width:100%;'>"
            "<thead><tr><th>날짜</th><th>상승 확률</th><th>하락 확률</th></tr></thead><tbody>"
        )
        for d in dates:
            up_v = rows_by_date[d].get("up", None)
            dn_v = rows_by_date[d].get("down", None)
            body_parts.append(
                "<tr>"
                f"<td>{html_lib.escape(d)}</td>"
                f"<td>{_fmt_pct(up_v) if up_v is not None else '-'}</td>"
                f"<td>{_fmt_pct(dn_v) if dn_v is not None else '-'}</td>"
                "</tr>"
            )
        body_parts.append("</tbody></table></div>")

    # SMA50 위/아래 기업 목록
    if has_sma_tables:
        body_parts.append("<h3 style='margin:18px 0 10px 0;'>📌 SMA50 위/아래 기업 목록</h3>")
        if sma_asof is not None:
            try:
                asof_str = sma_asof.strftime("%Y-%m-%d") if hasattr(sma_asof, "strftime") else str(sma_asof)[:10]
            except Exception:
                asof_str = str(sma_asof)[:10]
            body_parts.append(f"<div style='color:#555;margin:0 0 10px 0;'>기준일: {html_lib.escape(asof_str)}</div>")

        # 사용자 최신 정렬/표시개수 설정 반영
        above_view = _apply_sma_view(sma_above_df if isinstance(sma_above_df, pd.DataFrame) else pd.DataFrame())
        below_view = _apply_sma_view(sma_below_df if isinstance(sma_below_df, pd.DataFrame) else pd.DataFrame())
        above_html = _df_to_html_table(above_view if isinstance(above_view, pd.DataFrame) else pd.DataFrame())
        below_html = _df_to_html_table(below_view if isinstance(below_view, pd.DataFrame) else pd.DataFrame())
        body_parts.append("<div class='grid2'>")
        body_parts.append(
            "<div style='border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
            "<div style='font-weight:800;margin:4px 0 10px 0;'>✅ SMA50 위(가격 &gt; SMA50)</div>"
            f"{above_html if above_html else '<div style=\"color:#777;\">데이터 없음</div>'}"
            "</div>"
        )
        body_parts.append(
            "<div style='border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
            "<div style='font-weight:800;margin:4px 0 10px 0;'>❌ SMA50 아래(가격 &lt; SMA50)</div>"
            f"{below_html if below_html else '<div style=\"color:#777;\">데이터 없음</div>'}"
            "</div>"
        )
        body_parts.append("</div>")

    # 예측확률별 향후 수익률 통계(표 + 그래프)
    if has_return_stats:
        try:
            kind = html_lib.escape(str(ret_stats.get("kind", "")))
            lookback = html_lib.escape(str(ret_stats.get("lookback_days", "")))
            bin_size = html_lib.escape(str(ret_stats.get("bin_size", "")))
            latest_bin = ret_stats.get("latest_bin")
            body_parts.append("<h3 style='margin:18px 0 10px 0;'>📊 예측확률별 향후 수익률 통계</h3>")
            body_parts.append(f"<div style='color:#555;margin:0 0 10px 0;'>확률 종류: <b>{kind}</b> · 기간: 최근 {lookback}일 · 구간 폭: {bin_size}</div>")

            df_rs = pd.DataFrame(ret_stats.get("rows", []))
            # 최근 예측 구간 하이라이트
            if "prob_bin" in df_rs.columns:
                df_rs["prob_bin"] = df_rs["prob_bin"].astype(str)
            rs_html = _df_to_html_table_highlight_row(df_rs, highlight_col="prob_bin", highlight_value=str(latest_bin) if latest_bin is not None else None)
            body_parts.append(
                "<div style='border:1px solid #e6e8ef;border-radius:14px;overflow:hidden;background:#fff;'>"
                f"{rs_html if rs_html else '<div style=\"padding:10px;color:#777;\">데이터 없음</div>'}"
                "</div>"
            )

            if ret_stats_fig_json:
                try:
                    fig = pio.from_json(ret_stats_fig_json)
                    b64 = _try_plotly_png_base64(fig)
                    if b64:
                        body_parts.append(
                            "<div style='margin-top:12px;border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
                            f"<img alt='수익률 통계 그래프' style='width:100%;height:auto;display:block;' src='data:image/png;base64,{b64}'/>"
                            "</div>"
                        )
                    else:
                        include_js = "cdn" if not plotlyjs_included else False
                        plotlyjs_included = True
                        div = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
                        body_parts.append(
                            "<div style='margin-top:12px;border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
                            f"{div}"
                            "</div>"
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # SMA50 diff_pct 추이(Top 상승) (표 + 그래프)
    if has_sma_diffpct:
        try:
            period_label = html_lib.escape(str(sma_diffpct.get("period_label", "")))
            body_parts.append("<h3 style='margin:18px 0 10px 0;'>📈 SMA50 diff_pct 추이 (Top 상승)</h3>")
            body_parts.append(f"<div style='color:#555;margin:0 0 10px 0;'>기간: <b>{period_label}</b> · Top 10</div>")

            df_dp = pd.DataFrame(sma_diffpct.get("rows", []))
            dp_html = _df_to_html_table(df_dp)
            body_parts.append(
                "<div style='border:1px solid #e6e8ef;border-radius:14px;overflow:hidden;background:#fff;'>"
                f"{dp_html if dp_html else '<div style=\"padding:10px;color:#777;\">데이터 없음</div>'}"
                "</div>"
            )

            if sma_diffpct_fig_json:
                try:
                    fig = pio.from_json(sma_diffpct_fig_json)
                    b64 = _try_plotly_png_base64(fig)
                    if b64:
                        body_parts.append(
                            "<div style='margin-top:12px;border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
                            f"<img alt='SMA50 diff_pct 추이' style='width:100%;height:auto;display:block;' src='data:image/png;base64,{b64}'/>"
                            "</div>"
                        )
                    else:
                        include_js = "cdn" if not plotlyjs_included else False
                        plotlyjs_included = True
                        div = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
                        body_parts.append(
                            "<div style='margin-top:12px;border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
                            f"{div}"
                            "</div>"
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # Figure 영역
    for title, key in fig_items:
        fig_json = st.session_state.get(key)
        if not fig_json:
            continue

        try:
            fig = pio.from_json(fig_json)
        except Exception:
            continue

        body_parts.append(f"<h3 style='margin:18px 0 10px 0;'>{title}</h3>")

        b64 = _try_plotly_png_base64(fig)
        if b64:
            body_parts.append(
                "<div style='border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
                f"<img alt='{title}' style='width:100%;height:auto;display:block;' src='data:image/png;base64,{b64}'/>"
                "</div>"
            )
        else:
            # kaleido 미설치 등으로 PNG 변환이 안 되면 interactive HTML로 저장
            include_js = "cdn" if not plotlyjs_included else False
            plotlyjs_included = True
            div = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
            body_parts.append(
                "<div style='border:1px solid #e6e8ef;border-radius:14px;padding:10px;background:#fff;'>"
                f"{div}"
                "</div>"
            )

    html = f"""
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>화면 분석 결과</title>
    <style>
      body {{ font-family: Pretendard, "Noto Sans KR", "Segoe UI", system-ui, -apple-system, sans-serif; background:#f7f9ff; color:#0b1220; }}
      .wrap {{ max-width: 1100px; margin: 24px auto; padding: 0 14px; }}
      .card {{ background:#fff; border:1px solid #e6e8ef; border-radius:16px; padding:18px; box-shadow: 0 10px 30px rgba(0,35,110,0.06); }}
      h2,h3 {{ letter-spacing: -0.02em; }}
      .grid2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
      @media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
      table.tbl {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
      table.tbl th, table.tbl td {{ border: 1px solid #e6e8ef; padding: 6px 8px; text-align: left; vertical-align: top; }}
      table.tbl th {{ background: #f3f6ff; font-weight: 800; }}
      table.tbl tr:nth-child(even) td {{ background: #fbfcff; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        {''.join(body_parts)}
      </div>
    </div>
  </body>
</html>
""".strip()

    return html

def inject_shinhan_css():
    st.markdown(
        f"""
        <style>
          :root {{
            --shinhan-blue: {SHINHAN_BLUE};
            --shinhan-navy: {SHINHAN_NAVY};
            --shinhan-sky: {SHINHAN_SKY};
            --shinhan-light: {SHINHAN_LIGHT};
            --bg-soft: {BG_SOFT};
            --text-dark: {TEXT_DARK};
          }}

          /* App background */
          .stApp {{
            background: radial-gradient(1200px 600px at 10% -10%, rgba(0,70,255,0.18), transparent 60%),
                        radial-gradient(900px 500px at 95% 0%, rgba(75,175,245,0.18), transparent 55%),
                        linear-gradient(180deg, var(--bg-soft), #ffffff 70%);
          }}

          /* Main block spacing */
          [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {{
            gap: 0.75rem;
          }}

          /* Typography */
          html, body, [class*="css"] {{
            font-family: "Pretendard", "Noto Sans KR", "Segoe UI", system-ui, -apple-system, sans-serif;
            color: var(--text-dark);
          }}
          h1, h2, h3 {{
            letter-spacing: -0.02em;
          }}

          /* Hero header */
          .shinhan-hero {{
            padding: 18px 18px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(0,70,255,0.95), rgba(40,120,245,0.92));
            box-shadow: 0 14px 35px rgba(0, 35, 110, 0.18);
            color: white;
            border: 1px solid rgba(255,255,255,0.12);
          }}
          .shinhan-hero .kicker {{
            font-size: 13px;
            opacity: 0.92;
            margin: 0 0 6px 0;
          }}
          .shinhan-hero .title {{
            font-size: 28px;
            font-weight: 800;
            margin: 0;
          }}
          .shinhan-hero .subtitle {{
            margin: 8px 0 0 0;
            font-size: 14px;
            opacity: 0.92;
          }}

          /* Sidebar */
          [data-testid="stSidebar"] > div {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0,35,110,0.10);
          }}
          [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: var(--shinhan-navy);
          }}

          /* Buttons */
          .stButton > button {{
            border-radius: 12px !important;
            border: 1px solid rgba(0,70,255,0.28) !important;
            background: linear-gradient(135deg, var(--shinhan-blue), #2878f5) !important;
            color: #fff !important;
            box-shadow: 0 10px 22px rgba(0,70,255,0.18) !important;
            transition: transform .08s ease, box-shadow .08s ease, filter .12s ease;
          }}
          .stButton > button:hover {{
            filter: brightness(1.03);
            box-shadow: 0 14px 28px rgba(0,70,255,0.24) !important;
            transform: translateY(-1px);
          }}
          .stButton > button:active {{
            transform: translateY(0px);
            box-shadow: 0 10px 20px rgba(0,70,255,0.18) !important;
          }}

          /* Metrics as cards */
          [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(0,35,110,0.10);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 10px 24px rgba(0, 35, 110, 0.06);
          }}

          /* Expanders */
          details {{
            border-radius: 14px;
            border: 1px solid rgba(0,35,110,0.10);
            background: rgba(255,255,255,0.75);
            box-shadow: 0 10px 24px rgba(0, 35, 110, 0.06);
          }}
          details > summary {{
            padding: 10px 12px;
            font-weight: 650;
            color: var(--shinhan-navy);
          }}

          /* Dataframes */
          [data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(0,35,110,0.10);
            box-shadow: 0 10px 24px rgba(0, 35, 110, 0.06);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def apply_plotly_shinhan_theme():
    pio.templates.default = "plotly_white"
    pio.templates["shinhan"] = go.layout.Template(
        layout=dict(
            font=dict(family="Pretendard, Noto Sans KR, Segoe UI, sans-serif", color=TEXT_DARK),
            colorway=[SHINHAN_BLUE, SHINHAN_SKY, "#2ECC71", "#F1C40F", "#FF4B4B", SHINHAN_NAVY],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.92)",
            xaxis=dict(gridcolor="rgba(0,35,110,0.08)", zerolinecolor="rgba(0,35,110,0.12)"),
            yaxis=dict(gridcolor="rgba(0,35,110,0.08)", zerolinecolor="rgba(0,35,110,0.12)"),
            legend=dict(bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,35,110,0.10)", borderwidth=1),
        )
    )
    pio.templates.default = "shinhan"

inject_shinhan_css()
apply_plotly_shinhan_theme()

# Hero title
st.markdown(
    """
    <div class="shinhan-hero">
      <div class="kicker">Shinhan-style Dashboard</div>
      <div class="title">주가 예측 확률 분석</div>
      <div class="subtitle">RandomForest 기반 상승 확률 · 최근 5년 지수 비교 · SMA50 위/아래 스캐너</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 유틸: 지수 데이터(최근 5년) 조회
# =========================

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_index_adj_close(symbol_candidates, start_date, end_date):
    """
    FinanceDataReader 심볼이 환경/버전에 따라 다를 수 있어 후보를 순차 시도합니다.
    반환: (used_symbol, series) where series index=Date, values=Adj Close
    """
    last_err = None
    for sym in symbol_candidates:
        try:
            df = fdr.DataReader(sym, start_date, end_date)
            if df is None or df.empty:
                continue
            col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
            if col is None:
                continue
            s = df[col].copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            s = s[~s.isna()]
            if len(s) == 0:
                continue
            return sym, s
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"지수 데이터를 불러오지 못했습니다. candidates={symbol_candidates}, last_error={last_err}")

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_sp500_names():
    """
    FinanceDataReader의 S&P500 listing은 기본적으로 시가총액 컬럼이 없습니다.
    (Symbol/Name/Sector/Industry만 제공됨)
    여기서는 이름만 매핑으로 제공하고, 시가총액은 Yahoo Finance(yfinance)로 보강합니다.
    """
    try:
        listing = fdr.StockListing('S&P500')
        if listing is None or listing.empty:
            return {}
        listing = listing.copy()
        listing['Symbol'] = listing['Symbol'].astype(str).str.replace('.', '-', regex=False)
        return dict(zip(listing['Symbol'], listing['Name']))
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_market_caps_yahoo(tickers: tuple[str, ...]):
    """
    Yahoo Finance에서 시가총액을 가져옵니다.
    입력은 캐시 키 안정성을 위해 tuple로 받습니다.
    """
    try:
        import yfinance as yf
    except Exception:
        return {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = [str(t).strip() for t in tickers if str(t).strip()]
    if not symbols:
        return {}

    def fetch_one(original: str):
        candidates = [original]
        # 일부 티커는 Yahoo에서 '.' 표기일 수 있어 추가 시도
        if "-" in original:
            candidates.append(original.replace("-", "."))

        for sym in candidates:
            try:
                tk = yf.Ticker(sym)
                # fast_info가 있으면 우선 사용
                fi = getattr(tk, "fast_info", None)
                if fi and isinstance(fi, dict):
                    mcap = fi.get("market_cap")
                    if mcap is not None:
                        return original, float(mcap)

                info = tk.info if hasattr(tk, "info") else {}
                if isinstance(info, dict):
                    mcap = info.get("marketCap")
                    if mcap is not None:
                        return original, float(mcap)
            except Exception:
                continue

        return original, None

    out: dict[str, float | None] = {}
    max_workers = min(10, max(4, len(symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(fetch_one, s) for s in symbols]
        for f in as_completed(futs):
            k, v = f.result()
            out[k] = v

    return out

def rebase_to_100(series: pd.Series) -> pd.Series:
    """첫 값 기준 100으로 리베이스(정규화)"""
    s = series.dropna()
    if len(s) == 0:
        return series
    return (series / float(s.iloc[0])) * 100.0

def build_sma50_tables_from_collector(collector: StockDataCollector):
    """
    collector.collect_sma_data() 이후 collector.dataframes['sma50stock_df'] 기반으로
    '최근(오늘 실행 시점) 기준' SMA50 위/아래 티커를 표로 반환합니다.
    """
    df = collector.dataframes.get('sma50stock_df', None)
    if df is None or df.empty:
        return None, None, None

    df = df.copy()
    if 'Date' not in df.columns:
        # 방어: 혹시 Date가 없는 경우 인덱스에서 생성
        df = df.reset_index().rename(columns={'index': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Code', 'Date'])

    # 티커별 최신(가장 최근 거래일) 상태만 사용
    latest = df.groupby('Code', as_index=False).tail(1).copy()
    asof = latest['Date'].max()

    # 컬럼 이름은 SMA50/SMA50_YN로 구성됨
    yn_col = 'SMA50_YN'
    sma_col = 'SMA50'
    price_col = 'Adj Close'

    if yn_col not in latest.columns or sma_col not in latest.columns or price_col not in latest.columns:
        return asof, None, None

    latest['diff_pct'] = (latest[price_col] / latest[sma_col] - 1.0) * 100.0

    # 이름 + 시가총액 붙이기
    names = fetch_sp500_names()
    latest['Name'] = latest['Code'].map(lambda x: names.get(str(x)))

    tickers = tuple(sorted(set(latest['Code'].astype(str).tolist())))
    mcaps = fetch_market_caps_yahoo(tickers)
    latest['MarketCap'] = latest['Code'].map(lambda x: mcaps.get(str(x)))

    # 요청사항: Date 컬럼 제거
    view_cols = ['Code', 'Name', 'MarketCap', price_col, sma_col, 'diff_pct']

    above = latest[latest[yn_col] == 1][view_cols].copy()
    below = latest[latest[yn_col] == 0][view_cols].copy()

    # 기본 정렬: 시가총액 내림차순 (시가총액이 없으면 diff_pct 정렬로 fallback)
    if above['MarketCap'].notna().any():
        above = above.sort_values('MarketCap', ascending=False, na_position='last')
    else:
        above = above.sort_values('diff_pct', ascending=False)

    if below['MarketCap'].notna().any():
        below = below.sort_values('MarketCap', ascending=False, na_position='last')
    else:
        below = below.sort_values('diff_pct', ascending=True)

    # ⚠️ 표시 개수(top N)는 렌더링 단계에서 선택할 수 있어야 하므로,
    # 여기서는 잘라내지 않고 전체를 반환합니다.
    above = above.reset_index(drop=True)
    below = below.reset_index(drop=True)
    return asof, above, below


def render_sma50_tables_with_sort(above_df: pd.DataFrame, below_df: pd.DataFrame, key_prefix: str = "sma50_sort"):
    """
    SMA50 위/아래 테이블을 사용자 선택으로 정렬해서 표시합니다.
    (diff_pct, MarketCap, Adj Close 등)
    """
    if above_df is None or below_df is None:
        st.info("SMA50 테이블 데이터가 없습니다.")
        return

    # 정렬 후보 컬럼(존재하는 것만)
    candidate_cols = []
    for c in ["diff_pct", "MarketCap", "Adj Close", "SMA50", "Code", "Name"]:
        if (isinstance(above_df, pd.DataFrame) and c in above_df.columns) or (isinstance(below_df, pd.DataFrame) and c in below_df.columns):
            candidate_cols.append(c)

    if not candidate_cols:
        candidate_cols = list(above_df.columns)[:1] if isinstance(above_df, pd.DataFrame) and len(above_df.columns) else ["(없음)"]

    c_sort1, c_sort2, c_sort3 = st.columns([1.2, 1.0, 0.9])
    with c_sort1:
        sort_by = st.selectbox("정렬 기준", candidate_cols, index=0, key=f"{key_prefix}_by")
    with c_sort2:
        sort_dir = st.selectbox("정렬 방향", ["내림차순", "오름차순"], index=0, key=f"{key_prefix}_dir")
    with c_sort3:
        top_n = st.selectbox("표시 개수", [15, 30, 50, 100, 200, 500], index=0, key=f"{key_prefix}_n")

    ascending = (sort_dir == "오름차순")

    def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        tmp = df.copy()
        if sort_by in tmp.columns:
            # 숫자형 정렬 보강
            if sort_by in {"diff_pct", "MarketCap", "Adj Close", "SMA50"}:
                tmp[sort_by] = pd.to_numeric(tmp[sort_by], errors="coerce")
            tmp = tmp.sort_values(sort_by, ascending=ascending, na_position="last")
        n = int(top_n)
        # 데이터가 500 미만인 경우도 안전하게 처리
        n = min(n, len(tmp))
        return tmp.head(n).reset_index(drop=True)

    above_s = _sort_df(above_df)
    below_s = _sort_df(below_df)

    # HTML 저장에서도 동일하게 반영되도록 "마지막 선택"을 세션에 저장
    st.session_state["report_sma50_view"] = {
        "key_prefix": key_prefix,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "top_n": int(top_n),
        "updated_at": datetime.now().isoformat(),
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ SMA50 위(가격 > SMA50)")
        st.dataframe(above_s, use_container_width=True)
    with c2:
        st.markdown("#### ❌ SMA50 아래(가격 < SMA50)")
        st.dataframe(below_s, use_container_width=True)


def render_sma50_diffpct_trend_from_sma_dataframes(sma_dataframes: dict, key_prefix: str = "sma50_diffpct"):
    """
    SMA50 데이터(sma50stock_df)로부터 diff_pct 추이를 계산해,
    기간별(1주/1달/3달/6달) diff_pct 상승폭이 가장 큰 Top 10 기업의 추이 그래프를 그립니다.
    """
    if not (isinstance(sma_dataframes, dict) and sma_dataframes):
        st.info("SMA 데이터가 없어 diff_pct 추이를 계산할 수 없습니다.")
        return

    df = sma_dataframes.get("sma50stock_df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("sma50stock_df가 없어 diff_pct 추이를 계산할 수 없습니다.")
        return

    with st.expander("📈 SMA50 diff_pct 추이(기간별 Top 상승 10개)", expanded=False):
        period_map = {
            "최근 1주": ("1w", 7),
            "최근 1달": ("1m", 30),
            "최근 3달": ("3m", 90),
            "최근 6달": ("6m", 180),
        }
        period_label = st.selectbox("기간 선택", list(period_map.keys()), index=1, key=f"{key_prefix}_period")
        _, days = period_map[period_label]

        tmp = df.copy()
        # Date 정규화
        if "Date" in tmp.columns:
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        else:
            tmp = tmp.reset_index().rename(columns={"index": "Date"})
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")

        tmp = tmp.dropna(subset=["Date"]).sort_values(["Code", "Date"])

        # diff_pct 계산
        if "Adj Close" not in tmp.columns or "SMA50" not in tmp.columns:
            st.info("diff_pct 계산에 필요한 컬럼(Adj Close/SMA50)이 없습니다.")
            return

        tmp["Adj Close"] = pd.to_numeric(tmp["Adj Close"], errors="coerce")
        tmp["SMA50"] = pd.to_numeric(tmp["SMA50"], errors="coerce")
        tmp = tmp.dropna(subset=["Adj Close", "SMA50"])
        if tmp.empty:
            st.info("유효한 가격/SMA50 데이터가 없습니다.")
            return

        tmp["diff_pct"] = (tmp["Adj Close"] / tmp["SMA50"] - 1.0) * 100.0

        end_dt = tmp["Date"].max()
        start_dt = end_dt - pd.Timedelta(days=int(days))
        tmp_p = tmp[tmp["Date"] >= start_dt].copy()
        if tmp_p.empty:
            st.info("해당 기간에 데이터가 없습니다.")
            return

        # 코드별 시작/종료 diff_pct
        g = tmp_p.groupby("Code", as_index=False)
        first = g.first()[["Code", "Date", "diff_pct"]].rename(columns={"diff_pct": "start_diff_pct", "Date": "start_date"})
        last = g.last()[["Code", "Date", "diff_pct"]].rename(columns={"diff_pct": "end_diff_pct", "Date": "end_date"})
        merged = first.merge(last, on="Code", how="inner")
        merged["delta_diff_pct"] = merged["end_diff_pct"] - merged["start_diff_pct"]

        # Top 10 (상승폭 큰 순)
        top = merged.sort_values("delta_diff_pct", ascending=False).head(10).copy()
        if top.empty:
            st.info("Top 10을 만들 수 없습니다.")
            return

        # 이름 매핑(가능하면)
        names = fetch_sp500_names()
        top["Name"] = top["Code"].map(lambda x: names.get(str(x)))
        top_view = top[["Code", "Name", "start_date", "end_date", "start_diff_pct", "end_diff_pct", "delta_diff_pct"]].copy()
        for c in ["start_diff_pct", "end_diff_pct", "delta_diff_pct"]:
            top_view[c] = pd.to_numeric(top_view[c], errors="coerce").round(2)

        st.dataframe(top_view, use_container_width=True)

        # 추이 그래프
        top_codes = top["Code"].astype(str).tolist()
        plot_df = tmp_p[tmp_p["Code"].astype(str).isin(top_codes)].copy()
        plot_df = plot_df.sort_values(["Date", "Code"])

        fig = go.Figure()
        for code in top_codes:
            d = plot_df[plot_df["Code"].astype(str) == str(code)]
            if d.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=d["Date"],
                    y=d["diff_pct"],
                    mode="lines",
                    name=str(code),
                )
            )
        fig.update_layout(
            height=500,
            hovermode="x unified",
            title=f"SMA50 diff_pct 추이 - {period_label} Top 10 상승",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_yaxes(title_text="diff_pct (%)")
        fig.update_xaxes(title_text="날짜")
        st.plotly_chart(fig, use_container_width=True)

        # HTML 저장(리포트)용 저장
        try:
            st.session_state["report_sma50_diffpct"] = {
                "period_label": period_label,
                "rows": top_view.to_dict("records"),
            }
            st.session_state["report_sma50_diffpct_fig_json"] = fig.to_json()
        except Exception:
            pass

def ensure_sma50_tables_in_session_from_cached_sma() -> bool:
    """
    기존 모델 사용 시에도 SMA zip/패키지로 로드된 데이터가 있으면
    SMA50 위/아래 테이블을 세션 상태에 생성해 둡니다.
    """
    sma_dataframes = st.session_state.get("sma_dataframes")
    if not (isinstance(sma_dataframes, dict) and sma_dataframes):
        return False

    sig = (
        st.session_state.get("_sma_upload_sig")
        or st.session_state.get("_bundle_upload_sig")
        or ("sma_cache", st.session_state.get("sma_collector_date"), len(sma_dataframes))
    )

    already_ok = (
        st.session_state.get("_sma50_tables_sig") == sig
        and "sma50_above_df" in st.session_state
        and "sma50_below_df" in st.session_state
        and st.session_state.get("sma50_above_df") is not None
        and st.session_state.get("sma50_below_df") is not None
    )
    if already_ok:
        return True

    try:
        collector = StockDataCollector()
        collector.dataframes = sma_dataframes.copy()
        asof, above_df, below_df = build_sma50_tables_from_collector(collector)
        if above_df is None or below_df is None:
            return False

        st.session_state["sma50_asof"] = asof
        st.session_state["sma50_above_df"] = above_df
        st.session_state["sma50_below_df"] = below_df
        st.session_state["_sma50_tables_sig"] = sig
        return True
    except Exception:
        return False


def export_sma_data_zip(sma_dataframes: dict, meta: dict | None = None) -> bytes:
    """
    SMA 데이터프레임(dict)을 zip(bytes)로 내보냅니다.
    - 파일 구성: meta.json, sma15.csv, sma30.csv, sma50.csv (존재하는 것만)
    - pickle을 쓰지 않아 보안/호환성 측면에서 안전합니다.
    """
    meta = meta or {}
    # 직렬화 가능한 형태로 변환
    meta_out = {
        "exported_at": datetime.now().isoformat(),
        **meta,
        "keys": sorted(list(sma_dataframes.keys())),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta_out, ensure_ascii=False, indent=2))

        for k, df in sma_dataframes.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            # 파일명 안정화
            fname = f"{k}.csv"
            tmp = df.copy()
            # Date 컬럼이 있으면 ISO 문자열로 저장(로딩 안정성)
            if "Date" in tmp.columns:
                tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            csv_bytes = tmp.to_csv(index=False).encode("utf-8-sig")
            zf.writestr(fname, csv_bytes)

    return buf.getvalue()


def import_sma_data_zip(file_bytes: bytes) -> tuple[dict, dict]:
    """export_sma_data_zip()로 만든 zip(bytes)을 읽어 SMA 데이터(dict)와 meta(dict)를 반환합니다."""
    sma = {}
    meta = {}
    buf = io.BytesIO(file_bytes)
    with zipfile.ZipFile(buf, mode="r") as zf:
        if "meta.json" in zf.namelist():
            meta = json.loads(zf.read("meta.json").decode("utf-8"))

        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            key = os.path.splitext(os.path.basename(name))[0]
            df = pd.read_csv(io.BytesIO(zf.read(name)))
            # Date 복원(가능한 경우)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            sma[key] = df

    return sma, meta

# =========================
# 모델(pkl) 업/다운로드
# =========================

def export_model_pkl_bytes(predictor: StockPredictor) -> bytes | None:
    """현재 로드/학습된 모델을 pkl(bytes)로 내보냅니다."""
    if predictor is None or predictor.model is None:
        return None
    payload = {
        "model": predictor.model,
        "feature_columns": predictor.feature_columns,
        "exported_at": datetime.now().isoformat(),
    }
    return pickle.dumps(payload)


def load_model_from_pkl_bytes(predictor: StockPredictor, file_bytes: bytes) -> bool:
    """
    업로드한 pkl(bytes)에서 모델을 로드합니다.
    주의: pickle은 신뢰할 수 있는 파일만 업로드해야 합니다.
    """
    data = pickle.loads(file_bytes)

    # 우리가 저장하는 포맷(dict) 우선 지원
    if isinstance(data, dict) and "model" in data:
        predictor.model = data.get("model")
        predictor.feature_columns = data.get("feature_columns")
        return predictor.model is not None

    # 예외: 모델 객체만 바로 들어있는 경우
    predictor.model = data
    if predictor.feature_columns is None:
        predictor.feature_columns = []
    return predictor.model is not None

def export_training_bundle_zip(
    sma_dataframes: dict | None,
    model_pkl_bytes: bytes | None,
    meta: dict | None = None,
) -> bytes:
    """
    SMA 데이터 + 모델(pkl)을 하나의 zip으로 묶어서 다운로드합니다.
    구성:
      - meta.json
      - model.pkl (있을 때)
      - sma/*.csv (있을 때)
    """
    meta = meta or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps({"exported_at": datetime.now().isoformat(), **meta}, ensure_ascii=False, indent=2))

        if model_pkl_bytes:
            zf.writestr("model.pkl", model_pkl_bytes)

        if sma_dataframes and isinstance(sma_dataframes, dict):
            for k, df in sma_dataframes.items():
                if df is None or (hasattr(df, "empty") and df.empty):
                    continue
                tmp = df.copy()
                if "Date" in tmp.columns:
                    tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                zf.writestr(f"sma/{k}.csv", tmp.to_csv(index=False).encode("utf-8-sig"))

    return buf.getvalue()

def import_training_bundle_zip(file_bytes: bytes) -> tuple[bytes | None, dict, dict]:
    """
    export_training_bundle_zip()로 만든 번들 zip을 로드합니다.
    Returns:
      (model_pkl_bytes or None, sma_dataframes(dict), meta(dict))
    """
    model_bytes: bytes | None = None
    sma: dict = {}
    meta: dict = {}

    buf = io.BytesIO(file_bytes)
    with zipfile.ZipFile(buf, mode="r") as zf:
        names = zf.namelist()

        if "meta.json" in names:
            try:
                meta = json.loads(zf.read("meta.json").decode("utf-8"))
            except Exception:
                meta = {}

        # 모델
        if "model.pkl" in names:
            model_bytes = zf.read("model.pkl")

        # SMA: sma/*.csv 우선, 없으면 루트의 *.csv도 수용
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        for name in csv_names:
            base = os.path.basename(name)
            key = os.path.splitext(base)[0]
            # 경로가 sma/ 하위이면 그대로 key 사용
            if name.replace("\\", "/").startswith("sma/"):
                pass
            else:
                # 루트 csv도 허용 (단, meta.csv 같은 것은 제외)
                if key.lower() == "meta":
                    continue
            try:
                df = pd.read_csv(io.BytesIO(zf.read(name)))
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                sma[key] = df
            except Exception:
                continue

    return model_bytes, sma, meta

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 모델 로드/학습 선택
# 기존 모델 사용: 이전에 학습하여 저장된 모델 파일(stock_model.pkl)을 불러와서 사용합니다.
# 새 모델 학습: S&P500 주식 데이터를 수집하고 RandomForest 모델을 처음부터 학습합니다.
model_option = st.sidebar.radio(
    "모델 옵션",
    ["기존 모델 사용", "새 모델 학습"],
    index=0,
    help="기존 모델 사용: 저장된 모델 파일을 불러옵니다. 새 모델 학습: 처음부터 모델을 학습합니다."
)

max_tickers = st.sidebar.slider("분석할 주식 수", 10, 500, 100)

# =========================
# SMA 데이터 (소스/업로드)
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📦 이평선(SMA) 데이터")

has_sma_cache = ("sma_dataframes" in st.session_state and isinstance(st.session_state["sma_dataframes"], dict) and bool(st.session_state["sma_dataframes"]))

sma_source = st.sidebar.radio(
    "SMA 데이터 소스",
    ["업로드/캐시 사용(재계산 안함)", "새로 수집(시간 소요)"],
    index=0 if has_sma_cache else 1,
    help="업로드/캐시를 선택하면 S&P500 이평선 계산을 건너뛰고, 저장된 SMA 데이터로 바로 진행합니다.",
)

use_uploaded_sma = (sma_source == "업로드/캐시 사용(재계산 안함)")
if use_uploaded_sma and not has_sma_cache:
    st.sidebar.warning("⚠️ 업로드/캐시 SMA 데이터가 없습니다. 아래에서 업로드하거나, '새로 수집'을 선택하세요.")

# 모델 업/다운로드 UI (요청한 순서상 SMA 업로드와 함께 하단에 배치)
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 모델 파일")
st.sidebar.caption("⚠️ pkl 업로드는 신뢰할 수 있는 파일만 사용하세요(보안상 pickle 위험).")

# 모델 학습 시작 버튼(요청한 사이드바 순서에 맞춤)
start_train_clicked = False
if model_option == "새 모델 학습":
    start_train_clicked = st.sidebar.button("모델 학습 시작", type="primary")

# 패키지 zip 업로드: 모델 + SMA를 한 번에
uploaded_bundle_zip = st.sidebar.file_uploader(
    "패키지 업로드 (.zip) - 모델+SMA",
    type=["zip"],
    help="SMA+모델 패키지(zip)를 업로드하면 모델과 SMA 데이터를 한 번에 세션에 반영합니다.",
    key="bundle_zip_uploader",
)

if uploaded_bundle_zip is not None:
    try:
        sig = (uploaded_bundle_zip.name, uploaded_bundle_zip.size)
        prev_sig = st.session_state.get("_bundle_upload_sig")

        model_bytes, sma_loaded, meta_loaded = import_training_bundle_zip(uploaded_bundle_zip.getvalue())

        if model_bytes:
            st.session_state["uploaded_model_bytes"] = model_bytes
        if sma_loaded:
            st.session_state["sma_dataframes"] = sma_loaded
            st.session_state["sma_collector_date"] = datetime.now().date().isoformat()
            st.session_state["sma_upload_meta"] = meta_loaded

        st.session_state["_bundle_upload_sig"] = sig

        if model_bytes or sma_loaded:
            st.sidebar.success(
                f"✅ 패키지 업로드 완료 (모델: {'O' if model_bytes else 'X'}, SMA: {len(sma_loaded)})"
            )
            # 업로드 직후 라디오/캐시 상태 반영을 위해 1회 rerun
            if prev_sig != sig:
                st.rerun()
        else:
            st.sidebar.warning("업로드한 zip에서 model.pkl 또는 sma/*.csv 를 찾지 못했습니다.")
    except Exception as e:
        st.sidebar.error(f"패키지 업로드 실패: {str(e)}")

# 모델 업로드 (기존 모델 사용 시 업로드한 모델 우선)
uploaded_model_pkl = st.sidebar.file_uploader(
    "모델 업로드 (.pkl)",
    type=["pkl"],
    help="학습한 모델 pkl을 업로드하면 '기존 모델 사용'에서 업로드한 모델을 우선 사용합니다.",
    key="model_pkl_uploader",
)

if uploaded_model_pkl is not None:
    try:
        sig = (uploaded_model_pkl.name, uploaded_model_pkl.size)
        prev_sig = st.session_state.get("_model_upload_sig")
        st.session_state["uploaded_model_bytes"] = uploaded_model_pkl.getvalue()
        st.session_state["_model_upload_sig"] = sig
        st.sidebar.success("✅ 모델 업로드 완료")
        if prev_sig != sig:
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"모델 업로드 실패: {str(e)}")

# SMA 업로드(사이드바 가장 아래)
uploaded_sma_zip = st.sidebar.file_uploader(
    "SMA 데이터 업로드 (.zip)",
    type=["zip"],
    help="이 앱에서 다운로드한 SMA 데이터 zip을 업로드하면, 다음부터 SMA 계산 없이 재사용합니다.",
    key="sma_zip_uploader",
)

if uploaded_sma_zip is not None:
    try:
        sig = (uploaded_sma_zip.name, uploaded_sma_zip.size)
        prev_sig = st.session_state.get("_sma_upload_sig")

        sma_loaded, meta_loaded = import_sma_data_zip(uploaded_sma_zip.getvalue())
        if sma_loaded:
            st.session_state["sma_dataframes"] = sma_loaded
            st.session_state["sma_collector_date"] = datetime.now().date().isoformat()
            st.session_state["sma_upload_meta"] = meta_loaded
            st.session_state["_sma_upload_sig"] = sig
            st.sidebar.success(f"✅ SMA 데이터 업로드 완료 ({len(sma_loaded)}개 파일)")

            # 업로드 직후 즉시 has_sma_cache를 반영하기 위해 1회 rerun
            if prev_sig != sig:
                st.rerun()
        else:
            st.sidebar.warning("업로드된 zip에서 SMA CSV를 찾지 못했습니다.")
    except Exception as e:
        st.sidebar.error(f"SMA 업로드 실패: {str(e)}")

# 모델 및 예측기 초기화
predictor = StockPredictor('stock_model.pkl')

# 모델 로드 또는 학습
if model_option == "기존 모델 사용":
    # 업로드한 모델이 있으면 우선 사용
    if "uploaded_model_bytes" in st.session_state and st.session_state["uploaded_model_bytes"]:
        try:
            ok = load_model_from_pkl_bytes(predictor, st.session_state["uploaded_model_bytes"])
            if ok:
                st.sidebar.info("✅ 업로드한 모델을 사용합니다.")
            else:
                st.sidebar.warning("⚠️ 업로드한 모델을 로드하지 못했습니다. 저장된 모델을 시도합니다.")
                if not predictor.load_model():
                    st.sidebar.warning("⚠️ 저장된 모델이 없습니다. 새 모델을 학습하세요.")
                    model_option = "새 모델 학습"
        except Exception as e:
            st.sidebar.error(f"업로드 모델 로드 실패: {str(e)}")
            if not predictor.load_model():
                st.sidebar.warning("⚠️ 저장된 모델이 없습니다. 새 모델을 학습하세요.")
                model_option = "새 모델 학습"
    else:
        if not predictor.load_model():
            st.sidebar.warning("⚠️ 저장된 모델이 없습니다. 새 모델을 학습하세요.")
            model_option = "새 모델 학습"
        else:
            st.sidebar.info("✅ 모델이 로드되었습니다.")
            # 예측 실행 플래그는 버튼 클릭 시에만 True로 설정됨

if model_option == "새 모델 학습":
    if start_train_clicked:
        # 진행도 표시를 위한 상태 컨테이너 생성
        progress_container = st.container()
        status_container = st.container()
        
        try:
            with status_container:
                with st.status("🔄 모델 학습 진행 중...", expanded=True) as status:
                    # 1단계: S&P500 티커 리스트 가져오기
                    st.write("📋 1단계: S&P500 티커 리스트 수집 중...")
                    collector = StockDataCollector()
                    ticker_list = get_sp500_tickers()
                    st.write(f"✅ {len(ticker_list)}개 티커 수집 완료")
                    
                    # 2단계: 주식 데이터 수집
                    if use_uploaded_sma and "sma_dataframes" in st.session_state and st.session_state["sma_dataframes"]:
                        st.write("📊 2단계: 업로드한 SMA 데이터 사용 중... (재계산 없음)")
                        collector.dataframes = st.session_state["sma_dataframes"].copy()
                        tickers_above, tickers_below = [], []
                    else:
                        st.write(f"📊 2단계: {max_tickers}개 주식 이동평균선(SMA) 데이터 수집 중...")
                        st.write("⏳ 시간이 오래 걸릴 수 있습니다 (주식당 약 1-2초 소요)")

                        # 진행도 바 생성
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(value):
                            progress_bar.progress(value)

                        def update_status(text):
                            status_text.text(text)

                        tickers_above, tickers_below = collector.collect_sma_data(
                            ticker_list,
                            max_tickers,
                            progress_callback=update_progress,
                            status_callback=update_status
                        )
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ 데이터 수집 완료 (위: {len(tickers_above)}, 아래: {len(tickers_below)})")

                    # SMA50 위/아래 기업 목록 표시 + 세션 저장 (여기서 1회만 렌더링)
                    asof, above_df, below_df = build_sma50_tables_from_collector(collector)
                    if above_df is not None and below_df is not None:
                        st.session_state['sma50_asof'] = asof
                        st.session_state['sma50_above_df'] = above_df
                        st.session_state['sma50_below_df'] = below_df
                        # SMA 데이터도 세션에 저장 (다운로드/예측 재사용)
                        st.session_state["sma_dataframes"] = collector.dataframes.copy()
                        st.session_state["sma_collector_date"] = datetime.now().date().isoformat()

                        st.markdown("---")
                        st.subheader("📌 (오늘/최근 거래일 기준) SMA50 위/아래 기업 목록")
                        st.caption(f"기준일: {asof.strftime('%Y-%m-%d') if hasattr(asof, 'strftime') else str(asof)[:10]}")
                        render_sma50_tables_with_sort(above_df, below_df, key_prefix="sma50_train")
                        # diff_pct 추이(Top 상승) 추가
                        render_sma50_diffpct_trend_from_sma_dataframes(st.session_state.get("sma_dataframes"), key_prefix="sma50_diffpct_train")

                        # 수집 직후 바로 다운로드할 수 있도록 본문에도 버튼 제공
                        try:
                            sma_zip = export_sma_data_zip(
                                st.session_state["sma_dataframes"],
                                meta={
                                    "source": "collected",
                                    "max_tickers": max_tickers,
                                    "windows": getattr(collector, "list_window", None),
                                },
                            )
                            st.download_button(
                                label="⬇️ 수집한 SMA 데이터 다운로드(.zip)",
                                data=sma_zip,
                                file_name=f"sma_data_{datetime.now().date().isoformat()}_{max_tickers}.zip",
                                mime="application/zip",
                                help="한 번 수집한 SMA 데이터를 저장해두면, 다음에는 업로드/캐시로 재계산 없이 사용할 수 있습니다.",
                            )
                        except Exception as e:
                            st.warning(f"SMA zip 생성 실패: {str(e)}")
                    
                    # 3단계: 타겟 데이터 준비
                    st.write("🎯 3단계: 타겟 데이터 준비 중...")
                    # 기본값(for_prediction=False)이므로 키워드 인자를 넘기지 않아도 됩니다.
                    # (일부 환경/캐시에서 구버전 모듈이 로드될 때 호환성 이슈 방지)
                    spy = collector.prepare_target_data('IXIC')
                    st.write("✅ 타겟 데이터 준비 완료")
                    
                    # 4단계: 특성 데이터 추가
                    st.write("🔧 4단계: 기술적 지표 및 외부 데이터 추가 중...")
                    spy = collector.add_features(spy)
                    st.write("✅ 특성 데이터 추가 완료")
                    
                    # 5단계: 학습 데이터 준비
                    st.write("📦 5단계: 학습 데이터 준비 중...")
                    X = build_feature_matrix(spy)
                    y = spy['Target']
                    st.write(f"✅ 학습 데이터 준비 완료 (샘플 수: {len(X)}, 특성 수: {len(X.columns)})")
                    
                    # 6단계: 모델 학습
                    st.write("🤖 6단계: RandomForest 모델 학습 중...")
                    st.write("⏳ 이 과정은 시간이 오래 걸릴 수 있습니다 (2000개 트리 생성, 수분 소요)")
                    
                    # 모델 학습 진행도 표시
                    train_progress = st.progress(0)
                    train_status = st.empty()
                    
                    def update_train_progress(value):
                        train_progress.progress(value)
                    
                    def update_train_status(text):
                        train_status.text(text)
                    
                    train_score, test_score, oob_score = predictor.train_model(
                        X, y,
                        progress_callback=update_train_progress,
                        status_callback=update_train_status
                    )
                    
                    train_progress.progress(1.0)
                    train_status.text("✅ 모델 학습 완료!")
                    
                    # 7단계: 모델 저장
                    st.write("💾 7단계: 모델 저장 중...")
                    predictor.save_model()
                    st.write("✅ 모델 저장 완료")

                    # 학습 결과/아티팩트는 세션에 저장 (다운로드 클릭 rerun에도 화면 유지)
                    st.session_state["train_completed"] = True
                    st.session_state["train_scores"] = {
                        "train_score": float(train_score),
                        "test_score": float(test_score),
                        "oob_score": float(oob_score),
                    }
                    st.session_state["trained_model_bytes"] = export_model_pkl_bytes(predictor)
                    
                    # 학습 시 수집한 이평선 데이터를 세션 상태에 저장 (예측 시 재사용)
                    today_str = datetime.now().date().isoformat()
                    st.session_state['sma_dataframes'] = collector.dataframes.copy()
                    st.session_state['sma_collector_date'] = today_str
                    st.write("✅ 이평선 데이터 저장 완료 (예측 시 재사용)")
                    
                    # 학습 완료 후 예측 확률 계산 (예측 데이터 사용)
                    st.write("📊 예측 데이터로 예측 확률 계산 중...")
                    try:
                        # 예측용 최신 데이터 준비
                        sma_dataframes = collector.dataframes.copy()
                        X_pred, spy_pred = prepare_prediction_data(
                            progress_callback=None,
                            status_callback=None,
                            sma_dataframes=sma_dataframes
                        )
                        # 업로드/기존 모델과 feature mismatch 방지: Series로 전달(컬럼 align 가능)
                        current_prob = predictor.get_current_probability(X_pred.iloc[-1])
                        
                        # 예측 확률을 세션 상태에 저장 (하단 표시용)
                        st.session_state['training_prediction_prob'] = current_prob
                        st.session_state['training_prediction_X'] = X_pred
                        st.session_state['training_prediction_spy'] = spy_pred
                        st.write(f"✅ 예측 확률 계산 완료: {current_prob*100:.2f}%")
                    except Exception as e:
                        st.warning(f"예측 확률 계산 중 오류: {str(e)}")
                    
                    status.update(label="✅ 모델 학습 완료!", state="complete")
            
            # 학습 결과 표시
            st.sidebar.success("✅ 모델 학습 완료!")
            st.sidebar.metric("테스트 정확도", f"{test_score:.3f}")
            st.sidebar.metric("OOB 정확도", f"{oob_score:.3f}")
            
            # 학습 결과 상세 표시
            st.success("🎉 모델 학습이 성공적으로 완료되었습니다!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("훈련 세트 정확도", f"{train_score:.3f}")
            with col2:
                st.metric("테스트 세트 정확도", f"{test_score:.3f}")
            with col3:
                st.metric("OOB 정확도", f"{oob_score:.3f}")

            # 다운로드(패키지): SMA + 모델을 한 번에
            bundle = export_training_bundle_zip(
                sma_dataframes=st.session_state.get("sma_dataframes"),
                model_pkl_bytes=st.session_state.get("trained_model_bytes"),
                meta={
                    "type": "bundle",
                    "max_tickers": max_tickers,
                    "note": "SMA(csv) + model(pkl) bundle",
                },
            )
            st.download_button(
                label="⬇️ SMA+모델 패키지 다운로드(.zip)",
                data=bundle,
                file_name=f"bundle_{datetime.now().date().isoformat()}_{max_tickers}.zip",
                mime="application/zip",
                help="SMA 데이터와 학습된 모델을 한 번에 다운로드합니다. (다운로드 클릭 시 rerun되더라도 결과 화면은 유지됩니다.)",
            )
            
            # 학습 후 예측 확률 표시 (하단)
            if 'training_prediction_prob' in st.session_state:
                st.markdown("---")
                st.subheader("📊 예측 데이터 기반 예측 확률")
                training_prob = st.session_state['training_prediction_prob']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "현재 상승 확률",
                        f"{training_prob*100:.2f}%",
                        delta=f"{(training_prob-0.5)*100:.2f}%p",
                        delta_color="normal" if training_prob > 0.5 else "inverse"
                    )
                
                with col2:
                    if 'training_prediction_spy' in st.session_state:
                        training_spy = st.session_state['training_prediction_spy']
                        last_date = training_spy.index[-1]
                        last_date_str = str(last_date)[:10] if hasattr(last_date, '__str__') else str(last_date)
                        st.metric("데이터 기준일", last_date_str)
                
                # 확률 해석
                if training_prob >= 0.7:
                    st.success(f"🟢 높은 상승 확률 ({training_prob*100:.1f}%) - 강한 매수 신호")
                elif training_prob >= 0.6:
                    st.info(f"🔵 중간 상승 확률 ({training_prob*100:.1f}%) - 약한 매수 신호")
                elif training_prob >= 0.4:
                    st.warning(f"🟡 중립 ({training_prob*100:.1f}%) - 관망 권장")
                else:
                    st.error(f"🔴 낮은 상승 확률 ({training_prob*100:.1f}%) - 매도 고려")
                
                # 최근 5일간 예측 확률 (예측 데이터 기반)
                if 'training_prediction_X' in st.session_state:
                    training_X = st.session_state['training_prediction_X']
                    prob_history_recent = predictor.get_probability_history(training_X, days=5)
                    
                    if prob_history_recent is not None and len(prob_history_recent) > 0:
                        st.markdown("---")
                        st.subheader("📅 최근 5일간 예측 확률 (예측 데이터 기반)")
                        
                        # get_probability_history는 k=0부터 시작하므로:
                        # - 첫 번째 행: 가장 최근 날짜 (X.iloc[-1])
                        # - 다섯 번째 행: 4일 전 날짜 (X.iloc[-5])
                        # head(5)로 최신 5일을 가져오면 이미 최신 날짜가 첫 번째 행에 있음
                        # 역순 정렬하여 왼쪽부터 오래된 날짜 → 최신 날짜 순으로 표시
                        recent_5days = prob_history_recent.head(5).copy()
                        # 역순 정렬 (왼쪽: 오래된 날짜, 오른쪽: 최신 날짜)
                        recent_5days = recent_5days.iloc[::-1].copy()
                        recent_5days['Probability'] = recent_5days['Probability'] * 100
                        recent_5days['날짜'] = recent_5days.index.strftime('%Y-%m-%d')
                        recent_5days['예측 확률 (%)'] = recent_5days['Probability']
                        
                        # 실제 사용한 데이터도 함께 표시
                        display_df = recent_5days[['날짜', '예측 확률 (%)']].copy()
                        # HTML 저장(리포트)용 최근 5일 상승 확률 저장
                        try:
                            st.session_state["report_recent5_up"] = [
                                {"date": str(r.get("날짜")), "prob": float(r.get("예측 확률 (%)")) / 100.0}
                                for r in display_df.to_dict("records")
                                if r.get("날짜") is not None and r.get("예측 확률 (%)") is not None
                            ]
                        except Exception:
                            pass
                        
                        # 5개의 컬럼으로 날짜별 확률 표시
                        cols = st.columns(5)
                        for idx, (date_idx, row) in enumerate(display_df.iterrows()):
                            with cols[idx]:
                                prob_value = row['예측 확률 (%)']
                                date_str = row['날짜']
                                
                                # 확률에 따른 색상 결정
                                if prob_value >= 70:
                                    delta_color = "normal"
                                elif prob_value >= 60:
                                    delta_color = "normal"
                                elif prob_value >= 40:
                                    delta_color = "off"
                                else:
                                    delta_color = "inverse"
                                
                                # 이전 날짜와의 차이 계산
                                delta = None
                                if idx < len(display_df) - 1:
                                    prev_prob = display_df.iloc[idx + 1]['예측 확률 (%)']
                                    delta = prob_value - prev_prob
                                
                                st.metric(
                                    label=date_str,
                                    value=f"{prob_value:.2f}%",
                                    delta=f"{delta:.2f}%p" if delta is not None else None,
                                    delta_color=delta_color if delta is not None else "off"
                                )
                        
                        # 사용한 데이터 상세 보기
                        with st.expander("📊 사용한 데이터 상세 보기"):
                            st.write("**최근 5일간 예측 확률 계산에 사용한 데이터:**")
                            
                            # 각 날짜별로 사용한 feature 데이터 표시
                            for date_idx in recent_5days.index:
                                date_str = str(date_idx)[:10]
                                st.write(f"### {date_str}")
                                
                                # 해당 날짜의 feature 데이터 가져오기
                                if date_idx in training_X.index:
                                    feature_data = training_X.loc[date_idx]
                                    feature_df = pd.DataFrame({
                                        'Feature': feature_data.index,
                                        'Value': feature_data.values
                                    })
                                    st.dataframe(feature_df, use_container_width=True, hide_index=True)
                                
                                # 해당 날짜의 주가 정보도 표시
                                if 'training_prediction_spy' in st.session_state:
                                    training_spy = st.session_state['training_prediction_spy']
                                    if date_idx in training_spy.index:
                                        price_info = training_spy.loc[date_idx]
                                        price_cols = st.columns(4)
                                        with price_cols[0]:
                                            st.metric("종가", f"${price_info.get('Close', 0):,.2f}")
                                        with price_cols[1]:
                                            if 'rsi' in price_info:
                                                st.metric("RSI", f"{price_info['rsi']:.2f}")
                                        with price_cols[2]:
                                            if 'vix' in price_info:
                                                st.metric("VIX", f"{price_info['vix']:.2f}")
                                        with price_cols[3]:
                                            if 'Change20day' in price_info:
                                                st.metric("Change20day", f"{price_info['Change20day']:.2f}%")
                                st.markdown("---")

                    # 최근 5일 아래에 추가 시각화(확률 추이/지수 비교)
                    st.markdown("---")
                    st.subheader("📈 확률 추이 그래프 (예측 데이터 기반)")

                    prob_history = predictor.get_probability_history(training_X, days=min(500, len(training_X)))
                    if prob_history is not None and len(prob_history) > 0:
                        prob_history = prob_history.sort_index()
                        prob_dates = prob_history.index

                        # 주가 데이터(나스닥)도 함께 표시
                        try:
                            price_data = fdr.DataReader('IXIC', prob_dates[0], prob_dates[-1])
                            price_aligned = price_data.reindex(prob_dates, method='nearest')
                        except Exception:
                            price_aligned = None

                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(
                            go.Scatter(
                                x=prob_dates,
                                y=prob_history['Probability'] * 100,
                                name="상승 확률 (%)",
                                line=dict(color='skyblue', width=2),
                                mode='lines'
                            ),
                            secondary_y=False
                        )
                        fig.add_hline(
                            y=50,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.5,
                            annotation_text="기준선 (50%)",
                            secondary_y=False
                        )
                        if price_aligned is not None and len(price_aligned) > 0:
                            col = 'Adj Close' if 'Adj Close' in price_aligned.columns else ('Close' if 'Close' in price_aligned.columns else None)
                            if col is not None and not price_aligned[col].isna().all():
                                fig.add_trace(
                                    go.Scatter(
                                        x=prob_dates,
                                        y=price_aligned[col],
                                        name="IXIC 가격",
                                        line=dict(color='red', width=1, dash='dot'),
                                        opacity=0.5
                                    ),
                                    secondary_y=True
                                )

                        fig.update_xaxes(title_text="날짜")
                        fig.update_yaxes(title_text="상승 확률 (%)", secondary_y=False, range=[0, 100])
                        fig.update_yaxes(title_text="주가 (USD)", secondary_y=True)
                        fig.update_layout(
                            title="주가 상승 확률 추이 및 IXIC 가격 (예측 데이터 기반)",
                            height=600,
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("확률 히스토리가 비어있어 그래프를 그릴 수 없습니다.")

                    st.subheader("📊 최근 5년: 나스닥 / S&P500 / 다우존스 vs 예측 확률 (예측 데이터 기반)")
                    try:
                        prob_all = predictor.get_probability_history(training_X, days=len(training_X))
                        if prob_all is None or len(prob_all) == 0:
                            st.info("확률 히스토리를 계산할 수 없습니다.")
                        else:
                            prob_all = prob_all.sort_index()
                            end_dt = prob_all.index.max()
                            start_dt = end_dt - pd.DateOffset(years=5)
                            prob_5y = prob_all.loc[prob_all.index >= start_dt].copy()
                            prob_5y['prob_pct'] = prob_5y['Probability'] * 100.0

                            start_fetch = start_dt.date()
                            end_fetch = (end_dt.date() + timedelta(days=1))

                            nas_sym, nas = fetch_index_adj_close(['IXIC', '^IXIC'], start_fetch, end_fetch)
                            sp_sym, sp = fetch_index_adj_close(['US500', 'SPX', '^GSPC'], start_fetch, end_fetch)
                            dow_sym, dow = fetch_index_adj_close(['DJI', '^DJI'], start_fetch, end_fetch)

                            idx = prob_5y.index
                            nas_a = nas.reindex(idx, method='ffill')
                            sp_a = sp.reindex(idx, method='ffill')
                            dow_a = dow.reindex(idx, method='ffill')

                            nas_r = rebase_to_100(nas_a)
                            sp_r = rebase_to_100(sp_a)
                            dow_r = rebase_to_100(dow_a)

                            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                            fig2.add_trace(
                                go.Scatter(
                                    x=idx,
                                    y=prob_5y['prob_pct'],
                                    name="예측 상승 확률(%)",
                                    line=dict(color='skyblue', width=2),
                                    mode='lines'
                                ),
                                secondary_y=False
                            )
                            fig2.add_trace(
                                go.Scatter(
                                    x=idx,
                                    y=nas_r,
                                    name=f"나스닥({nas_sym}) 리베이스(100)",
                                    line=dict(color='#FF4B4B', width=1.5),
                                    mode='lines'
                                ),
                                secondary_y=True
                            )
                            fig2.add_trace(
                                go.Scatter(
                                    x=idx,
                                    y=sp_r,
                                    name=f"S&P500({sp_sym}) 리베이스(100)",
                                    line=dict(color='#2ECC71', width=1.5),
                                    mode='lines'
                                ),
                                secondary_y=True
                            )
                            fig2.add_trace(
                                go.Scatter(
                                    x=idx,
                                    y=dow_r,
                                    name=f"다우({dow_sym}) 리베이스(100)",
                                    line=dict(color='#F1C40F', width=1.5),
                                    mode='lines'
                                ),
                                secondary_y=True
                            )

                            fig2.update_xaxes(title_text="날짜")
                            fig2.update_yaxes(title_text="예측 상승 확률 (%)", secondary_y=False, range=[0, 100])
                            fig2.update_yaxes(title_text="지수 리베이스 (첫값=100)", secondary_y=True)
                            fig2.update_layout(
                                height=650,
                                hovermode='x unified',
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                                title="최근 5년 지수 움직임(리베이스)과 예측 확률 비교 (예측 데이터 기반)"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.warning(f"최근 5년 비교 그래프 생성 중 오류: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            st.exception(e)

    # 다운로드/새로고침 등 rerun 이후에도 학습 결과를 계속 보여주기
    if st.session_state.get("train_completed", False) and "train_scores" in st.session_state:
        st.markdown("---")
        st.subheader("📦 학습 결과(세션 유지)")
        scores = st.session_state["train_scores"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("훈련 세트 정확도", f"{scores.get('train_score', 0):.3f}")
        with c2:
            st.metric("테스트 세트 정확도", f"{scores.get('test_score', 0):.3f}")
        with c3:
            st.metric("OOB 정확도", f"{scores.get('oob_score', 0):.3f}")

        bundle = export_training_bundle_zip(
            sma_dataframes=st.session_state.get("sma_dataframes"),
            model_pkl_bytes=st.session_state.get("trained_model_bytes"),
            meta={
                "type": "bundle",
                "max_tickers": max_tickers,
                "note": "SMA(csv) + model(pkl) bundle",
            },
        )
        st.download_button(
            label="⬇️ SMA+모델 패키지 다운로드(.zip)",
            data=bundle,
            file_name=f"bundle_{datetime.now().date().isoformat()}_{max_tickers}.zip",
            mime="application/zip",
        )

# 메인 콘텐츠
# "새 모델 학습" 선택 시에는 메인 콘텐츠를 표시하지 않음 (사이드바 버튼만 표시)
if model_option == "새 모델 학습":
    # 사이드바에 "모델 학습 시작" 버튼이 표시됨
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.info("💡 사이드바에서 '모델 학습 시작' 버튼을 클릭하여 모델을 학습하세요.")
elif predictor.model is not None:
    # Hero 타이틀과 본문(버튼/섹션) 간 간격
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    # 기존 모델 사용 시에는 버튼을 눌러야만 예측 실행
    if model_option == "기존 모델 사용":
        # 캐시가 있으면(오늘 날짜 기준) 버튼을 누르지 않아도 그래프를 볼 수 있게 함
        today_str = datetime.now().date().isoformat()
        cache_key = f"spy_data_{max_tickers}"
        cache_date_key = f"cache_date_{max_tickers}"
        has_cache = (
            cache_key in st.session_state and
            cache_date_key in st.session_state and
            st.session_state[cache_date_key] == today_str
        )

        # 버튼이 클릭되었는지 확인
        button_clicked = st.button("🔄 예측 확률 계산", type="primary", key="predict_button")
        
        if button_clicked:
            st.session_state.run_prediction = True
        
        if not st.session_state.get('run_prediction', False) and not has_cache:
            st.info("💡 위의 '예측 확률 계산' 버튼을 클릭하여 예측을 시작하세요. (이미 계산된 캐시가 있으면 바로 표시됩니다.)")
            st.stop()
    
    # 현재 확률 계산
    st.subheader("📊 현재 예측 확률")

    # 기존 모델 사용 시에도 SMA(zip/패키지) 데이터가 있으면 동일 리스트 생성/표기
    ensure_sma50_tables_in_session_from_cached_sma()
    if 'sma50_above_df' in st.session_state and 'sma50_below_df' in st.session_state:
        if st.session_state.get('sma50_above_df') is not None and st.session_state.get('sma50_below_df') is not None:
            asof = st.session_state.get('sma50_asof', None)
            st.markdown("---")
            st.subheader("📌 (오늘/최근 거래일 기준) SMA50 위/아래 기업 목록")
            if asof is not None:
                st.caption(f"기준일: {asof.strftime('%Y-%m-%d') if hasattr(asof, 'strftime') else str(asof)[:10]}")
            render_sma50_tables_with_sort(
                st.session_state.get('sma50_above_df'),
                st.session_state.get('sma50_below_df'),
                key_prefix="sma50_main",
            )
            # diff_pct 추이(Top 상승) 추가
            render_sma50_diffpct_trend_from_sma_dataframes(st.session_state.get("sma_dataframes"), key_prefix="sma50_diffpct_main")
    
    try:
        # 세션 상태에 데이터 캐싱 (무한 루프 방지)
        cache_key = f"spy_data_{max_tickers}"
        cache_date_key = f"cache_date_{max_tickers}"
        
        # 캐시된 데이터가 있고 오늘 날짜와 같으면 재사용
        today_str = datetime.now().date().isoformat()
        use_cache = (
            cache_key in st.session_state and 
            cache_date_key in st.session_state and
            st.session_state[cache_date_key] == today_str
        )
        
        if use_cache:
            # 캐시된 데이터 사용
            spy = st.session_state[cache_key]
            X = st.session_state.get(f"{cache_key}_X", None)
            if X is None:
                X = build_feature_matrix(spy)
            
            # 캐시 사용 시에는 "한 번만" 플래그를 내립니다 (매 실행마다 내리면 rerun 루프 발생 가능)
            if model_option == "기존 모델 사용" and st.session_state.get('run_prediction', False):
                st.session_state.run_prediction = False
        else:
            # 예측 시에는 최신 데이터를 다시 가져와야 함
            # Train 데이터는 1월 2일까지지만, 예측 시에는 현재 날짜까지 데이터 필요
            # 참조 코드: k1 = fdr.DataReader('IXIC', '2015-01-01', a) - 여기서 a는 현재 날짜
            
            # 학습 시 수집한 이평선 데이터 재사용 (있으면)
            sma_dataframes = None
            if 'sma_dataframes' in st.session_state and 'sma_collector_date' in st.session_state:
                if st.session_state['sma_collector_date'] == today_str:
                    sma_dataframes = st.session_state['sma_dataframes']
            
            # 진행도 표시를 위한 컨테이너 생성
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # 진행도 콜백 함수 정의
            def update_progress(value):
                progress_bar.progress(value)
            
            def update_status(text):
                status_text.text(text)
            
            # 데이터 수집 (진행도 표시, 학습 시 수집한 이평선 데이터 재사용)
            X, spy = prepare_prediction_data(
                progress_callback=update_progress,
                status_callback=update_status,
                sma_dataframes=sma_dataframes
            )
            
            # 진행도 완료
            progress_bar.progress(1.0)
            status_text.text("✅ 데이터 준비 완료")
            
            # 세션 상태에 캐싱
            st.session_state[cache_key] = spy
            st.session_state[f"{cache_key}_X"] = X
            st.session_state[cache_date_key] = today_str
            
            # 예측 완료 후 플래그 리셋 (매 실행마다 내리면 rerun 루프 발생 가능)
            if model_option == "기존 모델 사용" and st.session_state.get('run_prediction', False):
                st.session_state.run_prediction = False
        
        # 최신 특성 데이터 (마지막 행이 최신 데이터)
        # 참조 코드: X9.iloc[-1] - 마지막 행을 사용하여 예측
        
        # 실제 최신 거래일 확인 (FinanceDataReader에서 직접 가져오기)
        try:
            latest_data = fdr.DataReader('IXIC', end_date=datetime.now().date() + timedelta(days=1))
            if len(latest_data) > 0:
                actual_last_date = latest_data.index[-1]
                # 주말/공휴일 제외하고 실제 거래일 확인
                actual_last_date_str = actual_last_date.strftime('%Y-%m-%d') if hasattr(actual_last_date, 'strftime') else str(actual_last_date)[:10]
            else:
                actual_last_date_str = str(spy.index[-1])[:10]
        except:
            actual_last_date_str = str(spy.index[-1])[:10]
        
        # 현재 확률 계산
        # 업로드/기존 모델과 feature mismatch 방지: Series로 전달(컬럼 align 가능)
        current_prob = predictor.get_current_probability(X.iloc[-1])
        # HTML 저장(리포트)용 세션 상태 저장
        try:
            st.session_state["report_current_up_prob"] = float(current_prob) if current_prob is not None else None
        except Exception:
            st.session_state["report_current_up_prob"] = None
        
        # 확률 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "현재 상승 확률",
                f"{current_prob*100:.2f}%",
                delta=f"{(current_prob-0.5)*100:.2f}%p" if current_prob > 0.5 else f"{(current_prob-0.5)*100:.2f}%p",
                delta_color="normal" if current_prob > 0.5 else "inverse"
            )
        
        with col2:
            current_price = spy['Close'].iloc[-1]
            st.metric("현재 가격 (IXIC)", f"${current_price:,.2f}")
        
        with col3:
            # 실제 최신 거래일 표시
            last_date = spy.index[-1]
            last_date_str = str(last_date)[:10] if hasattr(last_date, '__str__') else str(last_date)
            
            # 날짜 비교 및 경고 표시
            try:
                last_date_obj = pd.to_datetime(last_date_str).date()
                today = datetime.now().date()
                days_diff = (today - last_date_obj).days
                
                if days_diff > 3:  # 3일 이상 차이나면 경고
                    st.metric("마지막 업데이트", last_date_str, delta=f"{days_diff}일 전", delta_color="inverse")
                    st.caption(f"⚠️ 최신 거래일: {actual_last_date_str}")
                else:
                    st.metric("마지막 업데이트", last_date_str)
            except:
                st.metric("마지막 업데이트", last_date_str)
        
        # 확률 해석
        if current_prob >= 0.7:
            st.success(f"🟢 높은 상승 확률 ({current_prob*100:.1f}%) - 강한 매수 신호")
        elif current_prob >= 0.6:
            st.info(f"🔵 중간 상승 확률 ({current_prob*100:.1f}%) - 약한 매수 신호")
        elif current_prob >= 0.4:
            st.warning(f"🟡 중립 ({current_prob*100:.1f}%) - 관망 권장")
        else:
            st.error(f"🔴 낮은 상승 확률 ({current_prob*100:.1f}%) - 매도 고려")
        
        # 최근 5일간 예측 확률 표시
        st.markdown("---")
        st.subheader("📅 최근 5일간 예측 확률")
        
        try:
            # 최근 5일 확률 히스토리 가져오기
            prob_history_recent = predictor.get_probability_history(X, days=5)
            
            if prob_history_recent is not None and len(prob_history_recent) > 0:
                # get_probability_history는 k=0부터 시작하므로:
                # - 첫 번째 행: 가장 최근 날짜 (X.iloc[-1])
                # - 다섯 번째 행: 4일 전 날짜 (X.iloc[-5])
                # head(5)로 최신 5일을 가져오면 이미 최신 날짜가 첫 번째 행에 있음
                # 역순 정렬하여 왼쪽부터 오래된 날짜 → 최신 날짜 순으로 표시
                recent_5days = prob_history_recent.head(5).copy()
                # 역순 정렬 (왼쪽: 오래된 날짜, 오른쪽: 최신 날짜)
                recent_5days = recent_5days.iloc[::-1].copy()
                recent_5days['Probability'] = recent_5days['Probability'] * 100
                
                # 날짜 형식 변환
                recent_5days['날짜'] = recent_5days.index.strftime('%Y-%m-%d')
                recent_5days['예측 확률 (%)'] = recent_5days['Probability']
                
                # 컬럼 선택 및 정렬
                display_df = recent_5days[['날짜', '예측 확률 (%)']].copy()

                # HTML 저장(리포트)용 최근 5일 상승 확률 저장
                try:
                    st.session_state["report_recent5_up"] = [
                        {"date": str(r.get("날짜")), "prob": float(r.get("예측 확률 (%)")) / 100.0}
                        for r in display_df.to_dict("records")
                        if r.get("날짜") is not None and r.get("예측 확률 (%)") is not None
                    ]
                except Exception:
                    pass
                
                # 5개의 컬럼으로 날짜별 확률 표시
                cols = st.columns(5)
                for idx, (date_idx, row) in enumerate(display_df.iterrows()):
                    with cols[idx]:
                        prob_value = row['예측 확률 (%)']
                        date_str = row['날짜']
                        
                        # 확률에 따른 색상 결정
                        if prob_value >= 70:
                            color = "🟢"
                            delta_color = "normal"
                        elif prob_value >= 60:
                            color = "🔵"
                            delta_color = "normal"
                        elif prob_value >= 40:
                            color = "🟡"
                            delta_color = "off"
                        else:
                            color = "🔴"
                            delta_color = "inverse"
                        
                        # 이전 날짜와의 차이 계산 (첫 번째가 아니면)
                        delta = None
                        if idx < len(display_df) - 1:
                            prev_prob = display_df.iloc[idx + 1]['예측 확률 (%)']
                            delta = prob_value - prev_prob
                        
                        st.metric(
                            label=date_str,
                            value=f"{prob_value:.2f}%",
                            delta=f"{delta:.2f}%p" if delta is not None else None,
                            delta_color=delta_color if delta is not None else "off"
                        )
                
                # 테이블 형태로도 표시 (선택사항)
                with st.expander("📊 상세 데이터 보기"):
                    st.dataframe(
                        display_df.style.format({
                            '예측 확률 (%)': '{:.2f}%'
                        }).background_gradient(
                            subset=['예측 확률 (%)'],
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=100
                        ),
                        use_container_width=True
                    )
                
                # 사용한 데이터 상세 보기
                with st.expander("🔍 사용한 데이터 상세 보기"):
                    st.write("**최근 5일간 예측 확률 계산에 사용한 데이터:**")
                    
                    # 각 날짜별로 사용한 feature 데이터 표시
                    for date_idx in recent_5days.index:
                        date_str = str(date_idx)[:10]
                        st.write(f"### {date_str}")
                        
                        # 해당 날짜의 feature 데이터 가져오기
                        if date_idx in X.index:
                            feature_data = X.loc[date_idx]
                            feature_df = pd.DataFrame({
                                'Feature': feature_data.index,
                                'Value': feature_data.values
                            })
                            st.dataframe(feature_df, use_container_width=True, hide_index=True)
                        
                        # 해당 날짜의 주가 정보도 표시
                        if date_idx in spy.index:
                            price_info = spy.loc[date_idx]
                            price_cols = st.columns(4)
                            with price_cols[0]:
                                st.metric("종가", f"${price_info.get('Close', 0):,.2f}")
                            with price_cols[1]:
                                if 'rsi' in price_info:
                                    st.metric("RSI", f"{price_info['rsi']:.2f}")
                            with price_cols[2]:
                                if 'vix' in price_info:
                                    st.metric("VIX", f"{price_info['vix']:.2f}")
                            with price_cols[3]:
                                if 'Change20day' in price_info:
                                    st.metric("Change20day", f"{price_info['Change20day']:.2f}%")
                        st.markdown("---")
            else:
                st.info("최근 5일간의 확률 데이터를 가져올 수 없습니다.")
        except Exception as e:
            st.warning(f"최근 5일 확률 표시 중 오류: {str(e)}")

        # =========================
        # 예측확률별 향후 수익률 통계
        # =========================
        with st.expander("📊 예측확률별 향후 수익률 통계(백테스트 요약)", expanded=False):
            st.caption("기준: suik_rate(향후 15거래일 수익률, %)를 확률 구간별로 요약합니다. 마지막 N일 중 미래 수익률이 없는 구간(NaN)은 자동 제외됩니다.")

            if spy is None or X is None:
                st.info("통계를 계산할 데이터가 없습니다.")
            elif "suik_rate" not in spy.columns:
                st.info("suik_rate 컬럼이 없어 통계를 계산할 수 없습니다.")
            else:
                # 어떤 확률을 볼지(상승/하락)
                prob_options = ["상승 확률"]
                has_down_model = bool(st.session_state.get("down_model_bytes"))
                if has_down_model:
                    prob_options.append("하락 확률(모델)")

                c0, c1, c2 = st.columns([1.1, 1.0, 0.9])
                with c0:
                    which_prob = st.selectbox("확률 종류", prob_options, index=0, key="retstats_prob_kind")
                with c1:
                    lookback_days = st.slider("분석 기간(최근 N 거래일)", 200, max(200, min(5000, len(X))), min(1300, len(X)), step=100, key="retstats_days")
                with c2:
                    bin_size = st.selectbox("확률 구간 폭", [0.05, 0.1, 0.2], index=1, key="retstats_bin")

                def _get_model_for_kind():
                    if which_prob == "상승 확률":
                        return predictor
                    # 하락 모델 로드(세션 bytes 기반)
                    p_down = StockPredictor("stock_model_down.pkl")
                    ok = load_model_from_pkl_bytes(p_down, st.session_state.get("down_model_bytes"))
                    return p_down if ok else None

                model_use = _get_model_for_kind()
                if model_use is None or getattr(model_use, "model", None) is None:
                    st.warning("선택한 확률 모델을 사용할 수 없습니다. (하락 모델은 먼저 학습 필요)")
                else:
                    # 확률 히스토리 (최근 N일)
                    try:
                        prob_hist = model_use.get_probability_history(X, days=int(lookback_days))
                    except Exception as e:
                        st.warning(f"확률 히스토리 계산 실패: {str(e)}")
                        prob_hist = None

                    if prob_hist is None or len(prob_hist) == 0:
                        st.info("확률 히스토리가 비어 있습니다.")
                    else:
                        df = prob_hist.copy()
                        df = df.rename(columns={"Probability": "prob"})

                        # ✅ 하이라이트용 "가장 최근 날짜" 확률은 suik_rate 필터링 전(prob_hist) 기준으로 계산
                        # (suik_rate는 미래 수익률이라 최신 날짜는 NaN인 경우가 많아 dropna 후 기준을 쓰면 구간이 어긋남)
                        try:
                            latest_prob_for_highlight = float(prob_hist.sort_index().iloc[-1]["Probability"])
                        except Exception:
                            latest_prob_for_highlight = None

                        # 수익률 정합
                        try:
                            df["suik_rate"] = spy.reindex(df.index)["suik_rate"]
                        except Exception:
                            df["suik_rate"] = np.nan
                        df = df.dropna(subset=["prob", "suik_rate"]).copy()
                        if df.empty:
                            st.info("해당 기간에 유효한 suik_rate(미래 수익률) 데이터가 없어 통계를 만들 수 없습니다.")
                        else:
                            # 기본 요약
                            hit = (df["suik_rate"] > 0) if which_prob == "상승 확률" else (df["suik_rate"] < 0)
                            cA, cB, cC = st.columns(3)
                            with cA:
                                st.metric("표본 수", f"{len(df):,}")
                            with cB:
                                st.metric("평균 suik_rate(%)", f"{df['suik_rate'].mean():.2f}")
                            with cC:
                                st.metric("적중률(%)", f"{hit.mean()*100:.1f}")

                            # 구간별 통계
                            step = float(bin_size)
                            bins = np.arange(0.0, 1.0 + step, step)
                            df["prob_bin"] = pd.cut(df["prob"], bins=bins, include_lowest=True, right=False)

                            def _win_rate(s):
                                if which_prob == "상승 확률":
                                    return float((s > 0).mean())
                                return float((s < 0).mean())

                            grouped = (
                                df.groupby("prob_bin", dropna=True)
                                .agg(
                                    count=("suik_rate", "size"),
                                    prob_mean=("prob", "mean"),
                                    ret_mean=("suik_rate", "mean"),
                                    ret_median=("suik_rate", "median"),
                                    ret_p25=("suik_rate", lambda x: float(np.nanquantile(x, 0.25))),
                                    ret_p75=("suik_rate", lambda x: float(np.nanquantile(x, 0.75))),
                                    win_rate=("suik_rate", _win_rate),
                                )
                                .reset_index()
                            )
                            grouped["win_rate(%)"] = grouped["win_rate"] * 100.0
                            grouped = grouped.drop(columns=["win_rate"])

                            # 보기 좋게 정렬(확률 낮은→높은)
                            grouped = grouped.sort_values("prob_bin").reset_index(drop=True)

                            # 가장 최근 날짜 예측치가 포함된 구간 하이라이트(최신 확률 기준)
                            try:
                                if latest_prob_for_highlight is None:
                                    latest_bin_iv = None
                                    latest_bin = None
                                else:
                                    latest_bin_iv = pd.cut(
                                        pd.Series([latest_prob_for_highlight]),
                                        bins=bins,
                                        include_lowest=True,
                                        right=False,
                                    ).iloc[0]
                                    latest_bin = str(latest_bin_iv)
                            except Exception:
                                latest_bin_iv = None
                                latest_bin = None

                            def _hl_latest(row):
                                if latest_bin_iv is None:
                                    return [""] * len(row)
                                try:
                                    is_hit = (row.get("prob_bin") == latest_bin_iv)
                                except Exception:
                                    is_hit = (str(row.get("prob_bin")) == str(latest_bin))
                                return ["background-color: #fff3bf" if is_hit else "" for _ in row]

                            st.dataframe(
                                grouped.style.apply(_hl_latest, axis=1),
                                use_container_width=True,
                            )

                            # 표 아래 시각화(막대: 평균 수익률, 꺾은선: 적중률)
                            try:
                                x_labels = grouped["prob_bin"].astype(str)
                                # 최신 확률 구간 하이라이트 색(막대/선 마커)
                                try:
                                    highlight_mask = grouped["prob_bin"].apply(lambda v: v == latest_bin_iv)
                                except Exception:
                                    highlight_mask = grouped["prob_bin"].astype(str) == str(latest_bin)

                                bar_colors = [
                                    "#FFC107" if bool(m) else SHINHAN_SKY
                                    for m in highlight_mask.tolist()
                                ]
                                fig = make_subplots(specs=[[{"secondary_y": True}]])
                                fig.add_trace(
                                    go.Bar(
                                        x=x_labels,
                                        y=grouped["ret_mean"],
                                        name="평균 suik_rate(%)",
                                        marker_color=bar_colors,
                                    ),
                                    secondary_y=False,
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_labels,
                                        y=grouped["win_rate(%)"],
                                        name="적중률(%)",
                                        mode="lines+markers",
                                        line=dict(color=SHINHAN_BLUE, width=2),
                                        marker=dict(
                                            size=7,
                                            color=["#FFC107" if bool(m) else SHINHAN_BLUE for m in highlight_mask.tolist()],
                                        ),
                                    ),
                                    secondary_y=True,
                                )
                                fig.update_xaxes(title_text="예측 확률 구간")
                                fig.update_yaxes(title_text="평균 suik_rate(%)", secondary_y=False)
                                fig.update_yaxes(title_text="적중률(%)", secondary_y=True, range=[0, 100])
                                fig.update_layout(
                                    height=420,
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                # HTML 저장(리포트)용: 표/그래프 저장
                                try:
                                    st.session_state["report_return_stats"] = {
                                        "kind": which_prob,
                                        "lookback_days": int(lookback_days),
                                        "bin_size": float(step),
                                        "latest_bin": latest_bin,
                                        "rows": grouped.to_dict("records"),
                                    }
                                    st.session_state["report_return_stats_fig_json"] = fig.to_json()
                                except Exception:
                                    pass
                            except Exception as e:
                                st.caption(f"그래프 생성 실패: {str(e)}")
        
        # 확률 추이 그래프
        st.subheader("📈 확률 추이 그래프")
        
        prob_history = predictor.get_probability_history(X, days=500)
        
        if prob_history is not None and len(prob_history) > 0:
            # 확률 데이터의 날짜 범위로 주가 데이터 가져오기
            start_date_prob = prob_history.index[0]
            end_date_prob = prob_history.index[-1]
            
            # 주가 데이터 가져오기 (확률 데이터와 같은 날짜 범위)
            # (네트워크/프록시 환경 이슈가 있어도 아래 섹션(하락 확률)까지는 렌더링되도록 방어)
            try:
                price_data = fdr.DataReader('IXIC', start_date_prob, end_date_prob)
            except Exception:
                price_data = None
            
            # 확률 데이터의 날짜를 인덱스로 사용
            prob_dates = prob_history.index
            
            # 주가 데이터를 확률 데이터의 날짜와 맞추기
            try:
                price_aligned = price_data.reindex(prob_dates, method='nearest') if price_data is not None else None
            except Exception:
                price_aligned = None
            
            # 그래프 생성
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 확률 그래프 (확률이 있는 날짜만 표시)
            fig.add_trace(
                go.Scatter(
                    x=prob_dates,
                    y=prob_history['Probability'] * 100,
                    name="상승 확률 (%)",
                    line=dict(color='skyblue', width=2),
                    mode='lines+markers',
                    marker=dict(size=4)
                ),
                secondary_y=False
            )
            
            # 기준선 (50%)
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text="기준선 (50%)",
                secondary_y=False
            )
            
            # 주가 그래프 (확률 데이터와 같은 날짜 사용)
            if price_aligned is not None and len(price_aligned) > 0:
                col_price = 'Adj Close' if 'Adj Close' in price_aligned.columns else ('Close' if 'Close' in price_aligned.columns else None)
            else:
                col_price = None

            if col_price is not None and not price_aligned[col_price].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=prob_dates,
                        y=price_aligned[col_price],
                        name="IXIC 가격",
                        line=dict(color='red', width=1, dash='dot'),
                        opacity=0.5
                    ),
                    secondary_y=True
                )
            
            fig.update_xaxes(title_text="날짜")
            fig.update_yaxes(title_text="상승 확률 (%)", secondary_y=False)
            fig.update_yaxes(title_text="주가 (USD)", secondary_y=True)
            fig.update_layout(
                title="주가 상승 확률 추이 및 IXIC 가격",
                height=600,
                hovermode='x unified',
                xaxis=dict(
                    tickmode='linear',
                    tick0=prob_dates[0],
                    dtick=86400000.0 * 30  # 약 30일 간격
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            # HTML 저장(리포트)용 figure 저장 (json)
            try:
                st.session_state["report_fig_prob_trend_json"] = fig.to_json()
            except Exception:
                pass
        else:
            st.info("확률 히스토리가 비어있어 그래프를 그릴 수 없습니다. (먼저 예측 확률 계산이 완료되어야 합니다.)")

        # =========================
        # 최근 5년: 나스닥/ S&P/ 다우 vs 예측확률 비교
        # =========================
        st.subheader("📊 최근 5년: 나스닥 / S&P500 / 다우존스 vs 예측 확률")

        try:
            # 확률 히스토리(최대한 길게) → 최근 5년 필터
            prob_all = predictor.get_probability_history(X, days=len(X))
            if prob_all is None or len(prob_all) == 0:
                st.info("확률 히스토리를 계산할 수 없습니다.")
            else:
                prob_all = prob_all.sort_index()
                end_dt = prob_all.index.max()
                start_dt = end_dt - pd.DateOffset(years=5)
                prob_5y = prob_all.loc[prob_all.index >= start_dt].copy()
                prob_5y['prob_pct'] = prob_5y['Probability'] * 100.0

                # 지수 데이터 가져오기 (심볼 후보를 여러 개 둬서 호환성 확보)
                start_fetch = start_dt.date()
                end_fetch = (end_dt.date() + timedelta(days=1))

                nas_sym, nas = fetch_index_adj_close(['IXIC', '^IXIC'], start_fetch, end_fetch)
                sp_sym, sp = fetch_index_adj_close(['US500', 'SPX', '^GSPC'], start_fetch, end_fetch)
                dow_sym, dow = fetch_index_adj_close(['DJI', '^DJI'], start_fetch, end_fetch)

                # 확률 날짜 인덱스에 맞춰 정렬(가장 최근 거래일 기준으로 forward fill)
                idx = prob_5y.index
                nas_a = nas.reindex(idx, method='ffill')
                sp_a = sp.reindex(idx, method='ffill')
                dow_a = dow.reindex(idx, method='ffill')

                # 스케일 차이를 없애기 위해 100 기준 리베이스
                nas_r = rebase_to_100(nas_a)
                sp_r = rebase_to_100(sp_a)
                dow_r = rebase_to_100(dow_a)

                fig2 = make_subplots(specs=[[{"secondary_y": True}]])

                # 예측 확률(좌측 축)
                fig2.add_trace(
                    go.Scatter(
                        x=idx,
                        y=prob_5y['prob_pct'],
                        name="예측 상승 확률(%)",
                        line=dict(color='skyblue', width=2),
                        mode='lines'
                    ),
                    secondary_y=False
                )

                # 지수(우측 축, 리베이스 100)
                fig2.add_trace(
                    go.Scatter(
                        x=idx,
                        y=nas_r,
                        name=f"나스닥({nas_sym}) 리베이스(100)",
                        line=dict(color='#FF4B4B', width=1.5),
                        mode='lines'
                    ),
                    secondary_y=True
                )
                fig2.add_trace(
                    go.Scatter(
                        x=idx,
                        y=sp_r,
                        name=f"S&P500({sp_sym}) 리베이스(100)",
                        line=dict(color='#2ECC71', width=1.5),
                        mode='lines'
                    ),
                    secondary_y=True
                )
                fig2.add_trace(
                    go.Scatter(
                        x=idx,
                        y=dow_r,
                        name=f"다우({dow_sym}) 리베이스(100)",
                        line=dict(color='#F1C40F', width=1.5),
                        mode='lines'
                    ),
                    secondary_y=True
                )

                fig2.update_xaxes(title_text="날짜")
                fig2.update_yaxes(title_text="예측 상승 확률 (%)", secondary_y=False, range=[0, 100])
                fig2.update_yaxes(title_text="지수 리베이스 (첫값=100)", secondary_y=True)
                fig2.update_layout(
                    height=650,
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                    title="최근 5년 지수 움직임(리베이스)과 예측 확률 비교"
                )

                st.plotly_chart(fig2, use_container_width=True)

                # HTML 저장(리포트)용 figure 저장 (json)
                try:
                    st.session_state["report_fig_index_compare_json"] = fig2.to_json()
                except Exception:
                    pass

                with st.expander("ℹ️ 계산 방식 / 주의사항"):
                    st.write(
                        "- 지수는 날짜별 스케일 차이를 없애기 위해 **첫 값=100으로 리베이스**해서 표시합니다.\n"
                        "- 예측 확률은 모델이 학습된 feature(X) 기준으로 산출된 값입니다.\n"
                        "- FinanceDataReader 심볼은 환경에 따라 다를 수 있어 후보를 여러 개 시도합니다."
                    )
        except Exception as e:
            st.warning(f"최근 5년 비교 그래프 생성 중 오류: {str(e)}")
        
        # 주요 지표 표시
        st.subheader("📊 주요 기술적 지표")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'rsi' in spy.columns:
                current_rsi = spy['rsi'].iloc[-1]
                st.metric("RSI", f"{current_rsi:.2f}")
        
        with col2:
            if 'macd' in spy.columns:
                current_macd = spy['macd'].iloc[-1]
                st.metric("MACD", f"{current_macd:.2f}")
        
        with col3:
            if 'vix' in spy.columns:
                current_vix = spy['vix'].iloc[-1]
                st.metric("VIX", f"{current_vix:.2f}")
        
        with col4:
            if 'ratio_sma50' in spy.columns:
                current_ratio = spy['ratio_sma50'].iloc[-1]
                st.metric("SMA50 비율", f"{current_ratio*100:.2f}%")
        
        # 최근 확률 데이터 테이블
        with st.expander("📋 최근 확률 데이터 보기"):
            if prob_history is not None:
                recent_data = prob_history.tail(30).copy()
                recent_data['Probability'] = recent_data['Probability'] * 100
                recent_data = recent_data.rename(columns={'Probability': '상승 확률 (%)'})
                st.dataframe(recent_data.style.format({'상승 확률 (%)': '{:.2f}'}), use_container_width=True)
        
        # 모델 정보
        st.subheader("🤖 모델 정보")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**모델 타입**: RandomForest Classifier\n\n"
                   f"**특성 수**: {len(predictor.feature_columns)}\n\n"
                   f"**트리 수**: {predictor.model.n_estimators}")
        
        with col2:
            if hasattr(predictor.model, 'oob_score_'):
                st.info(f"**OOB 정확도**: {predictor.model.oob_score_:.3f}\n\n"
                       f"**최대 깊이**: {predictor.model.max_depth}\n\n"
                       f"**최소 샘플 리프**: {predictor.model.min_samples_leaf}")

        # =========================
        # 하락 확률: 탭 없이 하단에 표기
        # =========================
        st.markdown("---")
        st.subheader("📉 하락 확률(모델 새로 생성)")
        st.caption("하락 모델은 여기서 새로 학습합니다. (SMA는 세션/업로드 데이터를 재사용하여 다시 계산하지 않습니다.)")

        sma_dataframes = st.session_state.get("sma_dataframes")
        if not (isinstance(sma_dataframes, dict) and sma_dataframes):
            st.warning("SMA 데이터가 없습니다. SMA zip/패키지를 업로드하거나, 먼저 SMA 수집을 진행하세요.")
        else:
            today_str = datetime.now().date().isoformat()
            sma_sig = (
                st.session_state.get("_sma_upload_sig")
                or ("sma_cache", st.session_state.get("sma_collector_date"), len(sma_dataframes))
            )
            cached_ok = (
                st.session_state.get("down_model_date") == today_str
                and st.session_state.get("down_model_sma_sig") == sma_sig
                and st.session_state.get("down_model_bytes") is not None
            )

            train_down_clicked = st.button("📉 하락 모델 학습 후 예측", type="primary", key="train_down_button")

            if train_down_clicked:
                with st.status("🔄 하락 모델 학습 중...", expanded=True) as status:
                    try:
                        st.write("1/3: 학습 데이터 준비(IXIC + 피처)")

                        collector_down = StockDataCollector(
                            start_date="2015-01-01",
                            end_date=datetime.now().date() + timedelta(days=1),
                        )
                        collector_down.dataframes = sma_dataframes.copy()  # SMA 재사용

                        spy_down = collector_down.prepare_target_data("IXIC", for_prediction=False)
                        spy_down = collector_down.add_features(spy_down, skip_sma=False, for_prediction=False)

                        X_down = build_feature_matrix(spy_down)
                        y_down = spy_down.get("TargetDown")
                        if y_down is None:
                            raise RuntimeError("TargetDown 컬럼이 없습니다. (stock_analysis_refactored.py 변경이 반영되지 않았을 수 있습니다.)")

                        mask = ~pd.isna(y_down)
                        X_down = X_down.loc[mask]
                        y_down = y_down.loc[mask]

                        # 상승 모델과 동일한 피처셋으로 학습(가능한 경우)
                        if getattr(predictor, "feature_columns", None):
                            X_down = X_down.reindex(columns=predictor.feature_columns).fillna(0)

                        st.write(f"2/3: 모델 학습 시작 (샘플 {len(X_down)}, 특성 {len(X_down.columns)})")

                        predictor_down = StockPredictor("stock_model_down.pkl")
                        predictor_down.train_model(
                            X_down,
                            y_down,
                            # 상승 모델과 동일 조건(트리 수 포함) 적용
                            n_estimators=2000,
                            progress_callback=None,
                            status_callback=None,
                        )

                        st.write("3/3: 모델 저장(세션)")
                        st.session_state["down_model_bytes"] = export_model_pkl_bytes(predictor_down)
                        st.session_state["down_model_date"] = today_str
                        st.session_state["down_model_sma_sig"] = sma_sig
                        status.update(label="✅ 하락 모델 학습 완료", state="complete")
                    except Exception as e:
                        status.update(label="❌ 하락 모델 학습 실패", state="error")
                        st.error(str(e))

            if cached_ok or (st.session_state.get("down_model_bytes") is not None):
                predictor_down_use = StockPredictor("stock_model_down.pkl")
                ok = load_model_from_pkl_bytes(predictor_down_use, st.session_state["down_model_bytes"])
                if not ok:
                    st.error("하락 모델을 로드하지 못했습니다.")
                else:
                    # feature mismatch 방지: Series로 전달(컬럼 align 가능)
                    down_prob = predictor_down_use.get_current_probability(X.iloc[-1])
                    if down_prob is None:
                        st.error("하락 확률 계산 실패")
                    else:
                        # HTML 저장(리포트)용 세션 상태 저장
                        try:
                            st.session_state["report_current_down_prob"] = float(down_prob)
                        except Exception:
                            st.session_state["report_current_down_prob"] = None

                        st.metric("현재 하락 확률", f"{down_prob*100:.2f}%")
                        st.caption(f"기준일: {str(spy.index[-1])[:10]}")

                        with st.expander("📉 하락 확률 상세 보기", expanded=False):
                            st.subheader("📅 최근 5일간 하락 확률")
                            down_hist_5 = predictor_down_use.get_probability_history(X, days=5)
                            if down_hist_5 is not None and len(down_hist_5) > 0:
                                tmp = down_hist_5.copy().sort_index()
                                # HTML 저장(리포트)용 최근 5일 하락 확률 저장
                                try:
                                    st.session_state["report_recent5_down"] = [
                                        {"date": str(idx)[:10], "prob": float(p)}
                                        for idx, p in tmp["Probability"].items()
                                        if p is not None and not pd.isna(p)
                                    ]
                                except Exception:
                                    pass
                                tmp["Probability"] = tmp["Probability"] * 100
                                st.dataframe(tmp.rename(columns={"Probability": "하락 확률 (%)"}), use_container_width=True)
                            else:
                                st.info("최근 5일 확률 데이터를 만들 수 없습니다.")

                            # =========================
                            # 그래프 2종
                            # 1) 확률 추이(확률 + IXIC)
                            # 2) 최근 5년 지수 비교(나스닥/다우/S&P) + 확률
                            # =========================

                            st.markdown("---")
                            st.subheader("📈 확률 추이 그래프 (하락 확률)")

                            prob_history_down = predictor_down_use.get_probability_history(X, days=min(500, len(X)))
                            if prob_history_down is not None and len(prob_history_down) > 0:
                                prob_history_down = prob_history_down.sort_index()
                                start_date_prob = prob_history_down.index[0]
                                end_date_prob = prob_history_down.index[-1]

                                try:
                                    price_data = fdr.DataReader('IXIC', start_date_prob, end_date_prob)
                                    price_aligned = price_data.reindex(prob_history_down.index, method='nearest')
                                except Exception:
                                    price_aligned = None

                                fig_down = make_subplots(specs=[[{"secondary_y": True}]])
                                fig_down.add_trace(
                                    go.Scatter(
                                        x=prob_history_down.index,
                                        y=prob_history_down['Probability'] * 100,
                                        name="하락 확률 (%)",
                                        line=dict(color='#FF4B4B', width=2),
                                        mode='lines+markers',
                                        marker=dict(size=4),
                                    ),
                                    secondary_y=False
                                )
                                fig_down.add_hline(
                                    y=50,
                                    line_dash="dash",
                                    line_color="gray",
                                    opacity=0.5,
                                    annotation_text="기준선 (50%)",
                                    secondary_y=False
                                )
                                if price_aligned is not None and len(price_aligned) > 0:
                                    col_price = 'Adj Close' if 'Adj Close' in price_aligned.columns else ('Close' if 'Close' in price_aligned.columns else None)
                                    if col_price is not None and not price_aligned[col_price].isna().all():
                                        fig_down.add_trace(
                                            go.Scatter(
                                                x=prob_history_down.index,
                                                y=price_aligned[col_price],
                                                name="IXIC 가격",
                                                line=dict(color='red', width=1, dash='dot'),
                                                opacity=0.5
                                            ),
                                            secondary_y=True
                                        )

                                fig_down.update_xaxes(title_text="날짜")
                                fig_down.update_yaxes(title_text="하락 확률 (%)", secondary_y=False, range=[0, 100])
                                fig_down.update_yaxes(title_text="주가 (USD)", secondary_y=True)
                                fig_down.update_layout(
                                    title="하락 확률 추이 및 IXIC 가격",
                                    height=600,
                                    hovermode='x unified',
                                )
                                st.plotly_chart(fig_down, use_container_width=True)
                                # HTML 저장(리포트)용 figure 저장 (json)
                                try:
                                    st.session_state["report_fig_down_trend_json"] = fig_down.to_json()
                                except Exception:
                                    pass
                            else:
                                st.info("확률 히스토리가 비어있어 그래프를 그릴 수 없습니다.")

                            st.subheader("📊 최근 5년: 나스닥 / S&P500 / 다우존스 vs 하락 확률")
                            try:
                                prob_all = predictor_down_use.get_probability_history(X, days=len(X))
                                if prob_all is None or len(prob_all) == 0:
                                    st.info("확률 히스토리를 계산할 수 없습니다.")
                                else:
                                    prob_all = prob_all.sort_index()
                                    end_dt = prob_all.index.max()
                                    start_dt = end_dt - pd.DateOffset(years=5)
                                    prob_5y = prob_all.loc[prob_all.index >= start_dt].copy()
                                    prob_5y['prob_pct'] = prob_5y['Probability'] * 100.0

                                    start_fetch = start_dt.date()
                                    end_fetch = (end_dt.date() + timedelta(days=1))

                                    nas_sym, nas = fetch_index_adj_close(['IXIC', '^IXIC'], start_fetch, end_fetch)
                                    sp_sym, sp = fetch_index_adj_close(['US500', 'SPX', '^GSPC'], start_fetch, end_fetch)
                                    dow_sym, dow = fetch_index_adj_close(['DJI', '^DJI'], start_fetch, end_fetch)

                                    idx = prob_5y.index
                                    nas_a = nas.reindex(idx, method='ffill')
                                    sp_a = sp.reindex(idx, method='ffill')
                                    dow_a = dow.reindex(idx, method='ffill')

                                    nas_r = rebase_to_100(nas_a)
                                    sp_r = rebase_to_100(sp_a)
                                    dow_r = rebase_to_100(dow_a)

                                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                                    fig2.add_trace(
                                        go.Scatter(
                                            x=idx,
                                            y=prob_5y['prob_pct'],
                                            name="하락 확률(%)",
                                            line=dict(color='#FF4B4B', width=2),
                                            mode='lines'
                                        ),
                                        secondary_y=False
                                    )
                                    fig2.add_trace(
                                        go.Scatter(
                                            x=idx,
                                            y=nas_r,
                                            name=f"나스닥({nas_sym}) 리베이스(100)",
                                            line=dict(color='#0046ff', width=1.5),
                                            mode='lines'
                                        ),
                                        secondary_y=True
                                    )
                                    fig2.add_trace(
                                        go.Scatter(
                                            x=idx,
                                            y=sp_r,
                                            name=f"S&P500({sp_sym}) 리베이스(100)",
                                            line=dict(color='#2ECC71', width=1.5),
                                            mode='lines'
                                        ),
                                        secondary_y=True
                                    )
                                    fig2.add_trace(
                                        go.Scatter(
                                            x=idx,
                                            y=dow_r,
                                            name=f"다우({dow_sym}) 리베이스(100)",
                                            line=dict(color='#F1C40F', width=1.5),
                                            mode='lines'
                                        ),
                                        secondary_y=True
                                    )

                                    fig2.update_xaxes(title_text="날짜")
                                    fig2.update_yaxes(title_text="하락 확률 (%)", secondary_y=False, range=[0, 100])
                                    fig2.update_yaxes(title_text="지수 리베이스 (첫값=100)", secondary_y=True)
                                    fig2.update_layout(
                                        height=650,
                                        hovermode='x unified',
                                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                                        title="최근 5년 지수 움직임(리베이스)과 하락 확률 비교"
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                                    # HTML 저장(리포트)용 figure 저장 (json)
                                    try:
                                        st.session_state["report_fig_down_index_compare_json"] = fig2.to_json()
                                    except Exception:
                                        pass
                            except Exception as e:
                                st.warning(f"최근 5년 비교 그래프 생성 중 오류: {str(e)}")

        # =========================
        # 화면 분석 결과 HTML 저장
        # =========================
        st.markdown("---")
        st.subheader("💾 화면 분석 결과 저장")
        report_html = build_screen_analysis_report_html_from_session_state()
        if report_html is None:
            st.info("저장할 결과(확률/그래프)가 아직 없습니다. 먼저 예측을 실행하세요.")
        else:
            st.download_button(
                label="⬇️ 화면 분석 결과 HTML 저장(.html)",
                data=report_html.encode("utf-8"),
                file_name=f"screen_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                help="가능하면 그래프를 PNG 이미지로 포함해 저장합니다. (환경에 따라 interactive Plotly로 저장될 수 있습니다.)",
            )
        
    except Exception as e:
        st.error(f"❌ 데이터 처리 중 오류 발생: {str(e)}")
        st.exception(e)
        # 오류 발생 시에도 플래그 리셋 (매 실행마다 내리면 rerun 루프 발생 가능)
        if model_option == "기존 모델 사용" and st.session_state.get('run_prediction', False):
            st.session_state.run_prediction = False

else:
    st.warning("⚠️ 모델이 로드되지 않았습니다. 사이드바에서 모델을 학습하거나 로드하세요.")
    
    # 사용 방법 안내
    st.info("""
    ### 사용 방법:
    1. **기존 모델 사용**: 저장된 모델이 있으면 자동으로 로드됩니다.
    2. **새 모델 학습**: 
       - '새 모델 학습' 옵션 선택
       - '실시간 데이터 수집' 체크 (선택사항)
       - '모델 학습 시작' 버튼 클릭
       - 학습이 완료되면 자동으로 모델이 저장됩니다.
    
    **참고**: 모델 학습은 시간이 오래 걸릴 수 있습니다 (수십 분 소요 가능).
    """)

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>📈 주가 예측 확률 분석 대시보드</p>
        <p>⚠️ 이 예측은 참고용이며, 실제 투자 결정에 사용하기 전에 전문가의 조언을 구하세요.</p>
    </div>
    """,
    unsafe_allow_html=True
)
