# -*- coding: utf-8 -*-
"""Backtest: compare model predictions vs actual outcomes for last 30 trading days."""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INDEX_CONFIGS, TARGET_LOOKAHEAD_DAYS, TARGET_UP_THRESHOLD, SIGNAL_THRESHOLDS
from src.data.collectors import SMACollector
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.models.trainer import ModelTrainer


def run_backtest(backtest_days: int = 30, verbose: bool = True):
    """
    Backtest the NASDAQ model over the last N trading days.

    Trains a fresh model locally (skipping pickle compatibility issues),
    then evaluates predictions against actual outcomes.
    """
    def log(msg):
        if verbose:
            print(msg)

    index_name = "NASDAQ"
    cfg = INDEX_CONFIGS[index_name]
    ticker = cfg["ticker"]

    # 1. Load SMA cache
    log("1. SMA 캐시 로드...")
    cache = SMACache()
    raw_sma, sma_meta = cache.load()
    sma_ratios = {}
    if raw_sma:
        sma_collector = SMACollector()
        sma_collector.raw_dataframes = raw_sma
        sma_ratios = sma_collector.compute_ratios()
        log(f"   SMA 캐시 로드 완료")
    else:
        log("   WARNING: SMA 캐시 없음 - SMA 비율은 기본값(0.5) 사용")

    # 2. Build full dataset (with targets for verification)
    log("2. 데이터셋 구축...")
    builder = DatasetBuilder(sma_ratios=sma_ratios)
    X, spy, y = builder.build(ticker, for_prediction=False)
    log(f"   데이터셋: {len(X)} samples, {len(X.columns)} features")
    log(f"   기간: {spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')}")
    log(f"   Target=1 비율: {y.mean():.3f}")

    # 3. Walk-forward backtest: train on data BEFORE test period, predict on test period
    n_days = min(backtest_days, len(X) - 100)  # ensure enough training data
    split_idx = len(X) - n_days

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    spy_test = spy.loc[X_test.index]

    log(f"3. Walk-forward 백테스트...")
    log(f"   학습 데이터: {len(X_train)} samples ({X_train.index[0].strftime('%Y-%m-%d')} ~ {X_train.index[-1].strftime('%Y-%m-%d')})")
    log(f"   테스트 데이터: {len(X_test)} samples ({X_test.index[0].strftime('%Y-%m-%d')} ~ {X_test.index[-1].strftime('%Y-%m-%d')})")

    # Train ONLY on data before the test period
    import lightgbm as lgb
    from src.config import LGBM_PARAMS
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train.values, y_train.values)
    log(f"   학습 완료 (테스트 기간 데이터는 학습에서 제외)")

    # Predict on held-out test period
    log(f"\n4. 최근 {len(X_test)}일 예측...")
    proba = model.predict_proba(X_test.values)[:, 1]
    prob_df = pd.DataFrame({"Probability": proba}, index=X_test.index)

    # Build results table
    close_col = "Adj Close" if "Adj Close" in spy_test.columns else "Close"
    results = []

    for i, date in enumerate(X_test.index):
        actual_target = int(y_test.iloc[i])
        prob = float(prob_df.loc[date, "Probability"]) if date in prob_df.index else None

        if prob is None:
            continue

        # Signal
        if prob >= SIGNAL_THRESHOLDS["strong_buy"]:
            signal = "Strong Buy"
        elif prob >= SIGNAL_THRESHOLDS["buy"]:
            signal = "Buy"
        elif prob >= SIGNAL_THRESHOLDS["neutral"]:
            signal = "Neutral"
        else:
            signal = "Sell"

        # Predicted outcome (>50% = predict up)
        predicted = 1 if prob >= 0.5 else 0
        correct = predicted == actual_target

        # Price info
        price = float(spy_test[close_col].iloc[i])
        after_price = float(spy_test["after"].iloc[i]) if "after" in spy_test.columns and not pd.isna(spy_test["after"].iloc[i]) else None
        return_pct = float(spy_test["suik_rate"].iloc[i]) if "suik_rate" in spy_test.columns and not pd.isna(spy_test["suik_rate"].iloc[i]) else None

        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "weekday": ["월", "화", "수", "목", "금", "토", "일"][date.weekday()],
            "price": price,
            "after_price": after_price,
            "return_pct": return_pct,
            "probability": prob,
            "signal": signal,
            "predicted": predicted,
            "actual": actual_target,
            "correct": correct,
        })

    df_results = pd.DataFrame(results)

    # 5. Compute metrics
    log("\n5. 결과 분석...")
    total = len(df_results)
    correct_count = df_results["correct"].sum()
    accuracy = correct_count / total if total > 0 else 0

    # Signal-level accuracy
    tp = len(df_results[(df_results["predicted"] == 1) & (df_results["actual"] == 1)])
    fp = len(df_results[(df_results["predicted"] == 1) & (df_results["actual"] == 0)])
    tn = len(df_results[(df_results["predicted"] == 0) & (df_results["actual"] == 0)])
    fn = len(df_results[(df_results["predicted"] == 0) & (df_results["actual"] == 1)])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # By signal category
    signal_stats = {}
    for sig in ["Strong Buy", "Buy", "Neutral", "Sell"]:
        subset = df_results[df_results["signal"] == sig]
        if len(subset) > 0:
            sig_correct = subset["actual"].sum() if sig in ["Strong Buy", "Buy"] else (1 - subset["actual"]).sum()
            signal_stats[sig] = {
                "count": len(subset),
                "avg_prob": subset["probability"].mean(),
                "actual_up_rate": subset["actual"].mean(),
                "avg_return": subset["return_pct"].mean() if subset["return_pct"].notna().any() else 0,
            }

    metrics = {
        "total": total,
        "correct": int(correct_count),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "signal_stats": signal_stats,
    }

    log(f"   전체 정확도: {accuracy:.1%} ({correct_count}/{total})")
    log(f"   Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.1%}")

    return df_results, metrics, spy_test


def generate_html_report(df_results: pd.DataFrame, metrics: dict, spy_test: pd.DataFrame, output_path: str):
    """Generate a clean HTML backtest report."""

    total = metrics["total"]
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    signal_stats = metrics["signal_stats"]

    # Date range
    date_from = df_results["date"].iloc[0]
    date_to = df_results["date"].iloc[-1]

    # Build daily rows
    daily_rows = ""
    for _, row in df_results.iterrows():
        prob_pct = f"{row['probability']*100:.1f}%"
        ret_str = f"{row['return_pct']:+.2f}%" if row['return_pct'] is not None and not pd.isna(row['return_pct']) else "-"
        actual_str = "상승" if row["actual"] == 1 else "미달"
        correct_cls = "correct" if row["correct"] else "wrong"
        correct_str = "O" if row["correct"] else "X"

        signal_cls = row["signal"].lower().replace(" ", "-")

        price_str = f"{row['price']:,.2f}"
        after_str = f"{row['after_price']:,.2f}" if row["after_price"] is not None and not pd.isna(row['after_price']) else "-"

        daily_rows += f"""
        <tr class="{correct_cls}">
            <td>{row['date']} ({row['weekday']})</td>
            <td class="num">{price_str}</td>
            <td class="num">{after_str}</td>
            <td class="num">{ret_str}</td>
            <td class="prob"><span class="signal-badge {signal_cls}">{prob_pct}</span></td>
            <td>{row['signal']}</td>
            <td>{actual_str}</td>
            <td class="result-{correct_cls}">{correct_str}</td>
        </tr>"""

    # Signal stats rows
    signal_rows = ""
    for sig_name in ["Strong Buy", "Buy", "Neutral", "Sell"]:
        if sig_name in signal_stats:
            s = signal_stats[sig_name]
            signal_rows += f"""
            <tr>
                <td><span class="signal-badge {sig_name.lower().replace(' ', '-')}">{sig_name}</span></td>
                <td class="num">{s['count']}일</td>
                <td class="num">{s['avg_prob']*100:.1f}%</td>
                <td class="num">{s['actual_up_rate']*100:.1f}%</td>
                <td class="num">{s['avg_return']:+.2f}%</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NASDAQ Backtest Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0f1117;
        color: #e0e0e0;
        padding: 24px;
        line-height: 1.6;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
        color: #fff;
    }}
    .subtitle {{
        color: #888;
        font-size: 14px;
        margin-bottom: 32px;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }}
    .metric-card {{
        background: #1a1d27;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2d3a;
    }}
    .metric-value {{
        font-size: 36px;
        font-weight: 700;
        color: #4fc3f7;
    }}
    .metric-value.good {{ color: #66bb6a; }}
    .metric-value.warn {{ color: #ffa726; }}
    .metric-value.bad {{ color: #ef5350; }}
    .metric-label {{
        font-size: 13px;
        color: #888;
        margin-top: 4px;
    }}
    .section {{
        background: #1a1d27;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid #2a2d3a;
    }}
    .section h2 {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 16px;
        color: #fff;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    th {{
        text-align: left;
        padding: 10px 12px;
        border-bottom: 2px solid #2a2d3a;
        color: #888;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    td {{
        padding: 10px 12px;
        border-bottom: 1px solid #1e2130;
    }}
    td.num {{ font-family: 'SF Mono', Monaco, monospace; text-align: right; }}
    td.prob {{ text-align: center; }}
    tr.correct {{ background: rgba(102, 187, 106, 0.05); }}
    tr.wrong {{ background: rgba(239, 83, 80, 0.05); }}
    .result-correct {{ color: #66bb6a; font-weight: 700; text-align: center; }}
    .result-wrong {{ color: #ef5350; font-weight: 700; text-align: center; }}
    .signal-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    .signal-badge.strong-buy {{ background: rgba(102,187,106,0.2); color: #66bb6a; }}
    .signal-badge.buy {{ background: rgba(79,195,247,0.2); color: #4fc3f7; }}
    .signal-badge.neutral {{ background: rgba(255,167,38,0.2); color: #ffa726; }}
    .signal-badge.sell {{ background: rgba(239,83,80,0.2); color: #ef5350; }}
    .confusion-grid {{
        display: grid;
        grid-template-columns: auto 1fr 1fr;
        gap: 0;
        max-width: 360px;
        margin: 0 auto;
    }}
    .confusion-cell {{
        padding: 16px;
        text-align: center;
        font-size: 14px;
    }}
    .confusion-header {{
        padding: 12px;
        text-align: center;
        font-size: 12px;
        color: #888;
        font-weight: 600;
    }}
    .confusion-tp {{ background: rgba(102,187,106,0.15); color: #66bb6a; font-weight: 700; font-size: 24px; border-radius: 8px 0 0 0; }}
    .confusion-fp {{ background: rgba(239,83,80,0.08); color: #ef5350; font-size: 24px; border-radius: 0 8px 0 0; }}
    .confusion-fn {{ background: rgba(239,83,80,0.08); color: #ef5350; font-size: 24px; border-radius: 0 0 0 8px; }}
    .confusion-tn {{ background: rgba(102,187,106,0.15); color: #66bb6a; font-weight: 700; font-size: 24px; border-radius: 0 0 8px 0; }}
    .confusion-label {{ font-size: 11px; color: #888; display: block; margin-top: 4px; font-weight: 400; }}
    .footer {{
        text-align: center;
        color: #555;
        font-size: 12px;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #1e2130;
    }}
    .bar-chart {{
        display: flex;
        align-items: end;
        gap: 3px;
        height: 60px;
        margin-top: 8px;
    }}
    .bar {{
        flex: 1;
        border-radius: 2px 2px 0 0;
        min-width: 4px;
        position: relative;
    }}
    .bar.up {{ background: #66bb6a; }}
    .bar.down {{ background: #ef5350; }}
</style>
</head>
<body>
<div class="container">
    <h1>NASDAQ Backtest Report</h1>
    <div class="subtitle">
        {date_from} ~ {date_to} | {total}거래일 | Target: 20일 후 +3% 상승 여부 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value {'good' if accuracy >= 0.6 else 'warn' if accuracy >= 0.5 else 'bad'}">{accuracy:.1%}</div>
            <div class="metric-label">전체 정확도 ({metrics['correct']}/{total})</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {'good' if precision >= 0.6 else 'warn' if precision >= 0.5 else 'bad'}">{precision:.1%}</div>
            <div class="metric-label">Precision (정밀도)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {'good' if recall >= 0.6 else 'warn' if recall >= 0.5 else 'bad'}">{recall:.1%}</div>
            <div class="metric-label">Recall (재현율)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {'good' if f1 >= 0.6 else 'warn' if f1 >= 0.5 else 'bad'}">{f1:.1%}</div>
            <div class="metric-label">F1 Score</div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px;">
        <div class="section">
            <h2>Confusion Matrix</h2>
            <div class="confusion-grid">
                <div class="confusion-header"></div>
                <div class="confusion-header">실제 상승</div>
                <div class="confusion-header">실제 미달</div>
                <div class="confusion-header" style="text-align:right; padding-right:16px;">예측 상승</div>
                <div class="confusion-cell confusion-tp">{tp}<span class="confusion-label">True Positive</span></div>
                <div class="confusion-cell confusion-fp">{fp}<span class="confusion-label">False Positive</span></div>
                <div class="confusion-header" style="text-align:right; padding-right:16px;">예측 하락</div>
                <div class="confusion-cell confusion-fn">{fn}<span class="confusion-label">False Negative</span></div>
                <div class="confusion-cell confusion-tn">{tn}<span class="confusion-label">True Negative</span></div>
            </div>
        </div>

        <div class="section">
            <h2>Signal Category Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Signal</th>
                        <th style="text-align:right">빈도</th>
                        <th style="text-align:right">평균 확률</th>
                        <th style="text-align:right">실제 상승률</th>
                        <th style="text-align:right">평균 수익률</th>
                    </tr>
                </thead>
                <tbody>
                    {signal_rows}
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>Daily Predictions vs Actuals</h2>
        <table>
            <thead>
                <tr>
                    <th>날짜</th>
                    <th style="text-align:right">종가</th>
                    <th style="text-align:right">20일 후</th>
                    <th style="text-align:right">수익률</th>
                    <th style="text-align:center">예측 확률</th>
                    <th>Signal</th>
                    <th>실제</th>
                    <th style="text-align:center">적중</th>
                </tr>
            </thead>
            <tbody>
                {daily_rows}
            </tbody>
        </table>
    </div>

    <div class="footer">
        NASDAQ Backtest Report | Model: LightGBM + TimeSeriesSplit | Target: 20-day +3% | Generated by us-market-predictor
    </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NASDAQ Backtest")
    parser.add_argument("--days", type=int, default=30, help="Number of trading days to backtest")
    args = parser.parse_args()

    days = args.days
    print("=" * 60)
    print(f"  NASDAQ Backtest - 최근 {days}거래일 예측 검증")
    print("=" * 60)

    df_results, metrics, spy_test = run_backtest(backtest_days=days)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "backtest_report.html"
    )
    generate_html_report(df_results, metrics, spy_test, output_path)

    print(f"\nHTML 리포트 생성 완료: {output_path}")


if __name__ == "__main__":
    main()
