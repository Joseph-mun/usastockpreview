# -*- coding: utf-8 -*-
"""Backtest: prediction accuracy + TQQQ/SPY portfolio simulation."""

import sys
import os
import base64
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    INDEX_CONFIGS, TARGET_LOOKAHEAD_DAYS, SIGNAL_THRESHOLDS, LGBM_PARAMS,
    DECISION_THRESHOLD, MIN_TRAINING_SAMPLES,
    get_logger,
)

logger = get_logger(__name__)
from src.data import get_close_col
from src.data.collectors import SMACollector, get_spy_data
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.strategy.allocation import get_allocation
from src.strategy.portfolio_backtest import (
    run_portfolio_backtest, compute_backtest_metrics, compute_benchmark_returns,
)
from src.templates import load_template


def run_backtest(backtest_days: int = 252, retrain_freq: int = 60, verbose: bool = True):
    """Expanding walk-forward backtest with periodic retraining.

    Args:
        backtest_days: total evaluation period in trading days (default 252 = 1 year)
        retrain_freq: retrain every N trading days (default 60 = ~3 months)
    """
    def log(msg):
        if verbose:
            logger.info(msg)

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
    else:
        log("   WARNING: SMA 캐시 없음")

    # 2. Build full dataset
    log("2. 데이터셋 구축...")
    builder = DatasetBuilder(sma_ratios=sma_ratios)
    X, spy, y = builder.build(ticker, for_prediction=False)
    log(f"   데이터셋: {len(X)} samples, {len(X.columns)} features")
    log(f"   기간: {spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')}")
    log(f"   Target=1 비율: {y.mean():.3f}")

    close_col = get_close_col(spy)

    # 3. Expanding Walk-Forward
    eval_days = min(backtest_days, len(X) - MIN_TRAINING_SAMPLES)
    wf_origin = len(X) - eval_days  # first train/test split point

    from src.models.trainer import ModelTrainer

    all_predictions = []
    n_windows = 0

    log(f"3. Expanding Walk-Forward ({eval_days}일, retrain 주기={retrain_freq}일)...")

    for wf_start in range(wf_origin, len(X), retrain_freq):
        wf_end = min(wf_start + retrain_freq, len(X))
        X_train_wf = X.iloc[:wf_start]
        y_train_wf = y.iloc[:wf_start]
        X_test_wf = X.iloc[wf_start:wf_end]
        y_test_wf = y.iloc[wf_start:wf_end]

        if len(X_train_wf) < MIN_TRAINING_SAMPLES or len(X_test_wf) == 0:
            continue

        n_windows += 1
        log(f"\n   --- Window {n_windows}: train={len(X_train_wf)}, test={len(X_test_wf)} ---")

        trainer = ModelTrainer(index_name)
        result = trainer.train(
            X_train_wf, y_train_wf,
            status_callback=lambda msg: log(f"   {msg}") if verbose else None,
        )

        feature_cols = trainer.feature_columns
        calibrator = trainer.calibrator
        cal_method = trainer.calibration_method
        model = result["model"]

        X_test_aligned = X_test_wf.reindex(columns=feature_cols).fillna(0)
        raw_probs = model.predict_proba(X_test_aligned.values)[:, 1]

        if calibrator is not None:
            if cal_method == "isotonic":
                cal_probs = calibrator.predict(raw_probs)
            else:
                cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            cal_probs = raw_probs

        window_correct = 0
        for i, date in enumerate(X_test_wf.index):
            prob = float(cal_probs[i])
            actual = int(y_test_wf.iloc[i])
            predicted = 1 if prob >= DECISION_THRESHOLD else 0
            if predicted == actual:
                window_correct += 1
            all_predictions.append({
                "date": date,
                "prob": prob,
                "raw_prob": float(raw_probs[i]),
                "actual": actual,
                "predicted": predicted,
                "window": n_windows,
            })

        window_acc = window_correct / len(X_test_wf)
        log(f"   Window {n_windows} accuracy: {window_acc:.1%} ({window_correct}/{len(X_test_wf)})")

    log(f"\n4. 결과 집계 ({len(all_predictions)}일, {n_windows}개 윈도우)...")

    # Build results table from all_predictions
    spy_test = spy.loc[[p["date"] for p in all_predictions]]
    prob_series = pd.Series(
        [p["prob"] for p in all_predictions],
        index=[p["date"] for p in all_predictions],
        name="Probability",
    )

    results = []
    for p in all_predictions:
        date = p["date"]
        prob = p["prob"]

        if prob >= SIGNAL_THRESHOLDS["strong_buy"]:
            signal = "Strong Buy"
        elif prob >= SIGNAL_THRESHOLDS["buy"]:
            signal = "Buy"
        elif prob >= SIGNAL_THRESHOLDS["neutral"]:
            signal = "Neutral"
        else:
            signal = "Sell"

        price = float(spy.loc[date, close_col])
        after_price = float(spy.loc[date, "after"]) if "after" in spy.columns and not pd.isna(spy.loc[date, "after"]) else None
        return_pct = float(spy.loc[date, "suik_rate"]) if "suik_rate" in spy.columns and not pd.isna(spy.loc[date, "suik_rate"]) else None
        alloc = get_allocation(prob)

        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "weekday": ["월", "화", "수", "목", "금", "토", "일"][date.weekday()],
            "price": price,
            "after_price": after_price,
            "return_pct": return_pct,
            "probability": prob,
            "raw_prob": p.get("raw_prob", prob),
            "signal": signal,
            "predicted": p["predicted"],
            "actual": p["actual"],
            "correct": p["predicted"] == p["actual"],
            "tier": alloc.tier_label,
        })

    df_results = pd.DataFrame(results)

    # 5. Metrics
    total = len(df_results)
    correct_count = int(df_results["correct"].sum())
    accuracy = correct_count / total if total > 0 else 0
    tp = len(df_results[(df_results["predicted"] == 1) & (df_results["actual"] == 1)])
    fp = len(df_results[(df_results["predicted"] == 1) & (df_results["actual"] == 0)])
    tn = len(df_results[(df_results["predicted"] == 0) & (df_results["actual"] == 0)])
    fn = len(df_results[(df_results["predicted"] == 0) & (df_results["actual"] == 1)])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    signal_stats = {}
    for sig in ["Strong Buy", "Buy", "Neutral", "Sell"]:
        subset = df_results[df_results["signal"] == sig]
        if len(subset) > 0:
            signal_stats[sig] = {
                "count": len(subset),
                "avg_prob": float(subset["probability"].mean()),
                "actual_up_rate": float(subset["actual"].mean()),
                "avg_return": float(subset["return_pct"].mean()) if subset["return_pct"].notna().any() else 0,
            }

    # 6. Portfolio simulation
    log("\n5. 포트폴리오 시뮬레이션...")
    nasdaq_prices = spy_test[close_col]
    try:
        spy_data = get_spy_data()
        spy_prices = spy_data[get_close_col(spy_data)]
    except (RuntimeError, ValueError, KeyError) as e:
        log(f"   SPY 데이터 실패: {e}")
        spy_prices = nasdaq_prices * 0.4

    # Extract VIX/ADX for portfolio simulation risk filters
    vix_series = spy["vix"].reindex(prob_series.index).ffill() if "vix" in spy.columns else None
    adx_series = spy["adx"].reindex(prob_series.index).ffill() if "adx" in spy.columns else None

    port_df = run_portfolio_backtest(
        prob_series, nasdaq_prices, spy_prices,
        vix_series=vix_series, adx_series=adx_series,
    )
    port_metrics = compute_backtest_metrics(port_df) if not port_df.empty else {}
    benchmarks = compute_benchmark_returns(nasdaq_prices, spy_prices, prob_series.index) if not port_df.empty else {}

    if port_metrics:
        log(f"   전략 수익률: {port_metrics['total_return']*100:+.2f}%")
        log(f"   MDD: {port_metrics['max_drawdown']*100:.2f}%")
        log(f"   Sharpe: {port_metrics['sharpe_ratio']:.2f}")

    metrics = {
        "total": total, "correct": correct_count, "accuracy": accuracy,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_windows": n_windows, "retrain_freq": retrain_freq,
        "signal_stats": signal_stats,
        "portfolio": port_metrics, "benchmarks": benchmarks,
    }

    log(f"\n   전체 정확도: {accuracy:.1%} ({correct_count}/{total})")
    log(f"   평가 윈도우: {n_windows}개, 재학습 주기: {retrain_freq}일")
    return df_results, metrics, spy_test, port_df


def _make_portfolio_chart(port_df, benchmarks, initial=10000):
    """Generate portfolio equity curve as base64 PNG."""
    if port_df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")
    dates = pd.to_datetime(port_df["date"])
    ax.plot(dates, port_df["portfolio_value"], label="Strategy", color="#4fc3f7", linewidth=2)
    for bm_name, bm in benchmarks.items():
        style = "--" if "TQQQ" not in bm_name else ":"
        ax.axhline(y=bm["final_value"], color="#888" if "TQQQ" not in bm_name else "#ef5350", linestyle=style, alpha=0.5, label=f"{bm_name} ({bm['total_return']*100:+.1f}%)")
    ax.axhline(y=initial, color="#666", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_ylabel("Portfolio ($)", color="#aaa", fontsize=10)
    ax.tick_params(colors="#888")
    ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#333", labelcolor="#ccc")
    ax.grid(True, alpha=0.1)
    for s in ax.spines.values():
        s.set_color("#333")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _css_class(value, good=0.6, warn=0.5):
    """Return CSS class based on metric value thresholds."""
    if value >= good:
        return "good"
    return "warn" if value >= warn else "bad"


def generate_html_report(df_results, metrics, spy_test, port_df, output_path):
    """Generate enhanced HTML backtest report using template."""
    total = metrics["total"]
    accuracy = metrics["accuracy"]
    precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    signal_stats = metrics["signal_stats"]
    port_metrics = metrics.get("portfolio", {})
    benchmarks = metrics.get("benchmarks", {})
    date_from, date_to = df_results["date"].iloc[0], df_results["date"].iloc[-1]
    lookahead = TARGET_LOOKAHEAD_DAYS

    daily_rows = ""
    for _, r in df_results.iterrows():
        ret_str = f"{r['return_pct']:+.2f}%" if r['return_pct'] is not None and not pd.isna(r['return_pct']) else "-"
        after_str = f"{r['after_price']:,.2f}" if r["after_price"] is not None and not pd.isna(r['after_price']) else "-"
        cc = "correct" if r["correct"] else "wrong"
        daily_rows += f'<tr class="{cc}"><td>{r["date"]} ({r["weekday"]})</td><td class="num">{r["price"]:,.2f}</td><td class="num">{after_str}</td><td class="num">{ret_str}</td><td class="prob"><span class="signal-badge {r["signal"].lower().replace(" ","-")}">{r["probability"]*100:.1f}%</span></td><td>{r["signal"]}</td><td>{"상승" if r["actual"]==1 else "하락"}</td><td class="result-{cc}">{"O" if r["correct"] else "X"}</td><td><span class="tier-badge {r["tier"].lower()}">{r["tier"]}</span></td></tr>'

    signal_rows = ""
    for sn in ["Strong Buy", "Buy", "Neutral", "Sell"]:
        if sn in signal_stats:
            s = signal_stats[sn]
            signal_rows += f'<tr><td><span class="signal-badge {sn.lower().replace(" ","-")}">{sn}</span></td><td class="num">{s["count"]}일</td><td class="num">{s["avg_prob"]*100:.1f}%</td><td class="num">{s["actual_up_rate"]*100:.1f}%</td><td class="num">{s["avg_return"]:+.2f}%</td></tr>'

    port_section = _build_portfolio_section(port_metrics, benchmarks, port_df)

    template = load_template("backtest_report.html")
    html = template.safe_substitute(
        subtitle=f"{date_from} ~ {date_to} | {total}거래일 | Target: {lookahead}일 후 상승 여부 | Calibrated + Feature Selection | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        accuracy_class=_css_class(accuracy),
        accuracy_pct=f"{accuracy:.1%}",
        correct=metrics["correct"],
        total=total,
        precision_class=_css_class(precision),
        precision_pct=f"{precision:.1%}",
        recall_class=_css_class(recall),
        recall_pct=f"{recall:.1%}",
        f1_class=_css_class(f1),
        f1_pct=f"{f1:.1%}",
        tp=tp, fp=fp, fn=fn, tn=tn,
        signal_rows=signal_rows,
        port_section=port_section,
        daily_rows=daily_rows,
        lookahead=lookahead,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def _build_portfolio_section(port_metrics, benchmarks, port_df):
    """Build portfolio simulation HTML section."""
    if not port_metrics:
        return ""

    chart_b64 = _make_portfolio_chart(port_df, benchmarks)
    bm_rows = "".join(
        f'<tr><td>{n}</td><td class="num">${b["final_value"]:,.0f}</td>'
        f'<td class="num">{b["total_return"]*100:+.2f}%</td></tr>'
        for n, b in benchmarks.items()
    )
    tier_dist = port_metrics.get("tier_distribution", {})
    tier_colors = {"Aggressive": "#ef5350", "Growth": "#4fc3f7", "Moderate": "#ffa726", "Cautious": "#888", "Defensive": "#666"}
    tier_bars = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<span style="width:80px;font-size:12px;color:#aaa">{t}</span>'
        f'<div style="background:{tier_colors.get(t,"#888")};height:16px;width:{p*200}px;border-radius:3px"></div>'
        f'<span style="font-size:12px;color:#888">{p*100:.0f}%</span></div>'
        for t, p in sorted(tier_dist.items(), key=lambda x: x[1], reverse=True)
    )

    ret_class = "good" if port_metrics.get("total_return", 0) > 0 else "bad"
    sharpe_class = "good" if port_metrics.get("sharpe_ratio", 0) > 1 else "warn"
    chart_img = f"<img src='data:image/png;base64,{chart_b64}' style='width:100%;border-radius:8px;margin-bottom:16px'>" if chart_b64 else ""

    return (
        f'<div class="section"><h2>Portfolio Simulation (TQQQ/SPY Strategy)</h2>'
        f'<div class="metrics-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:16px">'
        f'<div class="metric-card"><div class="metric-value {ret_class}">{port_metrics.get("total_return",0)*100:+.2f}%</div><div class="metric-label">전략 수익률</div></div>'
        f'<div class="metric-card"><div class="metric-value bad">{port_metrics.get("max_drawdown",0)*100:.2f}%</div><div class="metric-label">MDD</div></div>'
        f'<div class="metric-card"><div class="metric-value {sharpe_class}">{port_metrics.get("sharpe_ratio",0):.2f}</div><div class="metric-label">Sharpe</div></div>'
        f'<div class="metric-card"><div class="metric-value">{port_metrics.get("n_rebalances",0)}</div><div class="metric-label">리밸런싱</div></div></div>'
        f'{chart_img}'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        f'<div><h3 style="font-size:14px;color:#888;margin-bottom:8px">vs Benchmark</h3>'
        f'<table><thead><tr><th>Name</th><th style="text-align:right">최종가치</th><th style="text-align:right">수익률</th></tr></thead>'
        f'<tbody><tr><td><b>Strategy</b></td><td class="num"><b>${port_metrics.get("final_value",10000):,.0f}</b></td>'
        f'<td class="num"><b>{port_metrics.get("total_return",0)*100:+.2f}%</b></td></tr>{bm_rows}</tbody></table></div>'
        f'<div><h3 style="font-size:14px;color:#888;margin-bottom:8px">Tier 분포</h3>{tier_bars}</div></div></div>'
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NASDAQ Backtest")
    parser.add_argument("--days", type=int, default=252, help="Evaluation period in trading days")
    parser.add_argument("--retrain-freq", type=int, default=60, help="Retrain every N trading days")
    args = parser.parse_args()
    logger.info("=" * 60)
    logger.info("  NASDAQ Backtest - %d거래일 Expanding WF (retrain=%d일)", args.days, args.retrain_freq)
    logger.info("=" * 60)
    df_results, metrics, spy_test, port_df = run_backtest(
        backtest_days=args.days, retrain_freq=args.retrain_freq,
    )
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.html")
    generate_html_report(df_results, metrics, spy_test, port_df, output_path)
    logger.info("HTML 리포트 생성 완료: %s", output_path)


if __name__ == "__main__":
    main()
