# -*- coding: utf-8 -*-
"""Diagnose signal inversion: why high-confidence predictions are inverted.

Phase 4 C-01: Root cause identification for Strong Buy actual up 16.7% anomaly.
Hypotheses: H1 (Momentum-Target Mismatch), H2 (Calibration Distortion),
            H3 (Small Sample), H4 (Regime-dependent overfitting).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import (
    INDEX_CONFIGS, TARGET_LOOKAHEAD_DAYS, TARGET_ROLLING_MEDIAN_WINDOW,
    SIGNAL_THRESHOLDS,
)
from src.data.collectors import SMACollector, get_spy_data
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.models.trainer import ModelTrainer


def main():
    index_name = "NASDAQ"
    cfg = INDEX_CONFIGS[index_name]
    ticker = cfg["ticker"]

    # 1. Load data (same as backtest.py)
    print("1. Loading dataset...")
    cache = SMACache()
    raw_sma, sma_meta = cache.load()
    sma_ratios = {}
    if raw_sma:
        sma_collector = SMACollector()
        sma_collector.raw_dataframes = raw_sma
        sma_ratios = sma_collector.compute_ratios()

    builder = DatasetBuilder(sma_ratios=sma_ratios)
    X, spy, y = builder.build(ticker, for_prediction=False)
    close_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
    close = spy[close_col]

    # Recompute rolling_median (same as build_target)
    past_ret = close.pct_change(TARGET_LOOKAHEAD_DAYS)
    rolling_med = past_ret.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
    future_ret = close.shift(-TARGET_LOOKAHEAD_DAYS) / close - 1

    print(f"   Dataset: {len(X)} samples, {len(X.columns)} features")

    # 2. Walk-Forward loop (replicate backtest.py)
    backtest_days = 252
    retrain_freq = 60
    eval_days = min(backtest_days, len(X) - 500)
    wf_origin = len(X) - eval_days

    records = []
    n_windows = 0

    print(f"2. Walk-Forward ({eval_days} days, retrain={retrain_freq})...")

    for wf_start in range(wf_origin, len(X), retrain_freq):
        wf_end = min(wf_start + retrain_freq, len(X))
        X_train_wf = X.iloc[:wf_start]
        y_train_wf = y.iloc[:wf_start]
        X_test_wf = X.iloc[wf_start:wf_end]
        y_test_wf = y.iloc[wf_start:wf_end]

        if len(X_train_wf) < 500 or len(X_test_wf) == 0:
            continue

        n_windows += 1
        trainer = ModelTrainer(index_name)
        result = trainer.train(X_train_wf, y_train_wf)

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

        for i, date in enumerate(X_test_wf.index):
            raw_p = float(raw_probs[i])
            cal_p = float(cal_probs[i])
            actual = int(y_test_wf.iloc[i])

            if cal_p >= SIGNAL_THRESHOLDS["strong_buy"]:
                signal = "Strong Buy"
            elif cal_p >= SIGNAL_THRESHOLDS["buy"]:
                signal = "Buy"
            elif cal_p >= SIGNAL_THRESHOLDS["neutral"]:
                signal = "Neutral"
            else:
                signal = "Sell"

            # Extract key features at this date
            mom_20d = float(spy.loc[date, "momentum_20d"]) if "momentum_20d" in spy.columns else np.nan
            mom_10d = float(spy.loc[date, "momentum_10d"]) if "momentum_10d" in spy.columns else np.nan
            zscore_20d = float(spy.loc[date, "zscore_20d"]) if "zscore_20d" in spy.columns else np.nan
            roc_20d = float(spy.loc[date, "roc_20d"]) if "roc_20d" in spy.columns else np.nan
            change20d = float(spy.loc[date, "Change20day"]) if "Change20day" in spy.columns else np.nan
            rm = float(rolling_med.loc[date]) if date in rolling_med.index and not pd.isna(rolling_med.loc[date]) else np.nan
            fr = float(future_ret.loc[date]) if date in future_ret.index and not pd.isna(future_ret.loc[date]) else np.nan

            records.append({
                "date": date,
                "window": n_windows,
                "raw_prob": raw_p,
                "cal_prob": cal_p,
                "actual": actual,
                "signal": signal,
                "momentum_20d": mom_20d,
                "momentum_10d": mom_10d,
                "zscore_20d": zscore_20d,
                "roc_20d": roc_20d,
                "Change20day": change20d,
                "rolling_median": rm,
                "future_ret": fr,
            })

    df = pd.DataFrame(records)
    print(f"   Collected {len(df)} predictions across {n_windows} windows\n")

    # === CHECK 1: Window-Signal Distribution ===
    print("=" * 60)
    print("=== CHECK 1: Window-Signal Distribution ===")
    print("=" * 60)
    for w in sorted(df["window"].unique()):
        w_df = df[df["window"] == w]
        parts = []
        for sig in ["Strong Buy", "Buy", "Neutral", "Sell"]:
            s_df = w_df[w_df["signal"] == sig]
            if len(s_df) > 0:
                up_rate = s_df["actual"].mean() * 100
                parts.append(f"{sig[:2]}={len(s_df)}({up_rate:.0f}% up)")
        print(f"Window {w}: {', '.join(parts)}")
    print()

    # === CHECK 2: Raw vs Calibrated Probability ===
    print("=" * 60)
    print("=== CHECK 2: Raw vs Calibrated Probability ===")
    print("=" * 60)
    for sig in ["Strong Buy", "Buy", "Neutral", "Sell"]:
        s_df = df[df["signal"] == sig]
        if len(s_df) > 0:
            print(f"{sig:12s}: raw_mean={s_df['raw_prob'].mean():.3f}, "
                  f"cal_mean={s_df['cal_prob'].mean():.3f}, "
                  f"actual_up={s_df['actual'].mean():.1%}, n={len(s_df)}")
    overall_raw = df["raw_prob"].mean()
    overall_cal = df["cal_prob"].mean()
    print(f"{'Overall':12s}: raw_mean={overall_raw:.3f}, cal_mean={overall_cal:.3f}")

    # Check monotonicity: is raw rank preserved?
    if len(df) > 0:
        rank_corr = df["raw_prob"].corr(df["cal_prob"], method="spearman")
        print(f"Raw-Cal rank correlation (Spearman): {rank_corr:.4f}")
    print()

    # === CHECK 3: Momentum-Target Correlation at Signal Extremes ===
    print("=" * 60)
    print("=== CHECK 3: Momentum-Target at Signal Extremes ===")
    print("=" * 60)
    sb_df = df[df["cal_prob"] >= SIGNAL_THRESHOLDS["strong_buy"]]
    sell_df = df[df["cal_prob"] < SIGNAL_THRESHOLDS["neutral"]]

    for label, subset in [("Strong Buy (cal>=0.60)", sb_df), ("Sell (cal<0.45)", sell_df), ("Overall", df)]:
        if len(subset) == 0:
            continue
        mom20 = subset["momentum_20d"].mean()
        roc20 = subset["roc_20d"].mean()
        ch20 = subset["Change20day"].mean()
        rm = subset["rolling_median"].mean()
        fr = subset["future_ret"].mean()
        gap = fr - rm if not (np.isnan(fr) or np.isnan(rm)) else np.nan
        print(f"\n{label} (n={len(subset)}):")
        print(f"  momentum_20d mean: {mom20:+.4f} ({mom20*100:+.1f}%)")
        print(f"  roc_20d mean:      {roc20:+.2f}")
        print(f"  Change20day mean:  {ch20:+.2f}")
        print(f"  rolling_median:    {rm:+.4f} ({rm*100:+.1f}%)")
        print(f"  future_ret mean:   {fr:+.4f} ({fr*100:+.1f}%)")
        print(f"  future_ret - rolling_med gap: {gap:+.4f}" if not np.isnan(gap) else "  gap: N/A")
    print()

    # === CHECK 4: Temporal Clustering ===
    print("=" * 60)
    print("=== CHECK 4: Temporal Clustering ===")
    print("=" * 60)
    for sig in ["Strong Buy", "Buy", "Neutral", "Sell"]:
        s_df = df[df["signal"] == sig]
        if len(s_df) > 0:
            dates = s_df["date"].sort_values()
            print(f"{sig:12s}: n={len(s_df)}, "
                  f"range={dates.iloc[0].strftime('%Y-%m-%d')} ~ {dates.iloc[-1].strftime('%Y-%m-%d')}")
            # Show date list for small samples
            if len(s_df) <= 20:
                date_strs = [d.strftime("%Y-%m-%d") for d in dates]
                print(f"             dates: {', '.join(date_strs)}")
    print()

    # === CHECK 5: Feature Importance at Extremes ===
    print("=" * 60)
    print("=== CHECK 5: Feature Values at Extremes ===")
    print("=" * 60)
    high_conf = df[df["cal_prob"] >= 0.60]
    low_conf = df[df["cal_prob"] <= 0.45]
    feature_cols = ["momentum_20d", "momentum_10d", "zscore_20d", "roc_20d", "Change20day", "rolling_median"]

    print(f"{'Feature':20s} | {'High cal (>=0.60)':>18s} | {'Low cal (<=0.45)':>18s} | {'Difference':>12s}")
    print("-" * 75)
    for col in feature_cols:
        if col in df.columns:
            h_mean = high_conf[col].mean() if len(high_conf) > 0 else np.nan
            l_mean = low_conf[col].mean() if len(low_conf) > 0 else np.nan
            diff = h_mean - l_mean if not (np.isnan(h_mean) or np.isnan(l_mean)) else np.nan
            h_str = f"{h_mean:+.4f}" if not np.isnan(h_mean) else "N/A"
            l_str = f"{l_mean:+.4f}" if not np.isnan(l_mean) else "N/A"
            d_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
            print(f"{col:20s} | {h_str:>18s} | {l_str:>18s} | {d_str:>12s}")
    print()

    # === VERDICT ===
    print("=" * 60)
    print("=== VERDICT ===")
    print("=" * 60)

    # H1: Momentum-Target Mismatch
    h1_confirmed = False
    if len(sb_df) > 0:
        sb_mom = sb_df["momentum_20d"].mean()
        overall_mom = df["momentum_20d"].mean()
        sb_rm = sb_df["rolling_median"].mean()
        overall_rm = df["rolling_median"].mean()
        sb_gap = sb_df["future_ret"].mean() - sb_df["rolling_median"].mean()

        if sb_mom > overall_mom * 1.5 and sb_rm > overall_rm * 1.3:
            h1_confirmed = True
        # Also confirm if gap is negative (future_ret < rolling_median for SB)
        if not np.isnan(sb_gap) and sb_gap < 0:
            h1_confirmed = True

    # H2: Calibration Distortion
    h2_confirmed = False
    if len(df) > 0:
        rank_corr = df["raw_prob"].corr(df["cal_prob"], method="spearman")
        if rank_corr < 0.95:
            h2_confirmed = True

    # H3: Small Sample
    h3_confirmed = len(sb_df) < 15

    # H4: Regime-dependent
    h4_confirmed = False
    if len(sb_df) > 0:
        sb_windows = sb_df["window"].unique()
        if len(sb_windows) <= 2:
            h4_confirmed = True

    print(f"H1 (Momentum-Target Mismatch): {'CONFIRMED' if h1_confirmed else 'REJECTED'}")
    print(f"H2 (Calibration Distortion):   {'CONFIRMED' if h2_confirmed else 'REJECTED'}")
    print(f"H3 (Small Sample):             {'CONFIRMED' if h3_confirmed else 'REJECTED'}")
    print(f"H4 (Regime-dependent):         {'CONFIRMED' if h4_confirmed else 'REJECTED'}")

    confirmed = []
    if h1_confirmed:
        confirmed.append("H1")
    if h2_confirmed:
        confirmed.append("H2")
    if h3_confirmed:
        confirmed.append("H3")
    if h4_confirmed:
        confirmed.append("H4")

    primary = confirmed[0] if confirmed else "NONE"
    secondary = ", ".join(confirmed[1:]) if len(confirmed) > 1 else "NONE"
    print(f"\nPrimary cause: {primary}")
    print(f"Secondary: {secondary}")


if __name__ == "__main__":
    main()
