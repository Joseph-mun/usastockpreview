#!/usr/bin/env python3
"""Verify no feature-target leakage exists in the dataset."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data.features import DatasetBuilder
from src.data.collectors import SMACollector
from src.data.cache import SMACache


def main():
    print("=" * 60)
    print("  Feature-Target Leakage Verification")
    print("=" * 60)

    # Build dataset
    cache = SMACache()
    raw_sma, _ = cache.load()
    sma_ratios = {}
    if raw_sma:
        sc = SMACollector()
        sc.raw_dataframes = raw_sma
        sma_ratios = sc.compute_ratios()

    builder = DatasetBuilder(sma_ratios=sma_ratios)
    X, spy, y = builder.build("IXIC", for_prediction=False)

    passed = 0
    failed = 0

    # Check 1: No feature has |correlation| > 0.5 with Target
    print("\n[CHECK 1] Feature-Target correlation (|r| < 0.5)")
    max_corr = 0
    max_corr_feature = ""
    fail_features = []
    for col in X.columns:
        corr = abs(X[col].corr(y))
        if np.isnan(corr):
            continue
        if corr > max_corr:
            max_corr = corr
            max_corr_feature = col
        if corr > 0.5:
            fail_features.append((col, corr))

    if not fail_features:
        print(f"  PASS: max |r| = {max_corr:.4f} ({max_corr_feature})")
        passed += 1
    else:
        for col, corr in fail_features:
            print(f"  FAIL: {col} has |r| = {corr:.4f}")
        failed += 1

    # Check 2: Target base rate between 40-60%
    print("\n[CHECK 2] Target base rate (40-60%)")
    base_rate = y.mean()
    if 0.40 <= base_rate <= 0.60:
        print(f"  PASS: base rate = {base_rate:.4f}")
        passed += 1
    else:
        print(f"  FAIL: base rate = {base_rate:.4f}")
        failed += 1

    # Check 3: Simple 1-fold OOS accuracy < 70%
    print("\n[CHECK 3] Simple OOS accuracy (< 70%)")
    from sklearn.ensemble import RandomForestClassifier
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_tr.values, y_tr.values)
    acc = rf.score(X_te.values, y_te.values)
    if acc < 0.70:
        print(f"  PASS: OOS accuracy = {acc:.4f}")
        passed += 1
    else:
        print(f"  WARN: OOS accuracy = {acc:.4f} (suspiciously high)")
        failed += 1

    # Check 4: momentum_20d not correlated with Target
    print("\n[CHECK 4] momentum_20d correlation with Target")
    if "momentum_20d" in X.columns:
        corr = abs(X["momentum_20d"].corr(y))
        if corr < 0.3:
            print(f"  PASS: |r| = {corr:.4f}")
            passed += 1
        else:
            print(f"  FAIL: |r| = {corr:.4f}")
            failed += 1
    else:
        print("  SKIP: momentum_20d not in features")
        passed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} PASS, {failed} FAIL")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
