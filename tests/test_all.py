"""Test suite. Prints a runtime 'N passed' line — quote that line,
never a count of check() call sites (checks inside loops run many times).
Run: python -m tests.test_all  (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest import max_drawdown, sharpe, walk_forward, summarise
from src.data_gen import ASSETS, DESK_MAP, generate
from src.features import FEATURES, build_features
from src.models import make_models, rule_momentum_signal

PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL: {name}")


def main():
    global PASSED, FAILED
    df = generate()
    # --- Data generation ---
    check("six assets", set(df["asset"].unique()) == set(ASSETS))
    check("1250 days per asset", all(len(df[df.asset == a]) == 1250 for a in ASSETS))
    check("no missing closes", df["close"].notna().all())
    check("prices positive", (df["close"] > 0).all())
    check("dates are business days", df["date"].dt.dayofweek.max() <= 4)
    check("desk map covers all assets", set(DESK_MAP) == set(ASSETS))
    check("returns finite", np.isfinite(df["ret"]).all())
    # Reproducibility: same seed -> identical data.
    check("seed reproducible", generate().equals(df))
    check("different seed differs", not generate(seed=7).equals(df))

    # --- Features: no lookahead by construction ---
    feat = build_features(df)
    check("feature columns present", all(c in feat.columns for c in FEATURES))
    check("target columns present", "target_ret" in feat and "target_dir" in feat)
    check("no NaN in features", feat[FEATURES].notna().all().all())
    # Target is the next-day return: verify against the raw series directly.
    _g = df[df.asset == "EQ"].sort_values("date").reset_index(drop=True)
    _f = feat[feat.asset == "EQ"].sort_values("date").reset_index(drop=True)
    _idx = df[df.asset == "EQ"].sort_values("date").reset_index(drop=True).index
    check("target is next-day return",
          abs(_f.iloc[0]["target_ret"] - _g.iloc[60]["ret"]) < 1e-12)
    check("target_dir is 0/1", set(feat["target_dir"].unique()) <= {0, 1})
    check("rows dropped only for warm-up windows", 0 < len(feat) < len(df))
    check("mom20 finite", np.isfinite(feat["mom20"]).all())
    check("rsi in [0,100]", feat["rsi14"].between(0, 100).all())
    check("xsec momentum sums to ~0 per day",
          feat.groupby("date")["xsec_mom20"].sum().abs().max() < 1e-9)

    # --- Models ---
    models = make_models()
    check("four models", set(models) == {"momentum_rule", "logistic", "linear", "random_forest"})
    check("rule signal is -1/+1", set(np.unique(rule_momentum_signal(feat))) <= {-1, 1})

    # --- Backtest ---
    results = walk_forward(feat)
    for name, r in results.items():
        check(f"{name}: has OOS days", len(r["port_ret"]) > 500)
        check(f"{name}: returns finite", np.isfinite(r["port_ret"]).all())
        check(f"{name}: hit rate in [0,1]", 0 <= np.mean(r["hit"]) <= 1)
        check(f"{name}: turnover in [0,1]", 0 <= np.mean(r["turnover"]) <= 1)
        check(f"{name}: dates align with returns", len(r["dates"]) == len(r["port_ret"]))
    summary = summarise(results)
    check("summary has four rows", len(summary) == 4)
    check("summary Sharpe finite", np.isfinite(summary["sharpe"]).all())
    check("summary sorted by Sharpe desc",
          summary["sharpe"].is_monotonic_decreasing)

    # --- Metric helpers ---
    check("sharpe of zero returns is 0", sharpe(np.zeros(100)) == 0.0)
    check("sharpe positive for positive drift",
          sharpe(np.random.default_rng(0).normal(0.001, 0.01, 500)) > 0)
    check("max drawdown of rising series is 0",
          max_drawdown(np.arange(1, 101, dtype=float)) == 0.0)
    check("max drawdown negative after a fall",
          max_drawdown(np.array([100.0, 120.0, 90.0, 110.0])) < 0)

    # --- Honest-finding guardrails: results must be modest, not magic ---
    for _, row in summary.iterrows():
        check(f"{row['model']}: Sharpe below 3 (no magic)", abs(row["sharpe"]) < 3.0)
        check(f"{row['model']}: hit rate below 0.65", row["hit_rate"] < 0.65)
    check("at least one model beats 50% hit rate",
          (summary["hit_rate"] > 0.50).any())

    print(f"{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
