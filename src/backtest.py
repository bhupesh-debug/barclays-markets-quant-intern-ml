"""Walk-forward backtest with transaction costs.

Protocol (no lookahead):
- Start after a 250-day warm-up. Retrain every 60 trading days on all
  data up to the retrain date (expanding window).
- Each day, each model outputs a -1/+1 position per asset for the
  NEXT day. Positions are equal-weighted across the six assets.
- Daily portfolio return = mean over assets of position * next-day
  return, minus 5 bps per unit of position change (turnover cost).
- Out-of-sample period only is reported; in-sample fit is shown
  separately and labelled as such.
"""
import numpy as np
import pandas as pd

from .features import FEATURES
from .models import make_models, predict_direction, rule_momentum_signal, TCOST_BP


def sharpe(returns, periods=252):
    r = np.asarray(returns, dtype=float)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def information_coefficient(pred_dir, realised):
    """Rank IC: correlation between predicted direction and realised return."""
    p = np.asarray(pred_dir, dtype=float)
    r = np.asarray(realised, dtype=float)
    if p.std() == 0 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(p, r)[0, 1])


def walk_forward(feat, warmup=250, retrain_every=60):
    feat = feat.sort_values(["date", "asset"]).reset_index(drop=True)
    dates = sorted(feat["date"].unique())
    models = make_models()
    # Per-model daily portfolio returns and per-asset predictions.
    results = {name: {"dates": [], "port_ret": [], "hit": [], "ic": [],
                      "turnover": []} for name in models}
    prev_pos = {name: None for name in models}
    fitted = {}
    for i, d in enumerate(dates):
        if i < warmup:
            continue
        day = feat[feat["date"] == d]
        if len(day) < 6:
            continue
        if (i - warmup) % retrain_every == 0:
            train = feat[feat["date"] < d]
            Xtr = train[FEATURES].values
            for name, m in models.items():
                if m is None:
                    continue
                if name == "linear":
                    m.fit(Xtr, train["target_ret"].values)
                else:
                    m.fit(Xtr, train["target_dir"].values)
                fitted[name] = m
        X = day[FEATURES].values
        realised = day["target_ret"].values
        for name in models:
            if name == "momentum_rule":
                pos = rule_momentum_signal(day)
            else:
                pos = predict_direction(name, fitted[name], X)
            pos = np.asarray(pos, dtype=float)
            gross = float(np.mean(pos * realised))
            turnover = 0.0
            if prev_pos[name] is not None:
                turnover = float(np.mean(np.abs(pos - prev_pos[name]))) / 2.0
            cost = turnover * TCOST_BP / 10000.0
            results[name]["dates"].append(d)
            results[name]["port_ret"].append(gross - cost)
            results[name]["hit"].append(float(np.mean((pos * realised) > 0)))
            results[name]["ic"].append(information_coefficient(pos, realised))
            results[name]["turnover"].append(turnover)
            prev_pos[name] = pos
    return results


def summarise(results):
    rows = []
    for name, r in results.items():
        ret = np.array(r["port_ret"])
        eq = np.cumprod(1 + ret)
        rows.append({
            "model": name,
            "n_days": len(ret),
            "hit_rate": float(np.mean(r["hit"])),
            "mean_daily_bp": float(ret.mean() * 10000),
            "sharpe": sharpe(ret),
            "total_return_pct": float((eq[-1] - 1) * 100),
            "max_drawdown_pct": float(max_drawdown(eq) * 100),
            "mean_ic": float(np.mean(r["ic"])),
            "mean_turnover": float(np.mean(r["turnover"])),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)
