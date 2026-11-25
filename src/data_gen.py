"""Synthetic multi-asset market data.

SYNTHETIC DATA ONLY. No real market prices are used anywhere in this
project. The generator is designed so the data-generating process is
known and auditable: daily log-returns carry a small, persistent
momentum component plus volatility clustering, which is exactly the
kind of weak, noisy signal a markets quant desk actually faces.
Nothing here is presented as live-market performance.
"""
import numpy as np
import pandas as pd

ASSETS = ["EQ", "CREDIT", "RATES", "FX", "MUNI", "SEC"]
# Desk mapping: which Barclays rotation each synthetic instrument stands in for.
DESK_MAP = {
    "EQ": "Equities",
    "CREDIT": "Credit",
    "RATES": "Rates",
    "FX": "Foreign Exchange",
    "MUNI": "Municipals",
    "SEC": "Securitized Products",
}
# Annualised drift (bp/day) and daily vol, per asset.
PARAMS = {
    "EQ": {"drift": 0.00025, "vol": 0.011, "mom_beta": 0.09},
    "CREDIT": {"drift": 0.00012, "vol": 0.006, "mom_beta": 0.07},
    "RATES": {"drift": -0.00005, "vol": 0.007, "mom_beta": 0.06},
    "FX": {"drift": 0.00002, "vol": 0.006, "mom_beta": 0.05},
    "MUNI": {"drift": 0.00010, "vol": 0.004, "mom_beta": 0.06},
    "SEC": {"drift": 0.00015, "vol": 0.008, "mom_beta": 0.08},
}


def generate(n_days=1250, seed=202608, start="2021-01-04"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    # Weak cross-asset correlation through a common factor.
    common = rng.normal(0, 1, n_days)
    out = {}
    for a in ASSETS:
        p = PARAMS[a]
        ret = np.zeros(n_days)
        vol = np.full(n_days, p["vol"])
        for t in range(1, n_days):
            # Volatility clustering: today's vol rises after large moves.
            vol[t] = p["vol"] * (0.85 + 0.30 * abs(ret[t - 1]) / p["vol"])
            vol[t] = min(vol[t], 3.0 * p["vol"])
            mom20 = ret[max(0, t - 20):t].mean() if t >= 5 else 0.0
            # Momentum signal: past 20-day mean return predicts today.
            ret[t] = (p["drift"] + p["mom_beta"] * mom20
                      + 0.25 * vol[t] * common[t] + 0.75 * vol[t] * rng.normal())
        price = 100.0 * np.exp(np.cumsum(ret))
        out[a] = pd.DataFrame({"date": dates, "asset": a, "close": price,
                               "ret": ret, "vol": vol})
    df = pd.concat(out.values(), ignore_index=True)
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    return df


def save_synthetic(df, path):
    df.to_csv(path, index=False)
    return path
