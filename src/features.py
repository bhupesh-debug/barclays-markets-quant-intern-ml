"""Feature engineering: desk-style signals, no lookahead.

Every feature at day t uses only data up to and including day t.
The target is the NEXT day's return / direction, so no leakage.
"""
import numpy as np
import pandas as pd

FEATURES = ["mom5", "mom20", "mom60", "vol20", "rsi14", "carry", "xsec_mom20"]


def _rsi(ret, window=14):
    gains = np.where(ret > 0, ret, 0.0)
    losses = np.where(ret < 0, -ret, 0.0)
    ag = pd.Series(gains).rolling(window).mean().values
    al = pd.Series(losses).rolling(window).mean().values
    rs = np.divide(ag, al, out=np.ones_like(ag), where=al != 0)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(df):
    frames = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("date").copy()
        r = g["ret"].values
        g["mom5"] = pd.Series(r).rolling(5).mean().values
        g["mom20"] = pd.Series(r).rolling(20).mean().values
        g["mom60"] = pd.Series(r).rolling(60).mean().values
        g["vol20"] = pd.Series(r).rolling(20).std().values
        g["rsi14"] = _rsi(r)
        # Carry proxy: constant asset-level yield, synthetic.
        g["carry"] = 0.0002
        g["target_ret"] = pd.Series(r).shift(-1).values
        g["target_dir"] = (g["target_ret"] > 0).astype(int)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    # Cross-sectional momentum: asset's mom20 minus the day's cross-sectional mean.
    out["xsec_mom20"] = out["mom20"] - out.groupby("date")["mom20"].transform("mean")
    out = out.dropna(subset=FEATURES + ["target_ret"]).reset_index(drop=True)
    return out
