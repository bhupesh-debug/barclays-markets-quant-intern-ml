"""Evaluation helpers: in-sample vs out-of-sample, regime splits."""
import numpy as np
import pandas as pd

from .features import FEATURES


def insample_fit(feat, models):
    """In-sample accuracy of each fitted model on the full feature set.

    Reported ONLY as a diagnostic, labelled in-sample. It is not
    performance evidence.
    """
    from sklearn.metrics import accuracy_score
    out = {}
    X = feat[FEATURES].values
    for name, m in models.items():
        if m is None:
            pred = (feat["mom20"] > 0).astype(int)
            out[name] = float(accuracy_score(feat["target_dir"], pred))
        elif name == "linear":
            pred = (m.predict(X) > 0).astype(int)
            out[name] = float(accuracy_score(feat["target_dir"], pred))
        else:
            out[name] = float(accuracy_score(feat["target_dir"], m.predict(X)))
    return out


def regime_split(feat, results):
    """Sharpe in high-vol vs low-vol days (vol20 above/below its median)."""
    feat = feat.sort_values(["date", "asset"]).reset_index(drop=True)
    vol_day = feat.groupby("date")["vol20"].mean()
    median_vol = vol_day.median()
    out = {}
    for name, r in results.items():
        ret = pd.Series(r["port_ret"], index=pd.to_datetime(r["dates"]))
        hi = ret[vol_day.reindex(ret.index) > median_vol]
        lo = ret[vol_day.reindex(ret.index) <= median_vol]
        def _sh(x):
            return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 10 and x.std() else 0.0
        out[name] = {"sharpe_high_vol": _sh(hi), "sharpe_low_vol": _sh(lo),
                     "n_high": len(hi), "n_low": len(lo)}
    return out
