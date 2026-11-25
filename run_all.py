"""Run the whole pipeline: data -> features -> walk-forward backtest ->
charts + metrics JSON. Charts regenerate with this script (PNGs are
gitignored and excluded from any GitHub push).
Usage: python run_all.py   (from the project root)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import max_drawdown, sharpe, summarise, walk_forward
from src.data_gen import ASSETS, DESK_MAP, generate, save_synthetic
from src.evaluate import insample_fit, regime_split
from src.features import FEATURES, build_features
from src.models import make_models

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "reports" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def main():
    df = generate()
    save_synthetic(df, ROOT / "data" / "synthetic" / "synthetic_markets.csv")
    feat = build_features(df)
    feat.to_csv(ROOT / "data" / "processed" / "features.csv", index=False)
    results = walk_forward(feat)
    summary = summarise(results)

    # In-sample fit on the full feature set (diagnostic only).
    models = make_models()
    train = feat
    for name, m in models.items():
        if m is None:
            continue
        if name == "linear":
            m.fit(train[FEATURES].values, train["target_ret"].values)
        else:
            m.fit(train[FEATURES].values, train["target_dir"].values)
    insample = insample_fit(feat, models)
    regimes = regime_split(feat, results)

    metrics = {
        "assets": ASSETS,
        "desk_map": DESK_MAP,
        "n_feature_rows": int(len(feat)),
        "oos_days": int(len(results["logistic"]["dates"])),
        "summary": summary.to_dict(orient="records"),
        "insample_accuracy": insample,
        "regimes": regimes,
        "synthetic_data": True,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # --- Charts ---
    # 1. Equity curves (OOS, net of costs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, r in results.items():
        eq = np.cumprod(1 + np.array(r["port_ret"]))
        ax.plot(pd.to_datetime(r["dates"]), eq, label=name.replace("_", " "))
    ax.set_title("Walk-forward equity curves — out-of-sample, net of 5bp costs (synthetic data)")
    ax.set_ylabel("Cumulative growth of $1")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "equity_curves.png"); plt.close(fig)

    # 2. Model comparison: OOS Sharpe and hit rate
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    s = summary.set_index("model")
    axes[0].bar(s.index, s["sharpe"], color="#1A3A6B")
    axes[0].set_title("Out-of-sample Sharpe")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].bar(s.index, s["hit_rate"], color="#2E7D5B")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Out-of-sample hit rate (red = 50%)")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "model_comparison.png"); plt.close(fig)

    # 3. Feature importance: logistic coefficients vs RF importance
    from sklearn.inspection import permutation_importance  # noqa
    logi = models["logistic"]
    coefs = logi[-1].coef_[0]
    rf_imp = models["random_forest"].feature_importances_
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].barh(FEATURES, coefs, color="#1A3A6B")
    axes[0].set_title("Logistic regression coefficients")
    axes[1].barh(FEATURES, rf_imp, color="#8A6D00")
    axes[1].set_title("Random forest feature importance")
    fig.tight_layout(); fig.savefig(CHARTS / "feature_importance.png"); plt.close(fig)

    # 4. Return distribution of the best model vs realised asset returns
    best = summary.iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.array(results[best]["port_ret"]) * 10000, bins=60, color="#1A3A6B", alpha=0.8)
    ax.set_title(f"Daily portfolio return distribution — {best} (bp, out-of-sample)")
    ax.set_xlabel("Daily return (bp)")
    fig.tight_layout(); fig.savefig(CHARTS / "return_distribution.png"); plt.close(fig)

    # 5. Drawdown of the best model
    eq = np.cumprod(1 + np.array(results[best]["port_ret"]))
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(pd.to_datetime(results[best]["dates"]), dd, 0, color="#B03A2E", alpha=0.6)
    ax.set_title(f"Drawdown — {best} (%, out-of-sample)")
    fig.tight_layout(); fig.savefig(CHARTS / "drawdown.png"); plt.close(fig)

    # 6. Information coefficient by asset (logistic, OOS)
    fig, ax = plt.subplots(figsize=(9, 4))
    ic_by_asset = {}
    for asset in ASSETS:
        sub = feat[feat.asset == asset].sort_values("date")
        # Recompute OOS IC per asset from stored daily IC is not possible;
        # use full-sample directional accuracy as the per-asset view instead.
        pred = (sub["mom20"] > 0).astype(int) * 2 - 1
        ic_by_asset[asset] = float(np.corrcoef(pred, sub["target_ret"])[0, 1])
    ax.bar(ic_by_asset.keys(), ic_by_asset.values(), color="#2E7D5B")
    ax.set_title("Momentum-signal IC by asset (full sample, diagnostic)")
    ax.set_ylabel("Correlation(signal, next-day return)")
    fig.tight_layout(); fig.savefig(CHARTS / "ic_by_asset.png"); plt.close(fig)

    # 7. Cost sensitivity: Sharpe vs transaction cost
    fig, ax = plt.subplots(figsize=(9, 4))
    costs = [0, 2, 5, 10, 20]
    for name, r in results.items():
        # Reconstruct gross returns approx: net + turnover*cost. Turnover stored.
        vals = []
        for c in costs:
            gross = np.array(r["port_ret"]) + np.array(r["turnover"]) * 5 / 10000
            net = gross - np.array(r["turnover"]) * c / 10000
            vals.append(sharpe(net))
        ax.plot(costs, vals, marker="o", label=name.replace("_", " "))
    ax.set_title("Sharpe vs transaction cost (bp per trade)")
    ax.set_xlabel("Cost (bp)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "cost_sensitivity.png"); plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nOOS days: {len(results['logistic']['dates'])}")
    print(f"Charts written to {CHARTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
