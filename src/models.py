"""Models: one interpretable baseline, two linear models, one tree model.

All models see the SAME feature set and the SAME walk-forward splits,
so the comparison is fair. The data-generating process is linear in
the momentum signal plus noise, so a linear model is expected to do
well and a random forest is expected to waste capacity on noise —
that is a finding about the data, not a rigged comparison.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TCOST_BP = 5.0  # 5 bps per one-way trade


def rule_momentum_signal(feat):
    """Baseline: long when 20-day momentum is positive, short otherwise."""
    return (feat["mom20"] > 0).astype(int) * 2 - 1


def make_models():
    return {
        "momentum_rule": None,  # handled by rule_momentum_signal
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000, C=1.0)),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
    }


def predict_direction(model_name, model, X):
    if model_name == "momentum_rule":
        return None  # caller uses rule_momentum_signal
    if model_name == "linear":
        return (model.predict(X) > 0).astype(int) * 2 - 1
    proba = model.predict_proba(X)[:, 1]
    return (proba > 0.5).astype(int) * 2 - 1
