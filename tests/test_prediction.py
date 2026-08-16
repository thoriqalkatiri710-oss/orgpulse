import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from src.prediction.flight_risk_model import evaluate_model


def test_model_roc_auc_above_baseline():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=20,
                                n_informative=10, random_state=42)
    X_df  = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_df, y)
    metrics = evaluate_model(model, X_df, pd.Series(y))
    assert metrics["roc_auc"] > 0.6, "Model seharusnya mengungguli random classifier"


def test_flight_risk_scores_bounded():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_df  = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_df, y)
    proba = model.predict_proba(X_df)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_shap_values_sum_to_prediction():
    import shap
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_df  = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    model = GradientBoostingClassifier(n_estimators=20, random_state=42)
    model.fit(X_df, y)

    # TreeExplainer dengan model_output="raw" → log-odds space
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_vals = explainer(X_df)

    # Prediksi dalam log-odds space
    from scipy.special import logit
    for i in range(len(X_df)):
        expected = shap_vals.base_values[i] + shap_vals.values[i].sum()
        actual   = logit(model.predict_proba(X_df.iloc[[i]])[0, 1])
        assert abs(expected - actual) < 0.02, f"SHAP values tidak konsisten untuk row {i}"