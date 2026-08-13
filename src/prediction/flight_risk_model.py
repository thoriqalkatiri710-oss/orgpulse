import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, classification_report)
import shap
import pickle


# ── Config ────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["department", "gender", "education", "network_role"]
DROP_COLS = [
    "employee_id", "is_flight_risk_sim",
    "avg_hours_worked",  # terlalu berkorelasi langsung dengan target
    "absent_rate",       # sama
    "no_show_count",     # sama
]


# ── 6.2.1 Feature Preparation ─────────────────────────────────────────────────

def prepare_features(df_features: pd.DataFrame,
                     target_col: str = "is_flight_risk_sim") -> tuple:
    """Encode categorical features dan pisahkan X dan y."""
    df = df_features.copy()
    le = LabelEncoder()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    y = df[target_col].astype(int) if target_col in df.columns else None

    return X, y


# ── 6.2.1 Model Training ──────────────────────────────────────────────────────

def train_flight_risk_model(X: pd.DataFrame, y: pd.Series,
                             seed: int = 42) -> tuple:
    """
    Gradient Boosting Classifier dengan 5-fold stratified CV.
    CV ROC-AUC sebagai estimasi performa yang lebih robust dari train score.
    """
    model = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X, y)
    return model, cv_scores


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "roc_auc":   round(roc_auc_score(y, y_proba), 3),
        "precision": round(precision_score(y, y_pred), 3),
        "recall":    round(recall_score(y, y_pred), 3),
        "f1_score":  round(f1_score(y, y_pred), 3),
        "report":    classification_report(y, y_pred),
    }


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, path: str = "results/models/flight_risk_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {path}")


def load_model(path: str = "results/models/flight_risk_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Load features
    df_features = pd.read_csv("data/processed/flight_risk_features.csv")
    df_emp      = pd.read_csv("data/simulated/employees.csv")

    # Merge ground truth
    df_features = df_features.merge(
        df_emp[["employee_id", "is_flight_risk_sim"]], on="employee_id", how="left"
    )

    print(f"Dataset: {df_features.shape}")
    print(f"Flight risk rate: {df_features['is_flight_risk_sim'].mean():.2%}")

    # Prepare & train
    X, y = prepare_features(df_features)
    print(f"\nFeatures: {X.shape[1]} kolom")
    print(f"Training model...")

    model, cv_scores = train_flight_risk_model(X, y)

    # Evaluate
    metrics = evaluate_model(model, X, y)
    print(f"\n── Model Metrics ──")
    print(f"ROC-AUC   : {metrics['roc_auc']}")
    print(f"Precision : {metrics['precision']}")
    print(f"Recall    : {metrics['recall']}")
    print(f"F1 Score  : {metrics['f1_score']}")
    print(f"\n{metrics['report']}")

    # Feature importance
    feat_importance = pd.DataFrame({
        "feature":   X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("── Top 10 Feature Importance ──")
    print(feat_importance.head(10).to_string(index=False))

    # Save
    save_model(model)
    X["flight_risk_score"] = model.predict_proba(X)[:, 1]
    X["employee_id"] = df_features["employee_id"].values
    X["is_flight_risk_sim"] = y.values
    X[["employee_id", "flight_risk_score", "is_flight_risk_sim"]].to_csv(
        "data/processed/flight_risk_scores.csv", index=False
    )
    print("✅ Saved: flight_risk_scores.csv")