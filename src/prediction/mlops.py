import json
import pickle
import pandas as pd
from datetime import datetime


# ── 18.1.1 Model Retraining Schedule ─────────────────────────────────────────

def schedule_model_retraining(current_model_date: str,
                               current_performance: dict,
                               threshold_auc: float = 0.75) -> dict:
    """
    Evaluasi apakah model perlu di-retrain.
    Trigger: model > 90 hari ATAU AUC turun di bawah threshold.
    """
    model_age_days      = (datetime.now() - datetime.fromisoformat(current_model_date)).days
    performance_degraded = current_performance.get("roc_auc", 1.0) < threshold_auc

    schedule = {
        "model_date":          current_model_date,
        "model_age_days":      model_age_days,
        "current_auc":         current_performance.get("roc_auc"),
        "retrain_recommended": model_age_days > 90 or performance_degraded,
        "retrain_priority": (
            "URGENT"     if performance_degraded  else
            "SCHEDULED"  if model_age_days > 90   else
            "NOT_NEEDED"
        ),
        "reason": (
            "AUC below threshold"  if performance_degraded  else
            "Model > 90 days old"  if model_age_days > 90   else
            "Model still fresh"
        )
    }
    return schedule


# ── 18.2.1 Model Versioning ───────────────────────────────────────────────────

def save_model_with_metadata(model, metrics: dict, features: list,
                              output_dir: str = "results/models") -> str:
    """
    Simpan model beserta metadata lengkap untuk reproducibility.
    Setiap model punya timestamp unik dan metadata terpisah.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_dir}/flight_risk_{timestamp}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_timestamp":  timestamp,
        "model_type":       type(model).__name__,
        "n_features":       len(features),
        "feature_names":    features,
        "training_metrics": metrics,
        "n_estimators":     getattr(model, "n_estimators", None),
        "max_depth":        getattr(model, "max_depth", None),
        "learning_rate":    getattr(model, "learning_rate", None),
    }

    metadata_path = f"{output_dir}/metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Model saved  : {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    return model_path


def load_model_with_metadata(model_path: str) -> tuple:
    """Load model dan metadata-nya."""
    metadata_path = model_path.replace("flight_risk_", "metadata_").replace(".pkl", ".json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


def list_model_registry(output_dir: str = "results/models") -> pd.DataFrame:
    """List semua model yang tersimpan beserta performance-nya."""
    import os
    records = []

    for fname in os.listdir(output_dir):
        if fname.startswith("metadata_") and fname.endswith(".json"):
            with open(f"{output_dir}/{fname}") as f:
                meta = json.load(f)
            records.append({
                "timestamp":  meta.get("model_timestamp"),
                "model_type": meta.get("model_type"),
                "n_features": meta.get("n_features"),
                "roc_auc":    meta.get("training_metrics", {}).get("roc_auc"),
                "f1_score":   meta.get("training_metrics", {}).get("f1_score"),
            })

    if not records:
        return pd.DataFrame(columns=["timestamp", "model_type", "n_features", "roc_auc", "f1_score"])

    return pd.DataFrame(records).sort_values("timestamp", ascending=False)

if __name__ == "__main__":
    import pickle

    print("── 18.1 Model Retraining Schedule ──")

    # Simulasi model lama (dibuat 100 hari lalu)
    from datetime import timedelta
    old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    new_date = datetime.now().strftime("%Y-%m-%d")

    scenarios = [
        ("Model lama, AUC baik",     old_date, {"roc_auc": 0.87}),
        ("Model baru, AUC baik",     new_date, {"roc_auc": 0.87}),
        ("Model baru, AUC turun",    new_date, {"roc_auc": 0.71}),
        ("Model lama, AUC turun",    old_date, {"roc_auc": 0.70}),
    ]

    for name, date, perf in scenarios:
        result = schedule_model_retraining(date, perf)
        print(f"\n{name}:")
        print(f"  Age        : {result['model_age_days']} hari")
        print(f"  AUC        : {result['current_auc']}")
        print(f"  Priority   : {result['retrain_priority']}")
        print(f"  Reason     : {result['reason']}")

    print("\n── 18.2 Model Registry ──")
    df_registry = list_model_registry()
    if not df_registry.empty:
        print(df_registry.to_string(index=False))
    else:
        print("Belum ada model tersimpan di registry.")

    # Demo save model dengan metadata
    print("\n── Demo Save Model with Metadata ──")
    with open("results/models/flight_risk_model.pkl", "rb") as f:
        model = pickle.load(f)

    metrics = {"roc_auc": 0.875, "precision": 0.82, "recall": 0.79, "f1_score": 0.80}
    features = ["job_level", "tenure_years", "salary_vs_dept_avg",
                "engagement_score", "avg_sentiment"]

    save_model_with_metadata(model, metrics, features)

    print("\n── Updated Registry ──")
    df_registry = list_model_registry()
    print(df_registry.to_string(index=False))