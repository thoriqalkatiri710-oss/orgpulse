import pandas as pd
import numpy as np


# ── 6.1.1 Flight Risk Feature Engineering ────────────────────────────────────

def build_flight_risk_features(df_employees: pd.DataFrame,
                                df_interactions: pd.DataFrame,
                                df_reviews: pd.DataFrame,
                                df_training: pd.DataFrame,
                                df_attendance: pd.DataFrame,
                                df_centrality: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan sinyal dari 5 sumber data menjadi feature matrix per karyawan.
    Total ~25 fitur dari: kompensasi, performa, absensi, network, training.
    Flight risk adalah fenomena multi-dimensional — satu metrik tidak cukup.
    """
    features = df_employees[[
        "employee_id", "department", "job_level", "tenure_years",
        "monthly_salary", "engagement_score", "age", "gender", "education"
    ]].copy()

    # ── SINYAL 1: Kompensasi relatif ──────────────────────────────────────────
    dept_avg_salary = df_employees.groupby("department")["monthly_salary"].mean()
    features["salary_vs_dept_avg"] = features.apply(
        lambda row: row["monthly_salary"] / dept_avg_salary.get(row["department"], 1),
        axis=1
    )

    # ── SINYAL 2: Performance trend ───────────────────────────────────────────
    review_features = df_reviews.groupby("employee_id").agg(
        latest_rating        =("rating", "last"),
        avg_rating           =("rating", "mean"),
        rating_trend         =("rating", lambda x: float(np.polyfit(range(len(x)), x, 1)[0]) if len(x) > 1 else 0),
        avg_sentiment        =("compound", "mean"),
        pct_negative_review  =("sentiment_label", lambda x: (x == "negative").mean()),
    ).reset_index()
    features = features.merge(review_features, on="employee_id", how="left")

    # ── SINYAL 3: Absensi ─────────────────────────────────────────────────────
    attendance_features = df_attendance.groupby("employee_id").agg(
        absent_rate      =("status", lambda x: (x != "present").mean()),
        avg_hours_worked =("hours_worked", "mean"),
        no_show_count    =("status", lambda x: (x == "no_show").sum()),
    ).reset_index()

    df_attendance["month"] = pd.to_datetime(df_attendance["date"]).dt.month
    late_months = df_attendance[df_attendance["month"] >= 10].groupby("employee_id").agg(
        absent_rate_q4=("status", lambda x: (x != "present").mean())
    ).reset_index()
    early_months = df_attendance[df_attendance["month"] <= 3].groupby("employee_id").agg(
        absent_rate_q1=("status", lambda x: (x != "present").mean())
    ).reset_index()

    absence_trend = late_months.merge(early_months, on="employee_id", how="outer")
    absence_trend["absence_worsening"] = (
        absence_trend["absent_rate_q4"].fillna(0) - absence_trend["absent_rate_q1"].fillna(0)
    )

    features = features.merge(attendance_features, on="employee_id", how="left")
    features = features.merge(absence_trend[["employee_id", "absence_worsening"]], on="employee_id", how="left")

    # ── SINYAL 4: Network isolation ───────────────────────────────────────────
    ona_features = df_centrality[[
        "employee_id", "influence_score", "in_degree_weighted_norm",
        "betweenness_norm", "network_role"
    ]].copy()
    ona_features["is_isolated"] = (ona_features["influence_score"] < 0.1).astype(int)
    features = features.merge(ona_features, on="employee_id", how="left")

    # ── SINYAL 5: Training completion ─────────────────────────────────────────
    training_features = df_training.groupby("employee_id").agg(
        avg_completion_rate   =("completion_rate", "mean"),
        n_trainings_completed =("completion_rate", lambda x: (x >= 0.8).sum()),
        avg_assessment_score  =("assessment_score", "mean"),
    ).reset_index()
    features = features.merge(training_features, on="employee_id", how="left")

    # Fill missing dengan median
    num_cols = features.select_dtypes(include=np.number).columns
    features[num_cols] = features[num_cols].fillna(features[num_cols].median())

    return features


if __name__ == "__main__":
    print("Loading data...")
    df_emp        = pd.read_csv("data/simulated/employees.csv")
    df_comm       = pd.read_csv("data/simulated/communications.csv")
    df_reviews    = pd.read_csv("data/processed/reviews_sentiment.csv")
    df_training   = pd.read_csv("data/simulated/training_data.csv")
    df_attendance = pd.read_csv("data/simulated/attendance.csv")
    df_centrality = pd.read_csv("data/processed/centrality_scores.csv")

    print("Building flight risk features...")
    df_features = build_flight_risk_features(
        df_emp, df_comm, df_reviews, df_training, df_attendance, df_centrality
    )

    print(f"\n── Feature Matrix ──")
    print(f"Shape  : {df_features.shape}")
    print(f"Columns: {df_features.columns.tolist()}")
    print(f"\nSample:\n{df_features.head(3).to_string()}")

    df_features.to_csv("data/processed/flight_risk_features.csv", index=False)
    print("\n✅ Saved: data/processed/flight_risk_features.csv")