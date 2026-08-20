import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# ── 19.1.1 Training Impact (DiD) ─────────────────────────────────────────────

def measure_training_impact(df_training: pd.DataFrame,
                             df_reviews: pd.DataFrame,
                             training_category: str = "leadership") -> dict:
    """
    Difference-in-Differences estimator untuk mengukur dampak training.
    Bandingkan rating H1 (sebelum) vs H2 (sesudah) untuk trained vs control group.
    Koefisien interaksi trained×period = efek bersih training.
    """
    trained     = df_training[df_training["category"] == training_category]
    trained_ids = trained["employee_id"].unique()

    reviews          = df_reviews.copy()
    reviews["period"] = reviews["review_cycle"].map({"2024-H1": 0, "2024-H2": 1})
    reviews["trained"] = reviews["employee_id"].isin(trained_ids).astype(int)
    reviews           = reviews.dropna(subset=["period"])

    model        = smf.ols("rating ~ trained * period + C(period)", data=reviews).fit()
    did_estimate = model.params.get("trained:period", 0)
    p_value      = model.pvalues.get("trained:period", 1)

    return {
        "training_category": training_category,
        "n_trained":         len(trained_ids),
        "did_estimate":      round(did_estimate, 3),
        "p_value":           round(p_value, 4),
        "significant":       p_value < 0.05,
        "interpretation": (
            f"Training '{training_category}' meningkatkan rating rata-rata "
            f"{did_estimate:+.3f} poin dibanding kelompok kontrol"
            if p_value < 0.05 else
            f"Belum ada bukti signifikan dampak training '{training_category}' terhadap rating"
        )
    }


def training_roi_analysis(df_training: pd.DataFrame,
                           df_reviews: pd.DataFrame,
                           cost_per_hour: float = 500_000) -> pd.DataFrame:
    """ROI analisis per kategori training — siap untuk presentasi ke CFO."""
    categories = df_training["category"].unique()
    results    = []

    for cat in categories:
        impact   = measure_training_impact(df_training, df_reviews, cat)
        cat_data = df_training[df_training["category"] == cat]
        total_cost = cat_data["duration_hours"].sum() * cost_per_hour

        results.append({
            "category":           cat,
            "n_participants":     impact["n_trained"],
            "total_hours":        cat_data["duration_hours"].sum(),
            "total_cost_idr":     total_cost,
            "did_rating_impact":  impact["did_estimate"],
            "significant":        impact["significant"],
            "p_value":            impact["p_value"],
            "cost_per_rating_pt": round(total_cost / (abs(impact["did_estimate"]) + 1e-6), 0),
        })

    return pd.DataFrame(results).sort_values("did_rating_impact", ascending=False)


# ── 19.2.1 Skills Gap Analysis ────────────────────────────────────────────────

FUTURE_SKILLS_REQUIREMENT = {
    "Engineering":      {"Python/ML": 0.80, "Cloud": 0.70, "System Design": 0.60},
    "Data & Analytics": {"ML/AI": 0.90, "SQL Advanced": 0.85, "Causal Inference": 0.50},
    "Product":          {"Data-Driven PM": 0.75, "UX Research": 0.65, "Agile": 0.80},
    "Marketing":        {"Digital Marketing": 0.80, "Analytics": 0.70, "Content": 0.60},
    "HR":               {"People Analytics": 0.70, "HRIS": 0.65, "Employment Law": 0.75},
}

TRAINING_SKILL_MAP = {
    "Python for Data":        "Python/ML",
    "SQL Advanced":           "SQL Advanced",
    "Cloud Computing":        "Cloud",
    "Leadership Fundamental": "Data-Driven PM",
    "Data Analytics Bootcamp":"Analytics",
}


def compute_skills_gap(df_training: pd.DataFrame,
                        df_employees: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung % karyawan per dept yang sudah punya skill tertentu vs yang diperlukan.
    Output menjadi basis perencanaan L&D — bukan berdasarkan training populer,
    tapi skill yang paling dibutuhkan organisasi 2 tahun ke depan.
    """
    training_done          = df_training[df_training["completion_rate"] >= 0.8].copy()
    training_done["skill"] = training_done["training_name"].map(TRAINING_SKILL_MAP)

    gap_records = []
    for dept, skill_reqs in FUTURE_SKILLS_REQUIREMENT.items():
        dept_emp = df_employees[df_employees["department"] == dept]["employee_id"].tolist()
        n_dept   = len(dept_emp)
        if n_dept == 0:
            continue

        for skill, required_coverage in skill_reqs.items():
            skilled_emp = training_done[
                (training_done["employee_id"].isin(dept_emp)) &
                (training_done["skill"] == skill)
            ]["employee_id"].nunique()

            current_coverage = skilled_emp / n_dept
            gap_records.append({
                "department":       dept,
                "skill":            skill,
                "required_pct":     required_coverage,
                "current_pct":      round(current_coverage, 3),
                "gap_pct":          round(required_coverage - current_coverage, 3),
                "n_need_training":  max(0, int((required_coverage - current_coverage) * n_dept)),
            })

    df_gap = pd.DataFrame(gap_records)
    df_gap["priority"] = pd.cut(
        df_gap["gap_pct"],
        bins=[-1, 0.10, 0.25, 0.50, 1.0],
        labels=["OK", "LOW", "MEDIUM", "HIGH"]
    )
    return df_gap.sort_values("gap_pct", ascending=False)


if __name__ == "__main__":
    df_training = pd.read_csv("data/simulated/training_data.csv")
    df_reviews  = pd.read_csv("data/processed/reviews_sentiment.csv")
    df_emp      = pd.read_csv("data/simulated/employees.csv")

    # ── Training ROI ──
    print("── Training ROI Analysis (DiD) ──")
    df_roi = training_roi_analysis(df_training, df_reviews)
    print(df_roi[["category", "n_participants", "total_cost_idr",
                  "did_rating_impact", "significant", "p_value"]].to_string(index=False))

    print("\n── Interpretasi per Kategori ──")
    for cat in df_training["category"].unique():
        result = measure_training_impact(df_training, df_reviews, cat)
        print(f"  {result['interpretation']}")

    df_roi.to_csv("results/reports/training_roi.csv", index=False)
    print("\n✅ Saved: training_roi.csv")

    # ── Skills Gap ──
    print("\n── Skills Gap Analysis ──")
    df_gap = compute_skills_gap(df_training, df_emp)
    print(df_gap.to_string(index=False))

    high_priority = df_gap[df_gap["priority"] == "HIGH"]
    print(f"\nSkill gaps prioritas HIGH: {len(high_priority)}")
    print(f"Total karyawan perlu training: {high_priority['n_need_training'].sum()}")

    df_gap.to_csv("results/reports/skills_gap_analysis.csv", index=False)
    print("\n✅ Saved: skills_gap_analysis.csv")