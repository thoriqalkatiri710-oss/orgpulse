import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# ── 8.1.1 Compensation Equity Analysis ───────────────────────────────────────

def run_compensation_equity_model(df_employees: pd.DataFrame,
                                   df_centrality: pd.DataFrame) -> dict:
    """
    Regresi gaji dengan kontrol bertahap.
    Model 3 unik di OrgPulse: memasukkan influence_score dari ONA
    sebagai kontrol variabel — tidak mungkin dilakukan tanpa data jaringan.
    """
    df = df_employees.merge(
        df_centrality[["employee_id", "influence_score"]], on="employee_id"
    ).copy()
    df["log_salary"] = np.log(df["monthly_salary"])

    models = {}

    # Model 1: Gap mentah (tanpa kontrol)
    models["raw"] = smf.ols("log_salary ~ gender", data=df).fit()

    # Model 2: Kontrol jabatan + departemen
    models["controlled"] = smf.ols(
        "log_salary ~ gender + C(job_level) + C(department)", data=df
    ).fit()

    # Model 3: Kontrol penuh termasuk tenure + influence (unik OrgPulse)
    models["full_control"] = smf.ols(
        "log_salary ~ gender + C(job_level) + C(department) + tenure_years + influence_score",
        data=df
    ).fit()

    return models


def salary_benchmarking(df_employees: pd.DataFrame) -> pd.DataFrame:
    """Posisi gaji tiap karyawan relatif terhadap peer (dept + level)."""
    dept_level_stats = df_employees.groupby(["department", "job_level"]).agg(
        avg_salary  =("monthly_salary", "mean"),
        p25_salary  =("monthly_salary", lambda x: x.quantile(0.25)),
        p50_salary  =("monthly_salary", "median"),
        p75_salary  =("monthly_salary", lambda x: x.quantile(0.75)),
        n_employees =("monthly_salary", "count"),
    ).reset_index()

    df_with_bench = df_employees.merge(dept_level_stats, on=["department", "job_level"])

    df_with_bench["salary_position"] = df_with_bench.apply(
        lambda r: (
            "below_p25"  if r["monthly_salary"] < r["p25_salary"] else
            "p25_to_p50" if r["monthly_salary"] < r["p50_salary"] else
            "p50_to_p75" if r["monthly_salary"] < r["p75_salary"] else
            "above_p75"
        ), axis=1
    )
    df_with_bench["salary_gap_to_median"] = (
        df_with_bench["monthly_salary"] - df_with_bench["p50_salary"]
    ) / df_with_bench["p50_salary"]

    return df_with_bench


def estimate_retention_investment_needed(df_employees: pd.DataFrame,
                                          df_flight_risk: pd.DataFrame,
                                          target_reduction_pct: float = 0.5) -> pd.DataFrame:
    """
    ROI kalkulator retensi untuk presentasi ke CFO.
    "Investasi Rp X untuk mempertahankan karyawan berisiko
    menghemat Rp 3X dalam biaya replacement."
    Benchmark industri: replacement cost = 0.5–2x gaji tahunan.
    """
    merged    = df_employees.merge(df_flight_risk[["employee_id", "flight_risk_score"]])
    high_risk = merged[merged["flight_risk_score"] >= 0.6].copy()

    high_risk["salary_increase_needed"]    = high_risk["monthly_salary"] * 0.10
    high_risk["retention_bonus_estimated"] = high_risk["monthly_salary"] * 0.20
    high_risk["total_retention_cost"]      = (
        high_risk["salary_increase_needed"] * 12 + high_risk["retention_bonus_estimated"]
    )

    # Replacement cost: 6x gaji bulanan (benchmark industri)
    high_risk["replacement_cost"]          = high_risk["monthly_salary"] * 6
    high_risk["net_savings_if_retained"]   = (
        high_risk["replacement_cost"] - high_risk["total_retention_cost"]
    )

    return high_risk.sort_values("net_savings_if_retained", ascending=False)


if __name__ == "__main__":
    df_emp        = pd.read_csv("data/simulated/employees.csv")
    df_centrality = pd.read_csv("data/processed/centrality_scores.csv")
    df_risk       = pd.read_csv("data/processed/flight_risk_scores.csv")

    # ── Compensation Equity Model ──
    print("Running compensation equity models...")
    models = run_compensation_equity_model(df_emp, df_centrality)

    for name, model in models.items():
        gender_coef = model.params.get("gender[T.P]", model.params.get("gender[T.L]", None))
        if gender_coef is not None:
            gap_pct = (np.exp(gender_coef) - 1) * 100
            pval    = model.pvalues.get("gender[T.P]", model.pvalues.get("gender[T.L]", None))
            print(f"\n{name}: gender gap = {gap_pct:+.2f}% (p={pval:.3f})")
        print(f"  R² = {model.rsquared:.3f}")

    # ── Salary Benchmarking ──
    print("\n── Salary Position Distribution ──")
    df_bench = salary_benchmarking(df_emp)
    print(df_bench["salary_position"].value_counts().to_string())

    print(f"\n── Employees Below P25 per Department ──")
    below_p25 = df_bench[df_bench["salary_position"] == "below_p25"]
    print(below_p25.groupby("department").size().sort_values(ascending=False).to_string())

    df_bench.to_csv("data/processed/salary_benchmarking.csv", index=False)
    print("\n✅ Saved: salary_benchmarking.csv")

    # ── Retention ROI ──
    print("\n── Retention Investment ROI ──")
    df_roi = estimate_retention_investment_needed(df_emp, df_risk)
    print(f"High-risk employees (score ≥ 0.6): {len(df_roi)}")
    print(f"Total retention investment needed : Rp {df_roi['total_retention_cost'].sum():,.0f}")
    print(f"Total replacement cost if resigned: Rp {df_roi['replacement_cost'].sum():,.0f}")
    print(f"Net savings if retained           : Rp {df_roi['net_savings_if_retained'].sum():,.0f}")

    print(f"\n── Top 10 Highest ROI Retention ──")
    print(df_roi[["full_name", "department", "monthly_salary",
                  "total_retention_cost", "replacement_cost",
                  "net_savings_if_retained"]].head(10).to_string(index=False))

    df_roi.to_csv("results/reports/retention_roi.csv", index=False)
    print("\n✅ Saved: retention_roi.csv")