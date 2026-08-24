import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── 16.1.1 Cohort Retention Analysis ─────────────────────────────────────────

def build_cohort_retention_matrix(df_employees: pd.DataFrame,
                                   observation_end: str = "2024-12-31") -> pd.DataFrame:
    """
    Bangun matrix retensi per cohort rekrutmen tahunan.
    Menjawab: apakah kualitas rekrutmen membaik dari tahun ke tahun?
    Cohort mana yang paling cepat meninggalkan perusahaan?
    """
    df       = df_employees.copy()
    end_date = pd.Timestamp(observation_end)

    df["hire_year"]    = pd.to_datetime(df["hire_date"]).dt.year
    df["hire_quarter"] = pd.to_datetime(df["hire_date"]).dt.to_period("Q")

    retention_records = []

    for cohort_year, cohort_df in df.groupby("hire_year"):
        n_start = len(cohort_df)

        for months_after in [3, 6, 12, 18, 24, 36]:
            cutoff = pd.Timestamp(f"{cohort_year}-01-01") + pd.DateOffset(months=months_after)
            if cutoff > end_date:
                break

            # Karyawan "masih aktif" jika bukan flight risk ATAU tenure > months_after
            still_active = cohort_df[
                ~cohort_df["is_flight_risk_sim"] |
                (cohort_df["tenure_years"] * 12 > months_after)
            ]

            retention_records.append({
                "cohort_year":       cohort_year,
                "months_after_hire": months_after,
                "n_retained":        len(still_active),
                "retention_rate":    round(len(still_active) / n_start, 3),
            })

    return pd.DataFrame(retention_records)


def plot_cohort_heatmap(cohort_matrix: pd.DataFrame, save_path: str = None):
    """
    Heatmap retensi cohort — merah = retensi rendah, hijau = retensi tinggi.
    Insight: cohort mana yang early exit, tren perbaikan/penurunan rekrutmen.
    """
    pivot = cohort_matrix.pivot(
        index="cohort_year",
        columns="months_after_hire",
        values="retention_rate"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn",
                vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("Cohort Retention Heatmap\n(% Karyawan Masih Aktif N Bulan Setelah Bergabung)")
    ax.set_xlabel("Bulan Setelah Tanggal Bergabung")
    ax.set_ylabel("Cohort Tahun Masuk")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── 16.2.1 New Hire Ramp Time ─────────────────────────────────────────────────

def analyze_new_hire_ramp_time(df_employees: pd.DataFrame,
                                df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Analisis waktu new hire mencapai full productivity.
    Ramp time = bulan dari hire_date sampai rating ≥ 4.
    """
    new_hires        = df_employees[df_employees["tenure_years"] <= 1.5].copy()
    new_hire_reviews = df_reviews[df_reviews["employee_id"].isin(new_hires["employee_id"])]

    ramp_analysis = new_hire_reviews.merge(
        new_hires[["employee_id", "department", "job_level", "hire_date"]]
    )
    ramp_analysis["months_since_hire"] = (
        (pd.to_datetime(ramp_analysis["review_date"]) -
         pd.to_datetime(ramp_analysis["hire_date"])).dt.days / 30.44
    )

    dept_ramp = ramp_analysis.groupby(["department", "months_since_hire"]).agg(
        avg_rating=("rating", "mean"),
        n_reviews =("review_id", "count"),
    ).reset_index()

    return dept_ramp


if __name__ == "__main__":
    print("Loading data...")
    df_emp     = pd.read_csv("data/simulated/employees.csv")
    df_reviews = pd.read_csv("data/processed/reviews_sentiment.csv")

    # ── Cohort Retention ──
    print("\nBuilding cohort retention matrix...")
    cohort_matrix = build_cohort_retention_matrix(df_emp)

    print(f"\n── Cohort Retention Matrix ──")
    pivot = cohort_matrix.pivot(
        index="cohort_year", columns="months_after_hire", values="retention_rate"
    )
    print(pivot.to_string())

    plot_cohort_heatmap(
        cohort_matrix,
        save_path="results/figures/cohort_retention_heatmap.png"
    )

    cohort_matrix.to_csv("data/processed/cohort_retention_matrix.csv", index=False)

    # ── New Hire Ramp Time ──
    print("\nAnalyzing new hire ramp time...")
    dept_ramp = analyze_new_hire_ramp_time(df_emp, df_reviews)

    print(f"\n── New Hire Ramp Time per Department ──")
    print(dept_ramp.groupby("department")["avg_rating"].mean().sort_values(ascending=False).round(2).to_string())

    dept_ramp.to_csv("data/processed/new_hire_ramp_analysis.csv", index=False)
    print("\n✅ Saved: cohort_retention_matrix.csv & new_hire_ramp_analysis.csv")