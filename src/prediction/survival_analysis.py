import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter


# ── 14.1.1 Survival Dataset ───────────────────────────────────────────────────

def prepare_survival_dataset(df_employees: pd.DataFrame,
                              df_attendance: pd.DataFrame,
                              df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun dataset survival dari data HR.
    - duration_months: tenure karyawan dalam bulan
    - event_observed : 1 = resign (flight risk), 0 = masih aktif (censored)
    Survival analysis menjawab: KAPAN puncak risiko terjadi,
    bukan hanya apakah berisiko.
    """
    df = df_employees.copy()
    df["duration_months"]  = (df["tenure_years"] * 12).round(1)
    df["event_observed"]   = df["is_flight_risk_sim"].astype(int)

    # Kovariate untuk Cox model
    df["low_engagement"]      = (df["engagement_score"] < 3.0).astype(int)
    df["below_median_salary"] = (df["monthly_salary"] < df["monthly_salary"].median()).astype(int)
    df["senior"]              = (df["job_level"] >= 5).astype(int)

    return df


# ── Kaplan-Meier ──────────────────────────────────────────────────────────────

def fit_kaplan_meier_by_segment(df_survival: pd.DataFrame,
                                 save_path: str = None):
    """
    Kurva survival per segmen engagement.
    Menunjukkan kapan probabilitas bertahan mulai turun signifikan.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mask, color in [
        ("Low Engagement (< 3.0)",  df_survival["low_engagement"] == 1, "#EF4444"),
        ("High Engagement (≥ 3.0)", df_survival["low_engagement"] == 0, "#22C55E"),
    ]:
        subset = df_survival[mask]
        kmf    = KaplanMeierFitter()
        kmf.fit(subset["duration_months"], subset["event_observed"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_title("Kurva Survival Karyawan per Segmen Engagement\n(Kaplan-Meier Estimator)")
    ax.set_xlabel("Masa Kerja (bulan)")
    ax.set_ylabel("Probabilitas Masih Aktif")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── Cox Proportional Hazards ──────────────────────────────────────────────────

def fit_cox_model_hr(df_survival: pd.DataFrame) -> CoxPHFitter:
    """
    Cox PH model untuk mengidentifikasi faktor yang mempercepat/memperlambat resign.
    Output: hazard ratio per variabel — interpretable untuk HR leadership.
    """
    cox_features = ["duration_months", "event_observed",
                    "low_engagement", "below_median_salary",
                    "senior", "tenure_years"]
    df_cox = df_survival[cox_features].dropna()

    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="duration_months", event_col="event_observed")
    return cph


def interpret_cox_hr(cph: CoxPHFitter) -> pd.DataFrame:
    """Terjemahkan output Cox ke bahasa HR yang mudah dipahami."""
    summary = cph.summary[["coef", "exp(coef)", "p",
                             "exp(coef) lower 95%", "exp(coef) upper 95%"]].copy()
    summary["hr_interpretation"] = summary["exp(coef)"].apply(
        lambda hr: (
            f"Risiko resign {abs(1-hr)*100:.0f}% "
            f"{'lebih tinggi' if hr > 1 else 'lebih rendah'} vs baseline"
        )
    )
    return summary.sort_values("p")


if __name__ == "__main__":
    print("Loading data...")
    df_emp        = pd.read_csv("data/simulated/employees.csv")
    df_attendance = pd.read_csv("data/simulated/attendance.csv")
    df_reviews    = pd.read_csv("data/processed/reviews_sentiment.csv")

    print("Preparing survival dataset...")
    df_survival = prepare_survival_dataset(df_emp, df_attendance, df_reviews)

    print(f"\n── Survival Dataset ──")
    print(f"Total karyawan      : {len(df_survival)}")
    print(f"Event (resign)      : {df_survival['event_observed'].sum()} ({df_survival['event_observed'].mean():.1%})")
    print(f"Censored (aktif)    : {(df_survival['event_observed'] == 0).sum()}")
    print(f"Median duration     : {df_survival['duration_months'].median():.1f} bulan")

    # Kaplan-Meier
    print("\nFitting Kaplan-Meier curves...")
    fit_kaplan_meier_by_segment(
        df_survival,
        save_path="results/figures/survival_kaplan_meier.png"
    )

    # Cox PH
    print("\nFitting Cox Proportional Hazards model...")
    cph = fit_cox_model_hr(df_survival)

    print("\n── Cox Model Summary ──")
    cph.print_summary()

    print("\n── HR Interpretation ──")
    interpretation = interpret_cox_hr(cph)
    print(interpretation[["exp(coef)", "p", "hr_interpretation"]].to_string())

    # Save
    df_survival.to_csv("data/processed/survival_dataset.csv", index=False)
    print("\n✅ Saved: survival_dataset.csv")