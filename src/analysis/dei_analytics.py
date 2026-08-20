import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 20.1.1 Representation Pipeline ───────────────────────────────────────────

def plot_representation_pipeline(df_employees: pd.DataFrame,
                                  group_col: str = "gender",
                                  save_path: str = None):
    """
    Visualisasi 'leaky pipeline' — apakah representasi perempuan
    di level bawah terbawa ke level atas?
    Panel kiri: jumlah absolut | Panel kanan: proporsi per level
    """
    pivot     = df_employees.groupby(["job_level", group_col]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel kiri: absolute count
    pivot.plot(kind="bar", ax=axes[0], colormap="Set2", edgecolor="white")
    axes[0].set_title("Jumlah Karyawan per Level × Gender")
    axes[0].set_xlabel("Job Level (1=Staff, 7=Director)")
    axes[0].tick_params(axis="x", rotation=0)

    # Panel kanan: proporsi (pipeline leak)
    pivot_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2", edgecolor="white")
    axes[1].set_title("Proporsi Gender per Level Jabatan\n(Leaky Pipeline Analysis)")
    axes[1].set_xlabel("Job Level")
    axes[1].set_ylabel("Persentase (%)")
    axes[1].axhline(50, color="black", linestyle="--", alpha=0.5, label="50% line")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend()

    fig.suptitle("DEI Pipeline Analysis — Representasi Gender", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


def compute_dei_metrics(df_employees: pd.DataFrame) -> dict:
    """Hitung metrik DEI utama: representasi, promotion parity, education gap."""
    gender_dist  = df_employees["gender"].value_counts(normalize=True)
    senior_mask  = df_employees["job_level"] >= 5
    senior_gender = df_employees[senior_mask]["gender"].value_counts(normalize=True)

    representation_ratio = senior_gender.get("P", 0) / gender_dist.get("P", 1e-6)

    return {
        "overall_female_pct":    round(gender_dist.get("P", 0) * 100, 1),
        "senior_female_pct":     round(senior_gender.get("P", 0) * 100, 1),
        "representation_ratio":  round(representation_ratio, 3),
        "pipeline_concern":      representation_ratio < 0.75,
        "education_gap":         df_employees.groupby("gender")["education"].apply(
                                     lambda x: (x == "S2").mean() * 100
                                 ).round(1).to_dict(),
    }


# ── 20.2.1 Intersectional Pay Analysis ───────────────────────────────────────

def intersectional_pay_analysis(df_employees: pd.DataFrame) -> pd.DataFrame:
    """
    Analisis gap gaji pada irisan beberapa dimensi sekaligus.
    Intersectionality: seseorang mungkin tidak mengalami diskriminasi
    berdasarkan satu dimensi saja, tapi berdasarkan kombinasi beberapa dimensi.
    Hanya group dengan n >= 5 untuk stabilitas estimasi.
    """
    df             = df_employees.copy()
    df["log_salary"] = np.log(df["monthly_salary"])

    groups = df.groupby(["department", "gender", "education"]).agg(
        avg_salary    =("monthly_salary", "mean"),
        median_salary =("monthly_salary", "median"),
        n             =("monthly_salary", "count"),
        avg_log_salary=("log_salary", "mean"),
    ).reset_index()

    groups = groups[groups["n"] >= 5].copy()

    # Bandingkan perempuan vs laki-laki sebagai baseline per dept+education
    male_ref = groups[groups["gender"] == "L"][
        ["department", "education", "avg_log_salary"]
    ].rename(columns={"avg_log_salary": "male_log_salary"})

    groups = groups[groups["gender"] == "P"].merge(
        male_ref, on=["department", "education"], how="inner"
    )

    groups["log_gap"]        = groups["avg_log_salary"] - groups["male_log_salary"]
    groups["pct_gap"]        = (np.exp(groups["log_gap"]) - 1) * 100
    groups["significant_gap"] = groups["pct_gap"] < -5  # >5% lebih rendah

    return groups.sort_values("pct_gap").reset_index(drop=True)


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")

    # ── DEI Metrics ──
    print("── DEI Metrics ──")
    metrics = compute_dei_metrics(df_emp)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    if metrics["pipeline_concern"]:
        print("\n⚠️  Pipeline concern terdeteksi — perempuan kurang terwakili di level senior")
    else:
        print("\n✅ Representasi gender di level senior relatif proporsional")

    # ── Pipeline Plot ──
    plot_representation_pipeline(
        df_emp,
        save_path="results/figures/dei_pipeline.png"
    )

    # ── Intersectional Pay Gap ──
    print("\n── Intersectional Pay Gap Analysis ──")
    df_gap = intersectional_pay_analysis(df_emp)

    if df_gap.empty:
        print("Tidak cukup data per segmen (n < 5) untuk analisis intersectional")
    else:
        print(f"Segmen yang dianalisis: {len(df_gap)}")
        print(f"Segmen dengan gap signifikan (>5%): {df_gap['significant_gap'].sum()}")
        print(f"\nTop 5 gap terbesar:")
        print(df_gap[["department", "education", "pct_gap", "n", "significant_gap"]].head(5).to_string(index=False))

    df_gap.to_csv("results/reports/dei_pay_gap.csv", index=False)
    print("\n✅ Saved: dei_pipeline.png & dei_pay_gap.csv")