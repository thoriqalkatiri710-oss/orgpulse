import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 7.1.1 Knowledge Risk Assessment ──────────────────────────────────────────

def compute_knowledge_risk_matrix(df_employees: pd.DataFrame,
                                   df_centrality: pd.DataFrame,
                                   df_flight_risk: pd.DataFrame) -> pd.DataFrame:
    """
    Identifikasi Critical Knowledge Holders:
    Karyawan dengan influence score tinggi (ONA) DAN flight risk tinggi
    adalah titik risiko paling kritis bagi organisasi.
    """
    merged = df_employees.merge(
        df_centrality[["employee_id", "influence_score", "network_role"]],
        on="employee_id"
    ).merge(
        df_flight_risk[["employee_id", "flight_risk_score"]],
        on="employee_id"
    )

    # Knowledge Risk Score
    merged["knowledge_risk"] = (
        merged["influence_score"] * 0.50 +
        merged["flight_risk_score"] * 0.50
    ).round(4)

    merged["risk_quadrant"] = merged.apply(
        lambda r: (
            "🔴 Critical (High Influence + High Risk)"   if r["influence_score"] >= 0.5 and r["flight_risk_score"] >= 0.6 else
            "🟠 Watchlist (High Influence + Med Risk)"   if r["influence_score"] >= 0.5 and r["flight_risk_score"] >= 0.4 else
            "🟡 Monitor (Low Influence + High Risk)"     if r["influence_score"] < 0.5  and r["flight_risk_score"] >= 0.6 else
            "🟢 Stable"
        ), axis=1
    )

    return merged.sort_values("knowledge_risk", ascending=False)


def plot_risk_influence_matrix(df_matrix: pd.DataFrame, save_path: str = None):
    """
    Scatter plot influence score vs flight risk score.
    Kuadran: Critical, Watchlist, Monitor, Stable.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        "🔴 Critical (High Influence + High Risk)": "#EF4444",
        "🟠 Watchlist (High Influence + Med Risk)": "#F97316",
        "🟡 Monitor (Low Influence + High Risk)":   "#EAB308",
        "🟢 Stable":                                "#22C55E",
    }

    for quadrant, color in colors.items():
        subset = df_matrix[df_matrix["risk_quadrant"] == quadrant]
        ax.scatter(
            subset["flight_risk_score"], subset["influence_score"],
            c=color, label=quadrant, alpha=0.7, s=80,
            edgecolors="white", linewidth=0.5
        )

    # Label top-10 paling kritis
    for _, row in df_matrix.head(10).iterrows():
        ax.annotate(
            row["full_name"].split()[0],
            xy=(row["flight_risk_score"], row["influence_score"]),
            fontsize=7, ha="center", va="bottom"
        )

    ax.axvline(0.6, color="red",  linestyle="--", alpha=0.4)
    ax.axhline(0.5, color="blue", linestyle="--", alpha=0.4)
    ax.set_xlabel("Flight Risk Score (↑ = semakin berisiko resign)")
    ax.set_ylabel("Network Influence Score (↑ = semakin berpengaruh)")
    ax.set_title("Knowledge Risk Matrix\nPrioritisasi Intervensi Retensi dan Succession Planning")
    ax.legend(loc="upper left", fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig
# ── 7.3.1 Succession Planning ─────────────────────────────────────────────────

def identify_succession_candidates(df_employees: pd.DataFrame,
                                    df_flight_risk: pd.DataFrame,
                                    df_centrality: pd.DataFrame,
                                    target_dept: str,
                                    target_level: int,
                                    top_n: int = 3) -> pd.DataFrame:
    """
    Identifikasi kandidat internal untuk critical roles.
    Kriteria:
    1. Departemen sama, level 1-2 di bawah target
    2. Flight risk rendah (tidak akan resign sendiri)
    3. Engagement tinggi
    4. Influence score memadai
    5. Tenure cukup
    """
    candidates = df_employees[
        (df_employees["department"] == target_dept) &
        (df_employees["job_level"].between(target_level - 2, target_level - 1))
    ].copy()

    if candidates.empty:
        print(f"Tidak ada kandidat di {target_dept} untuk level {target_level}")
        return pd.DataFrame()

    candidates = candidates.merge(
        df_flight_risk[["employee_id", "flight_risk_score"]], on="employee_id"
    ).merge(
        df_centrality[["employee_id", "influence_score"]], on="employee_id"
    )

    max_tenure = candidates["tenure_years"].max()
    candidates["readiness_score"] = (
        (1 - candidates["flight_risk_score"]) * 0.30 +
        candidates["engagement_score"] / 5  * 0.35 +
        candidates["influence_score"]        * 0.20 +
        (candidates["tenure_years"] / (max_tenure + 1e-6)) * 0.15
    ).round(4)

    return candidates.nlargest(top_n, "readiness_score")[[
        "employee_id", "full_name", "job_level_name", "tenure_years",
        "engagement_score", "flight_risk_score", "influence_score", "readiness_score"
    ]]

if __name__ == "__main__":
    df_emp        = pd.read_csv("data/simulated/employees.csv")
    df_centrality = pd.read_csv("data/processed/centrality_scores.csv")
    df_risk       = pd.read_csv("data/processed/flight_risk_scores.csv")

    print("Computing knowledge risk matrix...")
    df_matrix = compute_knowledge_risk_matrix(df_emp, df_centrality, df_risk)

    print(f"\n── Risk Quadrant Distribution ──")
    print(df_matrix["risk_quadrant"].value_counts().to_string())

    print(f"\n── Top 10 Critical Knowledge Holders ──")
    print(df_matrix[["full_name", "department", "job_level_name",
                      "influence_score", "flight_risk_score", "knowledge_risk",
                      "risk_quadrant"]].head(10).to_string(index=False))

    df_matrix.to_csv("data/processed/knowledge_risk_matrix.csv", index=False)
    print("\n✅ Saved: knowledge_risk_matrix.csv")

    plot_risk_influence_matrix(
        df_matrix,
        save_path="results/figures/knowledge_risk_matrix.png"
    )

    # Succession planning
    print("\n── Succession Planning: Engineering Level 5 (Manager) ──")
    df_succession = identify_succession_candidates(
        df_emp, df_risk, df_centrality,
        target_dept="Engineering", target_level=5, top_n=3
    )
    print(df_succession.to_string(index=False))

    print("\n── Succession Planning: Sales Level 4 (Lead) ──")
    df_succession2 = identify_succession_candidates(
        df_emp, df_risk, df_centrality,
        target_dept="Sales", target_level=4, top_n=3
    )
    print(df_succession2.to_string(index=False))

    df_succession.to_csv("results/reports/succession_candidates.csv", index=False)
    print("\n✅ Saved: results/reports/succession_candidates.csv")