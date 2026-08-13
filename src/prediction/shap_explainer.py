import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ── 6.3.1 Global SHAP ────────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Hitung SHAP values menggunakan TreeExplainer (cepat untuk tree-based models)."""
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    """Global feature importance — rata-rata kontribusi tiap fitur."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.title("Global Feature Importance (SHAP)\nFaktor Paling Berpengaruh pada Prediksi Flight Risk")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


# ── 6.3.2 Individual Explanation ─────────────────────────────────────────────

def explain_individual_risk(shap_values, X: pd.DataFrame,
                             df_employees: pd.DataFrame,
                             employee_id: str, save_path: str = None):
    """
    Waterfall plot untuk satu karyawan — 'kenapa karyawan ini berisiko?'
    Merah = meningkatkan risiko, Biru = menurunkan risiko.
    """
    if employee_id not in X.index:
        print(f"Employee {employee_id} tidak ditemukan.")
        return None

    row_idx = list(X.index).index(employee_id)
    emp_name = df_employees[df_employees["employee_id"] == employee_id]["full_name"].values
    name = emp_name[0] if len(emp_name) else employee_id

    fig = plt.figure(figsize=(12, 6))
    shap.waterfall_plot(shap_values[row_idx], show=False, max_display=10)
    plt.title(f"SHAP Waterfall — Penjelasan Flight Risk: {name}\n"
              f"(Merah = meningkatkan risiko, Biru = menurunkan risiko)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


def generate_risk_narrative(shap_values, X: pd.DataFrame,
                             df_employees: pd.DataFrame,
                             employee_id: str,
                             threshold: float = 0.6) -> str:
    """Generate narasi otomatis penjelasan flight risk per individu."""
    row_idx  = list(X.index).index(employee_id)
    shap_row = pd.Series(shap_values.values[row_idx], index=X.columns)

    top_positive = shap_row[shap_row > 0].nlargest(3)
    top_negative = shap_row[shap_row < 0].nsmallest(2)

    emp       = df_employees[df_employees["employee_id"] == employee_id].iloc[0]
    risk_prob = float(shap_values.base_values[row_idx] + shap_row.sum())
    risk_prob = max(0.0, min(1.0, risk_prob))

    status = "⚠️ TINGGI" if risk_prob >= threshold else "✅ NORMAL"

    narrative = f"""
RINGKASAN FLIGHT RISK — {emp['full_name']} ({employee_id})
Departemen : {emp['department']} | Level: {emp['job_level_name']} | Tenure: {emp['tenure_years']:.1f} tahun
Skor Risiko: {risk_prob:.1%} {status}

FAKTOR RISIKO UTAMA (meningkatkan kemungkinan resign):
{chr(10).join(f"  • {feat}: kontribusi +{val:.3f}" for feat, val in top_positive.items())}

FAKTOR PROTEKTIF (menurunkan kemungkinan resign):
{chr(10).join(f"  • {feat}: kontribusi {val:.3f}" for feat, val in top_negative.items())}

REKOMENDASI INTERVENSI HRD:
{generate_intervention_recommendation(top_positive.index.tolist(), emp)}
""".strip()

    return narrative


def generate_intervention_recommendation(risk_factors: list, emp: pd.Series) -> str:
    """Generate rekomendasi intervensi berdasarkan faktor risiko dominan."""
    interventions = []

    if "salary_vs_dept_avg" in risk_factors:
        interventions.append("→ Tinjau kompensasi vs pasar — gaji di bawah rata-rata departemen")
    if "engagement_score" in risk_factors:
        interventions.append("→ Jadwalkan career conversation 1-on-1 dengan manager")
    if "absence_worsening" in risk_factors:
        interventions.append("→ Periksa wellbeing — tren absensi memburuk")
    if "is_isolated" in risk_factors:
        interventions.append("→ Libatkan dalam cross-functional project untuk meningkatkan koneksi")
    if "pct_negative_review" in risk_factors:
        interventions.append("→ Review ulang performance feedback — banyak sentimen negatif")
    if "avg_sentiment" in risk_factors:
        interventions.append("→ Diskusikan kepuasan kerja — sentimen review sangat negatif")
    if not interventions:
        interventions.append("→ Monitor melalui check-in reguler")

    return "\n".join(interventions)

# ── 6.4.1 Risk Heatmap ───────────────────────────────────────────────────────

def plot_risk_heatmap(df_scores: pd.DataFrame, df_employees: pd.DataFrame,
                      save_path: str = None):
    """
    Heatmap flight risk per departemen × job level.
    Merah = risiko tinggi, Hijau = risiko rendah.
    """
    import seaborn as sns

    merged = df_scores.merge(
        df_employees[["employee_id", "department", "job_level"]],
        on="employee_id"
    )
    pivot = merged.pivot_table(
        values="flight_risk_score",
        index="department",
        columns="job_level",
        aggfunc="mean"
    ).round(3)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Flight Risk Heatmap — Rata-rata Skor Risiko per Departemen × Level Jabatan\n"
                 "(Merah = Risiko Tinggi, Hijau = Risiko Rendah)")
    ax.set_xlabel("Job Level")
    ax.set_ylabel("Departemen")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    import pickle
    from src.prediction.flight_risk_model import prepare_features

    print("Loading data...")
    df_features = pd.read_csv("data/processed/flight_risk_features.csv")
    df_emp      = pd.read_csv("data/simulated/employees.csv")

    df_features = df_features.merge(
        df_emp[["employee_id", "is_flight_risk_sim"]], on="employee_id", how="left"
    )

    X, y = prepare_features(df_features)
    X.index = df_features["employee_id"].values

    with open("results/models/flight_risk_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X)

    # Global summary plot
    plot_shap_summary(shap_values, X,
                      save_path="results/figures/shap_summary.png")

    # Individual explanation — ambil karyawan flight risk tertinggi
    risk_scores = pd.read_csv("data/processed/flight_risk_scores.csv")
    top_risk_emp = risk_scores.sort_values("flight_risk_score", ascending=False).iloc[0]
    emp_id = top_risk_emp["employee_id"]

    print(f"\nExplaining: {emp_id} (risk score: {top_risk_emp['flight_risk_score']:.3f})")

    explain_individual_risk(
        shap_values, X, df_emp, emp_id,
        save_path=f"results/figures/shap_waterfall_{emp_id}.png"
    )

    narrative = generate_risk_narrative(shap_values, X, df_emp, emp_id)
    print(f"\n{narrative}")

    # Simpan narrative ke file
    with open(f"results/reports/flight_risk_narrative_{emp_id}.txt", "w", encoding="utf-8") as f:
        f.write(narrative)
    print(f"\n✅ Narrative saved: results/reports/flight_risk_narrative_{emp_id}.txt")

    # Risk heatmap
    print("\nGenerating risk heatmap...")
    plot_risk_heatmap(
        risk_scores, df_emp,
        save_path="results/figures/flight_risk_heatmap.png"
    )