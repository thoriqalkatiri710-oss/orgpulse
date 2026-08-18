import pandas as pd
import numpy as np


# ── 17.2.1 Fairness Audit ─────────────────────────────────────────────────────

def audit_model_fairness(df_employees: pd.DataFrame,
                          df_flight_risk: pd.DataFrame) -> dict:
    """
    Disparate Impact Ratio (DIR): rasio tingkat prediksi positif antar kelompok.
    DIR < 0.80: potensi adverse impact (standar referensi industri EEOC).
    Audit ini wajib dijalankan setiap 6 bulan di production.
    """
    merged = df_employees.merge(
        df_flight_risk[["employee_id", "flight_risk_score"]], on="employee_id"
    )
    merged["predicted_high_risk"] = (merged["flight_risk_score"] >= 0.6).astype(int)

    audit_results = {}

    # ── Cek per gender ────────────────────────────────────────────────────────
    gender_rates = merged.groupby("gender")["predicted_high_risk"].mean()
    if len(gender_rates) >= 2:
        dir_gender = gender_rates.min() / gender_rates.max()
        audit_results["gender_rates"]   = gender_rates.round(3).to_dict()
        audit_results["gender_dir"]     = round(dir_gender, 3)
        audit_results["gender_concern"] = dir_gender < 0.80

    # ── Cek per usia ──────────────────────────────────────────────────────────
    merged["age_group"] = pd.cut(
        merged["age"], bins=[0, 30, 40, 999], labels=["<30", "30-40", ">40"]
    )
    age_rates = merged.groupby("age_group", observed=True)["predicted_high_risk"].mean()
    dir_age   = age_rates.min() / age_rates.max()
    audit_results["age_rates"]   = age_rates.round(3).to_dict()
    audit_results["age_dir"]     = round(dir_age, 3)
    audit_results["age_concern"] = dir_age < 0.80

    # ── Cek per education ─────────────────────────────────────────────────────
    edu_rates = merged.groupby("education")["predicted_high_risk"].mean()
    dir_edu   = edu_rates.min() / edu_rates.max()
    audit_results["education_rates"]   = edu_rates.round(3).to_dict()
    audit_results["education_dir"]     = round(dir_edu, 3)
    audit_results["education_concern"] = dir_edu < 0.80

    return audit_results


def print_fairness_report(audit_results: dict):
    """Print laporan fairness yang siap dikomunikasikan ke leadership."""
    print("\n── Fairness Audit Report ──")
    print(f"Threshold: DIR < 0.80 = potensi adverse impact\n")

    checks = [
        ("Gender",    "gender_dir",    "gender_concern",    "gender_rates"),
        ("Age Group", "age_dir",       "age_concern",       "age_rates"),
        ("Education", "education_dir", "education_concern", "education_rates"),
    ]

    all_pass = True
    for label, dir_key, concern_key, rates_key in checks:
        dir_val  = audit_results.get(dir_key, None)
        concern  = audit_results.get(concern_key, False)
        rates    = audit_results.get(rates_key, {})

        status = "⚠️ PERLU INVESTIGASI" if concern else "✅ LULUS"
        if concern:
            all_pass = False

        print(f"{label}:")
        print(f"  DIR           : {dir_val:.3f} {status}")
        print(f"  Rate per grup : {rates}")
        print()

    print("─" * 50)
    if all_pass:
        print("✅ Model lulus semua fairness checks — tidak ada adverse impact signifikan")
    else:
        print("⚠️  Ada dimensi yang perlu investigasi lebih lanjut")


if __name__ == "__main__":
    df_emp  = pd.read_csv("data/simulated/employees.csv")
    df_risk = pd.read_csv("data/processed/flight_risk_scores.csv")

    print("Running fairness audit...")
    audit_results = audit_model_fairness(df_emp, df_risk)
    print_fairness_report(audit_results)

    # Simpan hasil audit
    import json
    with open("results/reports/fairness_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2, default=str)
    print("\n✅ Saved: results/reports/fairness_audit.json")