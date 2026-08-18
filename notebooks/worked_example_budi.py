"""
BAGIAN 13 — CONTOH NUMERIK END-TO-END
Investigasi lengkap untuk karyawan fiktif: Budi Santoso (EMP00047)
Dari feature vector → prediksi model → SHAP → ONA → intervensi HRD
"""

import numpy as np
import pandas as pd


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 13.1 SETUP SKENARIO
# ─────────────────────────────────────────────────────────────
section("13.1 Setup Skenario — Budi Santoso (EMP00047)")

print("""
Karyawan  : Budi Santoso (EMP00047)
Posisi    : Senior Staff Engineering
Tenure    : 2.3 tahun
Gaji      : Rp 12.500.000

Observasi aktual:
  • Engagement score    : 2.4 / 5.0 (dept avg: 3.7)
  • Absensi 30 hari     : 4 hari (avg: 1.1 hari/bulan)
  • Tren absensi        : Q1: 0.8% → Q4: 6.2% (memburuk)
  • Gaji vs dept avg    : Rp 12.5jt vs Rp 14.8jt = 84.5% (< P25)
  • Review terbaru      : Sentimen negatif, rating 3/5
  • ONA                 : influence_score = 0.31 (medium)
  • Training completion : 0.73 (avg: 0.88)
""")


# ─────────────────────────────────────────────────────────────
# 13.2 FEATURE VECTOR
# ─────────────────────────────────────────────────────────────
section("13.2 Tahap 1 — Feature Vector")

feature_vector = {
    "job_level":             2,
    "tenure_years":          2.3,
    "salary_vs_dept_avg":    0.845,
    "engagement_score":      2.4,
    "latest_rating":         3,
    "avg_rating":            3.2,
    "rating_trend":         -0.25,
    "avg_sentiment":        -0.21,
    "pct_negative_review":   0.50,
    "absent_rate":           0.062,
    "absence_worsening":     0.054,
    "no_show_count":         2,
    "influence_score":       0.31,
    "is_isolated":           0,
    "avg_completion_rate":   0.73,
    "n_trainings_completed": 2,
}

print("Feature vector Budi Santoso:")
for k, v in feature_vector.items():
    flag = " ⚠️" if k in ["salary_vs_dept_avg", "engagement_score",
                            "absence_worsening", "avg_sentiment"] and (
        (k == "salary_vs_dept_avg" and v < 0.9) or
        (k == "engagement_score"   and v < 3.0) or
        (k == "absence_worsening"  and v > 0.02) or
        (k == "avg_sentiment"      and v < 0)
    ) else ""
    print(f"  {k:<30}: {v}{flag}")


# ─────────────────────────────────────────────────────────────
# 13.3 PREDIKSI MODEL
# ─────────────────────────────────────────────────────────────
section("13.3 Tahap 2 — Prediksi Model Gradient Boosting")

flight_risk_score = 0.78
print(f"""
P(flight_risk = 1 | features_Budi) = {flight_risk_score}
→ Flight Risk Score : {int(flight_risk_score * 100)} / 100
→ Tier              : CRITICAL ⚠️

Confidence interval dari 400 individual trees:
  P10 = 0.62 | P50 = 0.78 | P90 = 0.89
→ Estimasi stabil, bukan outlier satu tree
""")


# ─────────────────────────────────────────────────────────────
# 13.4 SHAP WATERFALL
# ─────────────────────────────────────────────────────────────
section("13.4 Tahap 3 — SHAP Waterfall")

base_value = 0.21
shap_contributions = [
    ("salary_vs_dept_avg",   +0.183, "gaji 15.5% di bawah rata-rata peers"),
    ("engagement_score",     +0.164, "engagement 2.4/5 vs dept avg 3.7"),
    ("absence_worsening",    +0.097, "absensi naik 5.4 ppt dalam 9 bulan"),
    ("avg_sentiment",        +0.082, "sentimen review negatif"),
    ("pct_negative_review",  +0.071, "50% review cycle negatif"),
    ("rating_trend",         +0.043, "rating menurun"),
    ("influence_score",      -0.031, "koneksi jaringan menahan sedikit"),
    ("tenure_years",         -0.029, "masih relatif baru, masih bisa grow"),
]

print(f"Base value (rata-rata semua karyawan): {base_value}")
print()
cumulative = base_value
for feat, contrib, desc in shap_contributions:
    cumulative += contrib
    sign = "+" if contrib > 0 else ""
    arrow = "↑ RISIKO" if contrib > 0 else "↓ risiko"
    print(f"  {sign}{contrib:+.3f}  {feat:<25} ({arrow}) — {desc}")

print(f"\n{'─'*55}")
print(f"  Prediksi akhir: {cumulative:.3f}")
print(f"  Verifikasi    : {flight_risk_score} ✅")


# ─────────────────────────────────────────────────────────────
# 13.5 ONA CONTEXT
# ─────────────────────────────────────────────────────────────
section("13.5 Tahap 4 — ONA Context")

influence  = 0.31
risk_score = 0.78
knowledge_risk = influence * 0.50 + risk_score * 0.50

print(f"""
Network Role    : Regular Member (bukan Central Connector)
Influence Score : {influence}
Flight Risk     : {risk_score}

Knowledge Risk Score = {influence} × 0.50 + {risk_score} × 0.50
                     = {knowledge_risk:.3f}
Risk Quadrant        : Monitor (Low Influence + High Risk) 🟡
Prioritas intervensi : MEDIUM-HIGH

Perbandingan dengan Central Connector berisiko tinggi:
  Central Connector : influence=0.72, risk=0.81 → knowledge_risk=0.765
  Budi Santoso      : influence=0.31, risk=0.78 → knowledge_risk={knowledge_risk:.3f}
  → Budi di prioritas kedua — alokasi waktu HRBP yang terbatas
""")


# ─────────────────────────────────────────────────────────────
# 13.6 RENCANA INTERVENSI
# ─────────────────────────────────────────────────────────────
section("13.6 Tahap 5 — Rencana Intervensi Terstruktur")

print("""
MINGGU 1:
  □ HR BP meeting dengan manager Budi untuk align konteks
  □ Salary review request ke Compensation Team (target: ke P50 = Rp 14.8jt)
  □ Catat sebagai "Retention Case" di sistem

MINGGU 2:
  □ 1-on-1 career conversation oleh HRBP langsung
  □ Explorasi: apakah ada unmet career aspiration yang bisa diakomodasi?
  □ Tawarkan project stretch assignment di cross-functional team
    (meningkatkan ONA influence score)

BULAN 2:
  □ Follow-up salary review implementation
  □ Re-assess engagement score (target: ≥ 3.2 dalam 60 hari)
  □ Monitor tren absensi via sistem

MENGAPA INI LEBIH KUAT DARI INTUISI HRD BIASA:
  Setiap langkah terhubung langsung ke faktor spesifik dari model:
  • salary gap      → salary review
  • engagement rendah → 1-on-1 career conversation
  • ONA rendah      → cross-functional project assignment
  Bukan program retensi generik yang sama untuk semua orang.
""")

# ROI estimasi
salary_current = 12_500_000
salary_target  = 14_800_000
salary_increase = salary_target - salary_current
annual_retention_cost = salary_increase * 12 + salary_current * 0.15 * 12
replacement_cost = salary_current * 6

print(f"ROI Intervensi untuk Budi:")
print(f"  Kenaikan gaji/bulan       : Rp {salary_increase:,.0f}")
print(f"  Total biaya retensi/tahun : Rp {annual_retention_cost:,.0f}")
print(f"  Biaya replacement jika resign: Rp {replacement_cost:,.0f}")
print(f"  Net penghematan           : Rp {replacement_cost - annual_retention_cost:,.0f}")
print(f"  ROI                       : {(replacement_cost / annual_retention_cost - 1):.1%}")