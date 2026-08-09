import pandas as pd
import numpy as np
from faker import Faker
from datetime import date

fake = Faker("id_ID")

REVIEW_TEMPLATES = {
    "high_performer": [
        "{name} secara konsisten melampaui target yang ditetapkan. Kemampuan teknis dan kepemimpinan tim sangat menonjol. Rekomendasikan untuk promosi ke level berikutnya dalam 6 bulan ke depan.",
        "Performa {name} luar biasa sepanjang kuartal ini. Berhasil menyelesaikan proyek {project} lebih awal dari jadwal dengan kualitas melebihi ekspektasi. Kontribusi pada tim sangat signifikan.",
        "{name} menunjukkan growth mindset yang kuat. Inisiatif dalam mengembangkan proses baru berhasil meningkatkan efisiensi tim sebesar {pct}%. Sangat direkomendasikan untuk program fast-track.",
    ],
    "average_performer": [
        "{name} memenuhi sebagian besar target yang ditetapkan. Perlu pengembangan lebih lanjut dalam hal {skill}. Disarankan mengikuti pelatihan {training} untuk mendukung pertumbuhan karier.",
        "Secara keseluruhan {name} menunjukkan performa yang adequate. Beberapa deliverable perlu peningkatan dari sisi kualitas. Komunikasi dengan stakeholder perlu diperkuat.",
        "{name} telah bekerja dengan baik dalam tanggung jawab rutinnya. Area pengembangan utama: {area}. Dengan bimbingan yang tepat, {name} berpotensi mencapai level berikutnya.",
    ],
    "underperformer": [
        "{name} belum mencapai target yang ditetapkan. Perlu improvement plan yang lebih terstruktur dalam 3 bulan ke depan. Disarankan untuk diskusi 1-on-1 lebih intensif.",
        "Terdapat beberapa concern terkait performa {name}, khususnya dalam hal {weakness}. Perlu perhatian khusus dan rencana perbaikan yang konkret.",
        "{name} menghadapi tantangan dalam memenuhi ekspektasi peran saat ini. Perlu evaluasi ulang mengenai kesesuaian role dengan strengths yang dimiliki.",
    ],
    "flight_risk_signals": [
        "{name} beberapa kali mengekspresikan kekhawatiran terkait kompensasi dan jenjang karier. Disarankan untuk dilakukan retention conversation secepatnya.",
        "Dalam diskusi 1-on-1, {name} menyampaikan rasa kurang puas dengan beban kerja saat ini. Perlu perhatian HRD untuk mencegah potensi turnover.",
        "{name} tampak kurang engaged dalam beberapa bulan terakhir. Absensi dalam team meeting meningkat. Perlu investigasi lebih lanjut.",
    ]
}

SKILLS     = ["komunikasi", "analisis data", "project management", "leadership", "presentasi"]
TRAININGS  = ["leadership fundamental", "data analytics bootcamp", "communication workshop", "agile methodology"]
PROJECTS   = ["migrasi sistem", "peluncuran produk baru", "optimasi pipeline data", "restrukturisasi SOP"]
AREAS      = ["time management", "stakeholder communication", "technical depth", "cross-functional collaboration"]
WEAKNESSES = ["ketepatan waktu deliverable", "proaktivitas", "dokumentasi pekerjaan", "kolaborasi lintas tim"]


def generate_performance_reviews(df_employees: pd.DataFrame,
                                  n_cycles: int = 2,
                                  seed: int = 42) -> pd.DataFrame:
    """
    Generate teks performance review semi-natural per karyawan per siklus.
    n_cycles=2 → 2 review per karyawan (H1 dan H2 2024)
    Total rows = 1000 karyawan × 2 siklus = 2000 review
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    rows = []

    for _, emp in df_employees.iterrows():
        for cycle in range(1, n_cycles + 1):
            is_flight_risk = emp["is_flight_risk_sim"]
            engagement = emp["engagement_score"]

            if engagement >= 4.0:
                template_key = "high_performer"
                rating = int(rng.choice([4, 5], p=[0.3, 0.7]))
            elif engagement >= 2.5:
                template_key = "average_performer"
                rating = int(rng.choice([3, 4], p=[0.6, 0.4]))
            else:
                template_key = rng.choice(["underperformer", "flight_risk_signals"])
                rating = int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))

            # Inject flight risk signals ke teks review (40% chance)
            if is_flight_risk and rng.random() < 0.4:
                template_key = "flight_risk_signals"

            template = rng.choice(REVIEW_TEMPLATES[template_key])
            text = template.format(
                name=emp["full_name"].split()[0],
                skill=rng.choice(SKILLS),
                training=rng.choice(TRAININGS),
                project=rng.choice(PROJECTS),
                area=rng.choice(AREAS),
                weakness=rng.choice(WEAKNESSES),
                pct=int(rng.integers(5, 30)),
            )

            rows.append({
                "review_id":    f"REV{len(rows)+1:06d}",
                "employee_id":  emp["employee_id"],
                "review_cycle": f"2024-H{cycle}",
                "review_date":  date(2024, 6 * cycle, 15),
                "reviewer_id":  emp.get("manager_id", emp["employee_id"]),
                "rating":       rating,
                "review_text":  text,
                "word_count":   len(text.split()),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")
    print(f"Loaded {len(df_emp)} employees")

    df_reviews = generate_performance_reviews(df_emp, n_cycles=2)
    print(f"Total reviews: {len(df_reviews)}")
    print(f"\nRating distribution:\n{df_reviews['rating'].value_counts().sort_index()}")
    print(f"\nCycle distribution:\n{df_reviews['review_cycle'].value_counts()}")
    print(f"\nSample review:\n{df_reviews['review_text'].iloc[0]}")

    df_reviews.to_csv("data/simulated/performance_reviews.csv", index=False)
    print("\n✅ Saved: data/simulated/performance_reviews.csv")