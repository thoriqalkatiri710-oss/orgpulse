import pandas as pd
import numpy as np

TRAINING_CATALOG = {
    "technical":   ["Python for Data", "SQL Advanced", "Cloud Computing", "Cybersecurity Basics"],
    "leadership":  ["Leadership Fundamental", "Coaching Skills", "Strategic Thinking", "Change Management"],
    "soft_skills": ["Communication", "Presentation", "Negotiation", "Time Management"],
    "compliance":  ["Anti-Fraud", "Data Privacy", "Code of Conduct", "Health & Safety"],
}


def generate_training_data(df_employees: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate data pelatihan per karyawan.
    1-5 training per karyawan → total ~3000 rows untuk 1000 karyawan.
    completion_rate: distribusi beta right-skewed (kebanyakan >80%).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _, emp in df_employees.iterrows():
        n_trainings = int(rng.integers(1, 6))
        for _ in range(n_trainings):
            category = rng.choice(list(TRAINING_CATALOG.keys()))
            training_name = rng.choice(TRAINING_CATALOG[category])
            completion_rate = float(rng.beta(8, 2))  # right-skewed: kebanyakan >80%

            rows.append({
                "employee_id":      emp["employee_id"],
                "training_name":    training_name,
                "category":         category,
                "training_date":    pd.Timestamp("2024-01-01") + pd.Timedelta(
                                        days=int(rng.integers(0, 365))
                                    ),
                "duration_hours":   int(rng.choice([4, 8, 16, 24, 40])),
                "completion_rate":  round(completion_rate, 2),
                "assessment_score": round(float(rng.beta(6, 2)) * 100, 1),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")
    print(f"Loaded {len(df_emp)} employees")

    df_training = generate_training_data(df_emp)
    print(f"Total training records: {len(df_training)}")
    print(f"\nCategory distribution:\n{df_training['category'].value_counts()}")
    print(f"\nAvg completion rate: {df_training['completion_rate'].mean():.2%}")
    print(f"Avg assessment score: {df_training['assessment_score'].mean():.1f}")

    df_training.to_csv("data/simulated/training_data.csv", index=False)
    print("\n✅ Saved: data/simulated/training_data.csv")