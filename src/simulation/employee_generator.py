import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta

fake = Faker("id_ID")

# Data 50% dari panduan asli: total ~135 karyawan (asli ~270)
DEPARTMENTS = {
    "Engineering":      {"size": 200, "avg_salary": 15_000_000, "turnover_risk": 0.22},
    "Product":          {"size": 60,  "avg_salary": 18_000_000, "turnover_risk": 0.18},
    "Marketing":        {"size": 80,  "avg_salary": 11_000_000, "turnover_risk": 0.20},
    "Sales":            {"size": 150, "avg_salary": 12_000_000, "turnover_risk": 0.30},
    "Finance":          {"size": 60,  "avg_salary": 13_000_000, "turnover_risk": 0.10},
    "HR":               {"size": 40,  "avg_salary": 10_000_000, "turnover_risk": 0.12},
    "Operations":       {"size": 100, "avg_salary":  9_000_000, "turnover_risk": 0.15},
    "Data & Analytics": {"size": 60,  "avg_salary": 17_000_000, "turnover_risk": 0.25},
    "Legal":            {"size": 30,  "avg_salary": 16_000_000, "turnover_risk": 0.08},
    "Customer Success": {"size": 80,  "avg_salary": 10_000_000, "turnover_risk": 0.28},
    "IT Infrastructure":{"size": 70,  "avg_salary": 13_000_000, "turnover_risk": 0.18},
    "Research":         {"size": 70,  "avg_salary": 16_000_000, "turnover_risk": 0.20},
}
# Total: 1000 karyawan

JOB_LEVELS = {
    1: "Staff", 2: "Senior Staff", 3: "Specialist",
    4: "Lead", 5: "Manager", 6: "Senior Manager", 7: "Director"
}


def generate_employees(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    rows = []
    emp_id = 1

    for dept, config in DEPARTMENTS.items():
        for _ in range(config["size"]):
            hire_date = date(2024, 1, 1) - timedelta(
                days=int(rng.integers(30, 2000))
            )
            tenure_years = (date(2024, 12, 31) - hire_date).days / 365.25
            job_level = int(np.clip(rng.normal(2.5, 1.2), 1, 7))

            salary_multiplier = 1 + (job_level - 1) * 0.25
            base = config["avg_salary"] * salary_multiplier
            salary = int(rng.lognormal(np.log(base), 0.15))

            engagement = float(np.clip(
                rng.normal(3.5 - 0.1 * max(tenure_years - 3, 0), 0.7), 1, 5
            ))

            base_risk = config["turnover_risk"]
            risk = base_risk * (1.5 if engagement < 2.5 else 0.7 if engagement > 4.0 else 1.0)
            risk *= (1.3 if salary < config["avg_salary"] * 0.85 else 1.0)
            is_high_risk = bool(rng.random() < risk)

            rows.append({
                "employee_id":       f"EMP{emp_id:05d}",
                "full_name":         fake.name(),
                "department":        dept,
                "job_level":         job_level,
                "job_level_name":    JOB_LEVELS[job_level],
                "hire_date":         hire_date,
                "tenure_years":      round(tenure_years, 2),
                "monthly_salary":    salary,
                "gender":            rng.choice(["L", "P"], p=[0.58, 0.42]),
                "age":               int(rng.integers(22, 52)),
                "education":         rng.choice(["D3", "S1", "S2", "S3"], p=[0.10, 0.65, 0.23, 0.02]),
                "manager_id":        None,
                "engagement_score":  round(engagement, 1),
                "is_flight_risk_sim": is_high_risk,
            })
            emp_id += 1

    df = pd.DataFrame(rows)

    # Assign manager_id per departemen
    for dept in DEPARTMENTS:
        dept_mask = df["department"] == dept
        dept_emp = df[dept_mask & (df["job_level"] >= 4)]["employee_id"].tolist()
        if dept_emp:
            df.loc[dept_mask & (df["job_level"] < 4), "manager_id"] = rng.choice(dept_emp)

    return df


if __name__ == "__main__":
    df = generate_employees()
    print(f"Total karyawan: {len(df)}")
    print(f"Per departemen:\n{df.groupby('department').size()}")
    print(f"\nSample:\n{df.head()}")
    df.to_csv("data/simulated/employees.csv", index=False)
    print("\n✅ Saved: data/simulated/employees.csv")