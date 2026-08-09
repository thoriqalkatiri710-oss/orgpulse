import pandas as pd
import numpy as np


def generate_attendance_data(df_employees: pd.DataFrame,
                              year: int = 2024,
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate data absensi harian per karyawan.
    1000 karyawan × 262 hari kerja = ~262,000 baris.
    Flight risk karyawan → lebih banyak absensi (leading indicator attrition).
    Simpan batch per departemen untuk efisiensi memori.
    """
    rng = np.random.default_rng(seed)
    work_days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    rows = []

    for _, emp in df_employees.iterrows():
        is_flight_risk = emp["is_flight_risk_sim"]

        # Flight risk: 5-20 hari absen, normal: 0-10 hari
        avg_absent_days = rng.integers(5, 20) if is_flight_risk else rng.integers(0, 10)
        absent_days_set = set(
            rng.choice(len(work_days), size=min(avg_absent_days, len(work_days)), replace=False)
        )

        for day_idx, work_day in enumerate(work_days):
            if day_idx in absent_days_set:
                status = rng.choice(
                    ["sick_leave", "personal_leave", "no_show"],
                    p=[0.6, 0.35, 0.05]
                )
                hours_worked = 0.0
            else:
                status = "present"
                # Late arrival / early departure untuk flight risk
                if is_flight_risk and rng.random() < 0.15:
                    hours_worked = round(float(rng.uniform(4, 7)), 1)
                else:
                    hours_worked = round(float(rng.normal(8, 0.5)), 1)

            rows.append({
                "employee_id":  emp["employee_id"],
                "date":         work_day,
                "status":       status,
                "hours_worked": hours_worked,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")
    print(f"Loaded {len(df_emp)} employees")
    print(f"Estimasi total rows: {len(df_emp)} × 262 hari = ~{len(df_emp)*262:,} baris")
    print("Generating... (mungkin 1-2 menit)")

    df_att = generate_attendance_data(df_emp)
    print(f"\nTotal rows: {len(df_att):,}")
    print(f"\nStatus distribution:\n{df_att['status'].value_counts()}")
    print(f"\nAbsensi rate: {(df_att['status'] != 'present').mean():.2%}")

    # Cek perbedaan flight risk vs normal
    emp_risk = df_emp[df_emp["is_flight_risk_sim"] == True]["employee_id"].tolist()
    df_att["is_flight_risk"] = df_att["employee_id"].isin(emp_risk)
    absent_risk = df_att[df_att["is_flight_risk"]]["status"].ne("present").mean()
    absent_normal = df_att[~df_att["is_flight_risk"]]["status"].ne("present").mean()
    print(f"\nAbsensi flight risk: {absent_risk:.2%}")
    print(f"Absensi normal:      {absent_normal:.2%}")

    df_att.drop(columns=["is_flight_risk"], inplace=True)
    df_att.to_csv("data/simulated/attendance.csv", index=False)
    print("\n✅ Saved: data/simulated/attendance.csv")