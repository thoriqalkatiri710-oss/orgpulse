import pandas as pd
import numpy as np


def generate_communication_data(df_employees: pd.DataFrame,
                                 n_interactions: int = 50_000,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generator data komunikasi/kolaborasi untuk ONA.
    70% intra-departemen, 30% cross-departemen — angka realistis
    berdasarkan studi ONA di perusahaan Fortune 500.
    """
    rng = np.random.default_rng(seed)
    emp_ids = df_employees["employee_id"].tolist()
    dept_map = df_employees.set_index("employee_id")["department"].to_dict()
    level_map = df_employees.set_index("employee_id")["job_level"].to_dict()

    rows = []
    for _ in range(n_interactions):
        sender = rng.choice(emp_ids)
        sender_dept = dept_map[sender]
        sender_level = level_map[sender]

        # 70% komunikasi dalam departemen sama
        same_dept = [e for e in emp_ids if dept_map[e] == sender_dept and e != sender]
        other_dept = [e for e in emp_ids if dept_map[e] != sender_dept and e != sender]

        if rng.random() < 0.70 and same_dept:
            receiver = rng.choice(same_dept)
        else:
            receiver = rng.choice(other_dept) if other_dept else rng.choice(emp_ids)

        # Level tinggi: lebih banyak email & meeting
        # Level rendah: lebih banyak slack & jira
        channel = rng.choice(
            ["email", "slack", "meeting", "jira_comment"],
            p=[0.35, 0.40, 0.15, 0.10] if sender_level <= 3 else [0.45, 0.25, 0.25, 0.05]
        )

        rows.append({
            "interaction_id":       f"INT{len(rows)+1:08d}",
            "sender_id":            sender,
            "receiver_id":          receiver,
            "channel":              channel,
            "interaction_date":     pd.Timestamp("2024-01-01") + pd.Timedelta(
                                        days=int(rng.integers(0, 365))
                                    ),
            "response_time_hours":  float(rng.exponential(4)) if channel == "email" else None,
            "sentiment_raw":        float(rng.normal(0.1, 0.3)),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")
    print(f"Loaded {len(df_emp)} employees")

    df_comm = generate_communication_data(df_emp, n_interactions=50_000)
    print(f"Total interactions: {len(df_comm)}")
    print(f"\nChannel distribution:\n{df_comm['channel'].value_counts()}")

    # Cek rasio intra vs cross departemen
    dept_map = df_emp.set_index("employee_id")["department"].to_dict()
    df_comm["sender_dept"] = df_comm["sender_id"].map(dept_map)
    df_comm["receiver_dept"] = df_comm["receiver_id"].map(dept_map)
    intra = (df_comm["sender_dept"] == df_comm["receiver_dept"]).mean()
    print(f"\nIntra-dept ratio: {intra:.1%} (target ~70%)")
    print(f"Cross-dept ratio: {1-intra:.1%} (target ~30%)")

    df_comm.drop(columns=["sender_dept", "receiver_dept"], inplace=True)
    df_comm.to_csv("data/simulated/communications.csv", index=False)
    print("\n✅ Saved: data/simulated/communications.csv")


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")
    df_comm = generate_communication_data(df_emp, n_interactions=50_000)
    print(f"Total interactions: {len(df_comm)}")
    print(f"\nChannel distribution:\n{df_comm['channel'].value_counts()}")
    dept_map = df_emp.set_index("employee_id")["department"].to_dict()
    df_comm["sender_dept"] = df_comm["sender_id"].map(dept_map)
    df_comm["receiver_dept"] = df_comm["receiver_id"].map(dept_map)
    intra = (df_comm["sender_dept"] == df_comm["receiver_dept"]).mean()
    print(f"\nIntra-dept ratio: {intra:.1%} (target ~70%)")
    print(f"Cross-dept ratio: {1-intra:.1%} (target ~30%)")
    df_comm.drop(columns=["sender_dept", "receiver_dept"], inplace=True)
    df_comm.to_csv("data/simulated/communications.csv", index=False)
    print("\n✅ Saved: data/simulated/communications.csv")