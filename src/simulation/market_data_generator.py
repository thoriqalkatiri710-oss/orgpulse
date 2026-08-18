import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 15.1.1 Market Salary Data ─────────────────────────────────────────────────

def generate_market_salary_data(seed: int = 42) -> pd.DataFrame:
    """
    Simulasi data survei gaji pasar (analogous to Mercer, WTW, Korn Ferry).
    Digunakan untuk benchmarking kompensasi internal vs pasar.
    """
    rng = np.random.default_rng(seed)

    MARKET_BENCHMARKS = {
        ("Engineering",      1): {"p25":  8_500_000, "p50": 11_000_000, "p75": 14_000_000},
        ("Engineering",      2): {"p25": 11_000_000, "p50": 14_500_000, "p75": 18_000_000},
        ("Engineering",      3): {"p25": 14_000_000, "p50": 18_000_000, "p75": 23_000_000},
        ("Data & Analytics", 1): {"p25":  9_000_000, "p50": 12_000_000, "p75": 16_000_000},
        ("Data & Analytics", 2): {"p25": 12_000_000, "p50": 16_000_000, "p75": 21_000_000},
        ("Product",          2): {"p25": 13_000_000, "p50": 17_500_000, "p75": 22_000_000},
        ("Sales",            2): {"p25": 10_000_000, "p50": 13_500_000, "p75": 17_000_000},
        ("HR",               2): {"p25":  8_000_000, "p50": 11_000_000, "p75": 14_000_000},
        ("Finance",          2): {"p25":  9_500_000, "p50": 13_000_000, "p75": 17_000_000},
        ("Operations",       2): {"p25":  8_000_000, "p50": 10_500_000, "p75": 13_500_000},
        ("Marketing",        2): {"p25":  9_000_000, "p50": 11_500_000, "p75": 15_000_000},
        ("Research",         2): {"p25": 11_000_000, "p50": 15_000_000, "p75": 20_000_000},
    }

    rows = []
    for (dept, level), benchmarks in MARKET_BENCHMARKS.items():
        rows.append({
            "department":    dept,
            "job_level":     level,
            "market_p25":    benchmarks["p25"],
            "market_p50":    benchmarks["p50"],
            "market_p75":    benchmarks["p75"],
            "survey_source": "Simulated Market Survey 2024",
            "n_data_points": int(rng.integers(50, 300)),
        })

    return pd.DataFrame(rows)


# ── 15.1.2 Compa-Ratio ────────────────────────────────────────────────────────

def compute_compa_ratio(df_employees: pd.DataFrame,
                         df_market: pd.DataFrame) -> pd.DataFrame:
    """
    Compa-ratio = Gaji Aktual / Market P50
    < 0.85  : significantly underpaid (flight risk signal)
    0.85-0.95: somewhat below market
    0.95-1.05: at market
    > 1.05  : above market
    """
    merged = df_employees.merge(df_market, on=["department", "job_level"], how="left")

    merged["compa_ratio"] = merged["monthly_salary"] / merged["market_p50"].where(
        merged["market_p50"].notna(), merged["monthly_salary"]
    )

    merged["pay_positioning"] = pd.cut(
        merged["compa_ratio"],
        bins=[0, 0.85, 0.95, 1.05, 999],
        labels=["Below Market", "Slightly Below", "At Market", "Above Market"]
    )

    return merged


# ── 15.2.1 Visualization ──────────────────────────────────────────────────────

def plot_compa_ratio_distribution(df_compa: pd.DataFrame,
                                   save_path: str = None):
    """
    Panel kiri  : distribusi compa-ratio seluruh karyawan
    Panel kanan : % below market per departemen
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel kiri: distribusi compa-ratio
    df_compa["compa_ratio"].hist(bins=30, ax=axes[0], color="#3B82F6", alpha=0.7)
    axes[0].axvline(1.0,  color="red",    linestyle="--", label="Market P50")
    axes[0].axvline(0.85, color="orange", linestyle="--", label="Below Market threshold")
    axes[0].set_title("Distribusi Compa-Ratio Seluruh Karyawan")
    axes[0].set_xlabel("Compa-Ratio (1.0 = tepat di median pasar)")
    axes[0].legend()

    # Panel kanan: % below market per departemen
    dept_below = df_compa.groupby("department").apply(
        lambda g: (g["compa_ratio"] < 0.85).mean() * 100
    ).reset_index(name="pct_below_market")
    dept_below_sorted = dept_below.sort_values("pct_below_market", ascending=True)

    colors = [
        "#EF4444" if v > 30 else "#F97316" if v > 15 else "#22C55E"
        for v in dept_below_sorted["pct_below_market"]
    ]
    axes[1].barh(dept_below_sorted["department"],
                 dept_below_sorted["pct_below_market"], color=colors)
    axes[1].set_title("% Karyawan Below Market per Departemen")
    axes[1].set_xlabel("% Karyawan dengan Compa-Ratio < 0.85")
    axes[1].axvline(15, color="gray", linestyle="--", alpha=0.5, label="Benchmark toleransi")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    df_emp = pd.read_csv("data/simulated/employees.csv")

    print("Generating market salary data...")
    df_market = generate_market_salary_data()
    print(f"Market benchmarks: {len(df_market)} dept-level combinations")

    print("\nComputing compa-ratios...")
    df_compa = compute_compa_ratio(df_emp, df_market)

    print(f"\n── Compa-Ratio Summary ──")
    print(f"Avg compa-ratio  : {df_compa['compa_ratio'].mean():.3f}")
    print(f"Below market (<0.85): {(df_compa['compa_ratio'] < 0.85).mean():.1%}")
    print(f"At market (0.95-1.05): {((df_compa['compa_ratio'] >= 0.95) & (df_compa['compa_ratio'] <= 1.05)).mean():.1%}")

    print(f"\n── Pay Positioning Distribution ──")
    print(df_compa["pay_positioning"].value_counts().to_string())

    print(f"\n── % Below Market per Department ──")
    dept_below = df_compa.groupby("department").apply(
        lambda g: (g["compa_ratio"] < 0.85).mean() * 100
    ).sort_values(ascending=False)
    print(dept_below.round(1).to_string())

    plot_compa_ratio_distribution(
        df_compa,
        save_path="results/figures/compa_ratio_distribution.png"
    )

    df_market.to_csv("data/simulated/market_salary_data.csv", index=False)
    df_compa.to_csv("data/processed/compa_ratio_analysis.csv", index=False)
    print("\n✅ Saved: market_salary_data.csv & compa_ratio_analysis.csv")