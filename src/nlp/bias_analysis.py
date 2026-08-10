import pandas as pd
import numpy as np


# ── 5.5.1 Gender Language Bias Detection ─────────────────────────────────────

GENDERED_WORDS = {
    "communal": [
        "kolaboratif", "suportif", "peduli", "empatik", "ramah",
        "kooperatif", "hangat", "pengertian"
    ],
    "agentic": [
        "tegas", "mandiri", "kompetitif", "ambisius", "dominan",
        "percaya diri", "pemimpin", "berani", "analitis"
    ],
}


def analyze_language_bias(df_reviews: pd.DataFrame,
                           df_employees: pd.DataFrame) -> pd.DataFrame:
    """
    Deteksi bias gender dalam pilihan kata reviewer.
    Penelitian menunjukkan reviewer cenderung pakai kata berbeda untuk
    karyawan L vs P meski performa setara — sinyal bias dalam proses review.

    Interpretasi:
    - P lebih banyak kata 'communal' (kolaboratif, suportif)
    - L lebih banyak kata 'agentic' (ambisius, tegas)
    → Bias sistematis yang bisa disampaikan sebagai temuan audit HR.
    """
    merged = df_reviews.merge(
        df_employees[["employee_id", "gender"]], on="employee_id"
    )

    def count_word_category(text, category):
        text_lower = text.lower()
        return sum(1 for word in GENDERED_WORDS[category] if word in text_lower)

    merged["communal_count"] = merged["review_text"].apply(
        lambda t: count_word_category(t, "communal")
    )
    merged["agentic_count"] = merged["review_text"].apply(
        lambda t: count_word_category(t, "agentic")
    )
    merged["agentic_ratio"] = merged["agentic_count"] / (
        merged["communal_count"] + merged["agentic_count"] + 1e-6
    )

    bias_summary = merged.groupby("gender").agg(
        avg_communal      =("communal_count", "mean"),
        avg_agentic       =("agentic_count",  "mean"),
        avg_agentic_ratio =("agentic_ratio",  "mean"),
        avg_rating        =("rating",         "mean"),
        n_reviews         =("review_id",      "count"),
    ).round(4)

    return bias_summary, merged


if __name__ == "__main__":
    df_reviews = pd.read_csv("data/processed/reviews_sentiment.csv")
    df_emp     = pd.read_csv("data/simulated/employees.csv")
    print(f"Loaded {len(df_reviews)} reviews, {len(df_emp)} employees")

    bias_summary, merged = analyze_language_bias(df_reviews, df_emp)

    print("\n── Gender Language Bias Summary ──")
    print(bias_summary.to_string())

    print("\n── Interpretasi ──")
    communal_L = bias_summary.loc["L", "avg_communal"]
    communal_P = bias_summary.loc["P", "avg_communal"]
    agentic_L  = bias_summary.loc["L", "avg_agentic"]
    agentic_P  = bias_summary.loc["P", "avg_agentic"]

    if communal_P > communal_L:
        print(f"⚠️  Perempuan mendapat lebih banyak kata communal: P={communal_P:.3f} vs L={communal_L:.3f}")
    if agentic_L > agentic_P:
        print(f"⚠️  Laki-laki mendapat lebih banyak kata agentic: L={agentic_L:.3f} vs P={agentic_P:.3f}")

    rating_L = bias_summary.loc["L", "avg_rating"]
    rating_P = bias_summary.loc["P", "avg_rating"]
    print(f"\nAvg rating: L={rating_L:.3f}, P={rating_P:.3f}")
    if abs(rating_L - rating_P) < 0.2:
        print("✅ Rating relatif setara — perbedaan kata bukan karena perbedaan performa")

    merged.to_csv("data/processed/reviews_bias_analysis.csv", index=False)
    print("\n✅ Saved: data/processed/reviews_bias_analysis.csv")