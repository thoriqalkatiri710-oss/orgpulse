import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ── Stopwords ─────────────────────────────────────────────────────────────────

STOPWORDS_ID = set(stopwords.words("indonesian") if "indonesian" in stopwords.fileids() else [])

CUSTOM_STOPWORDS = {
    "karyawan", "kinerja", "performance", "periode", "review",
    "tahun", "bulan", "kuartal", "triwulan", "saat", "dalam"
}

ALL_STOPWORDS = STOPWORDS_ID | CUSTOM_STOPWORDS


# ── 5.1.1 Text Preprocessing ──────────────────────────────────────────────────

def clean_review_text(text: str) -> str:
    """
    Preprocessing teks performance review:
    1. Lowercase
    2. Hapus karakter non-alfabet
    3. Tokenisasi
    4. Hapus stopwords + kata pendek
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_reviews(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """Terapkan preprocessing ke seluruh dataset review."""
    df = df_reviews.copy()
    df["cleaned_text"]       = df["review_text"].apply(clean_review_text)
    df["word_count_cleaned"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    return df


if __name__ == "__main__":
    df_reviews = pd.read_csv("data/simulated/performance_reviews.csv")
    print(f"Loaded {len(df_reviews)} reviews")

    df_processed = preprocess_reviews(df_reviews)

    print(f"\n── Sample ──")
    print(f"Original : {df_reviews['review_text'].iloc[0]}")
    print(f"Cleaned  : {df_processed['cleaned_text'].iloc[0]}")
    print(f"\nAvg word count (original) : {df_reviews['word_count'].mean():.1f}")
    print(f"Avg word count (cleaned)  : {df_processed['word_count_cleaned'].mean():.1f}")

    df_processed.to_csv("data/processed/reviews_cleaned.csv", index=False)
    print("\n✅ Saved: data/processed/reviews_cleaned.csv")