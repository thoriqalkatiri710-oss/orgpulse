import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ── Custom HR Lexicon ─────────────────────────────────────────────────────────
# Kata-kata yang maknanya berbeda di konteks HR vs makna umum

HR_LEXICON_OVERRIDE = {
    "concern":      -2.0,
    "potensi":       1.5,
    "tantangan":    -0.5,
    "adequate":     -0.3,   # dalam HR: lebih negatif dari makna harfiahnya
    "improvement":  -0.8,
    "disarankan":   -0.5,
    "perlu":        -0.3,
    "belum":        -1.5,
    "konsisten":     1.5,
    "melampaui":     2.0,
    "retensi":      -1.0,   # muncul dalam konteks flight risk
    "turnover":     -1.5,
    "resign":       -2.0,
    "mengundurkan": -2.0,
    "puas":          1.5,
    "tidak puas":   -2.0,
    "engage":        1.5,
    "disengaged":   -2.0,
}


# ── 5.3.1 Sentiment Analysis ──────────────────────────────────────────────────

def analyze_sentiment_hr(df_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    VADER sentiment analysis dengan custom HR lexicon.
    VADER didesain untuk teks pendek — cocok untuk performance review.
    HR lexicon override memastikan kata seperti 'adequate' dan 'concern'
    mendapat skor yang sesuai konteks HR, bukan makna umum.
    """
    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(HR_LEXICON_OVERRIDE)

    sentiments = []
    for text in df_reviews["review_text"]:
        scores = sid.polarity_scores(text)
        sentiments.append({
            "compound":        scores["compound"],
            "positive":        scores["pos"],
            "negative":        scores["neg"],
            "neutral":         scores["neu"],
            "sentiment_label": (
                "positive" if scores["compound"] >= 0.05 else
                "negative" if scores["compound"] <= -0.05 else
                "neutral"
            )
        })

    return df_reviews.join(pd.DataFrame(sentiments))


if __name__ == "__main__":
    df_reviews = pd.read_csv("data/processed/reviews_cleaned.csv")
    print(f"Loaded {len(df_reviews)} reviews")

    df_sentiment = analyze_sentiment_hr(df_reviews)

    print(f"\n── Sentiment Distribution ──")
    print(df_sentiment["sentiment_label"].value_counts())

    print(f"\n── Avg Compound Score per Rating ──")
    print(df_sentiment.groupby("rating")["compound"].mean().round(3))

    print(f"\n── Sample Negative Review ──")
    neg = df_sentiment[df_sentiment["sentiment_label"] == "negative"].iloc[0]
    print(f"Text     : {neg['review_text']}")
    print(f"Compound : {neg['compound']:.3f}")

    df_sentiment.to_csv("data/processed/reviews_sentiment.csv", index=False)
    print("\n✅ Saved: data/processed/reviews_sentiment.csv")