import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel


# ── 5.4.1 LDA Topic Modeling ──────────────────────────────────────────────────

def fit_lda_model(df_reviews: pd.DataFrame,
                  text_col: str = "cleaned_text",
                  n_topics: int = 6,
                  seed: int = 42) -> tuple:
    """
    LDA topic modeling pada teks performance review.
    6 topik yang diharapkan:
    1. Kinerja & Target
    2. Kepemimpinan & Tim
    3. Pengembangan Karier
    4. Komunikasi
    5. Flight Risk Signals
    6. Compliance & Proses
    """
    tokenized  = [text.split() for text in df_reviews[text_col]]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus     = [dictionary.doc2bow(doc) for doc in tokenized]

    print(f"Vocabulary size: {len(dictionary)}")
    print(f"Fitting LDA ({n_topics} topics, 15 passes)...")

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=seed,
        passes=15,
        alpha="auto",
        eta="auto",
    )

    coherence = CoherenceModel(
        model=lda_model, texts=tokenized,
        dictionary=dictionary, coherence="c_v"
    ).get_coherence()
    print(f"Coherence score: {coherence:.4f}")

    return lda_model, dictionary, corpus, tokenized


def get_document_topics(lda_model, corpus, n_topics: int) -> pd.DataFrame:
    """Return topic distribution per dokumen."""
    topic_assignments = []
    for doc_bow in corpus:
        topic_dist = dict(lda_model.get_document_topics(doc_bow, minimum_probability=0))
        topic_assignments.append({
            f"topic_{i}": topic_dist.get(i, 0) for i in range(n_topics)
        })
    return pd.DataFrame(topic_assignments)


def print_topics(lda_model, n_words: int = 10):
    """Print top words per topik."""
    print("\n── Top Words per Topic ──")
    for i in range(lda_model.num_topics):
        words = lda_model.show_topic(i, topn=n_words)
        word_str = ", ".join([w for w, _ in words])
        print(f"Topic {i}: {word_str}")


if __name__ == "__main__":
    df_reviews = pd.read_csv("data/processed/reviews_cleaned.csv")
    print(f"Loaded {len(df_reviews)} reviews")

    lda_model, dictionary, corpus, tokenized = fit_lda_model(
        df_reviews, text_col="cleaned_text", n_topics=6
    )

    print_topics(lda_model)

    # Topic distribution per dokumen
    df_topics = get_document_topics(lda_model, corpus, n_topics=6)
    df_topics["dominant_topic"] = df_topics.idxmax(axis=1)
    df_topics["employee_id"]    = df_reviews["employee_id"].values
    df_topics["review_cycle"]   = df_reviews["review_cycle"].values

    print(f"\n── Dominant Topic Distribution ──")
    print(df_topics["dominant_topic"].value_counts())

    # Simpan model dan hasil
    lda_model.save("results/models/lda_model")
    dictionary.save("results/models/lda_dictionary")
    df_topics.to_csv("data/processed/topic_assignments.csv", index=False)
    print("\n✅ Saved: lda_model, lda_dictionary, topic_assignments.csv")