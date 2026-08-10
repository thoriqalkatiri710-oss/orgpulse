import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


# ── 5.2.1 Sentence Embeddings ─────────────────────────────────────────────────

def generate_review_embeddings(df_reviews: pd.DataFrame,
                                text_col: str = "review_text",
                                model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                                batch_size: int = 32,
                                save_path: str = None) -> np.ndarray:
    """
    Generate semantic embeddings untuk teks performance review.
    Model: paraphrase-multilingual-MiniLM-L12-v2
    - Support 50+ bahasa termasuk Bahasa Indonesia
    - Dimensi embedding: 384
    - Ringan dan cepat untuk dataset ~2000 dokumen
    """
    model = SentenceTransformer(model_name)
    texts = df_reviews[text_col].tolist()

    print(f"Encoding {len(texts)} texts dengan model {model_name}...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    if save_path:
        np.save(save_path, embeddings)
        print(f"✅ Embeddings saved: {save_path}")

    return embeddings


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


if __name__ == "__main__":
    df_reviews = pd.read_csv("data/processed/reviews_cleaned.csv")
    print(f"Loaded {len(df_reviews)} reviews")

    embeddings = generate_review_embeddings(
        df_reviews,
        text_col="review_text",
        batch_size=32,
        save_path="data/processed/review_embeddings.npy"
    )

    print(f"\n── Embedding Stats ──")
    print(f"Shape     : {embeddings.shape}")
    print(f"Dimensi   : {embeddings.shape[1]}")
    print(f"Dtype     : {embeddings.dtype}")
    print(f"Sample[0] : {embeddings[0][:5]}...")