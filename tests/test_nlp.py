import pytest
import pandas as pd
from src.nlp.sentiment import analyze_sentiment_hr
from src.nlp.text_processor import clean_review_text


def test_clean_text_removes_special_chars():
    raw     = "Kinerja sangat baik! 100% target tercapai. (2024)"
    cleaned = clean_review_text(raw)
    assert "!" not in cleaned
    assert "%" not in cleaned
    assert "100" not in cleaned


def test_clean_text_lowercases():
    raw     = "KINERJA KARYAWAN BAIK"
    cleaned = clean_review_text(raw)
    assert cleaned == cleaned.lower()


def test_sentiment_label_correct():
    df = pd.DataFrame({"review_text": [
        "Kinerja sangat luar biasa dan melampaui ekspektasi",
        "Ada beberapa concern terkait performa dan keterlambatan",
        "Performa cukup memadai tanpa kekurangan berarti"
    ]})
    result = analyze_sentiment_hr(df)
    assert "sentiment_label" in result.columns
    assert "compound" in result.columns
    assert result["compound"].between(-1, 1).all()


def test_positive_text_gets_positive_sentiment():
    df = pd.DataFrame({"review_text": [
        "Melampaui target dengan hasil yang sangat konsisten dan membanggakan"
    ]})
    result = analyze_sentiment_hr(df)
    assert result["compound"].iloc[0] > 0


def test_negative_text_gets_negative_sentiment():
    df = pd.DataFrame({"review_text": [
        "Ada banyak concern serius terkait belum tercapainya target dan kedisiplinan"
    ]})
    result = analyze_sentiment_hr(df)
    assert result["compound"].iloc[0] < 0