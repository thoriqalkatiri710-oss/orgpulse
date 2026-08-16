import pytest
import pandas as pd
import numpy as np
from datetime import date
from src.simulation.employee_generator import generate_employees, DEPARTMENTS


def test_employee_count_matches_config():
    df = generate_employees(seed=42)
    expected = sum(cfg["size"] for cfg in DEPARTMENTS.values())
    assert len(df) == expected


def test_no_null_employee_ids():
    df = generate_employees(seed=42)
    assert df["employee_id"].isna().sum() == 0
    assert df["employee_id"].nunique() == len(df)


def test_salary_distribution_realistic():
    df = generate_employees(seed=42)
    assert (df["monthly_salary"] >= 5_000_000).all()
    median_salary = df["monthly_salary"].median()
    assert 8_000_000 <= median_salary <= 25_000_000


def test_engagement_score_bounded():
    df = generate_employees(seed=42)
    assert df["engagement_score"].between(1, 5).all()


def test_tenure_consistent_with_hire_date():
    df = generate_employees(seed=42)
    df["hire_date"] = pd.to_datetime(df["hire_date"]).dt.date
    computed_tenure = df["hire_date"].apply(
        lambda d: (date(2024, 12, 31) - d).days / 365.25
    )
    assert (abs(computed_tenure - df["tenure_years"]) < 0.1).all()


def test_flight_risk_rate_realistic():
    df = generate_employees(seed=42)
    flight_risk_rate = df["is_flight_risk_sim"].mean()
    assert 0.10 <= flight_risk_rate <= 0.35