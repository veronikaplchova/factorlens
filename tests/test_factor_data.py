
"""Tests for the Fama-French factor data loader."""

import numpy as np
import pandas as pd
import pytest

from src.factor_data import (
    load_fama_french_factors,
    load_fama_french_five_factors,
)


def make_sample_data():
    """Create sample data resembling the French data library."""
    dates = pd.period_range("2025-01-02", periods=3, freq="D")

    factors = pd.DataFrame(
        {
            "Mkt-RF": [-0.16, 1.34, 0.55],
            "SMB": [0.10, 0.60, -0.38],
            "HML": [-0.29, -0.92, -0.32],
            "RF": [0.02, 0.02, 0.02],
        },
        index=dates,
    )

    return {0: factors}


def test_factor_data_conversion(monkeypatch):
    """Percent values must be converted to decimal returns."""
    def fake_reader(*args, **kwargs):
        return make_sample_data()

    monkeypatch.setattr(
        "src.factor_data.web.DataReader",
        fake_reader,
    )

    result = load_fama_french_factors(
        "2025-01-01", "2025-01-31"
    )

    assert result.iloc[0]["Mkt-RF"] == pytest.approx(-0.0016)
    assert result.iloc[1]["SMB"] == pytest.approx(0.006)
    assert result.iloc[0]["RF"] == pytest.approx(0.0002)


def test_factor_dates_are_timestamps(monkeypatch):
    """PeriodIndex dates should become DatetimeIndex."""
    monkeypatch.setattr(
        "src.factor_data.web.DataReader",
        lambda *args, **kwargs: make_sample_data(),
    )

    result = load_fama_french_factors(
        "2025-01-01", "2025-01-31"
    )

    assert isinstance(result.index, pd.DatetimeIndex)


def test_invalid_date_range():
    """Start date must precede end date."""
    with pytest.raises(ValueError, match="before end date"):
        load_fama_french_factors(
            "2025-02-01", "2025-01-01"
        )


def test_missing_factor_column(monkeypatch):
    """Missing required factors should raise an error."""
    sample = make_sample_data()
    sample[0] = sample[0].drop(columns=["HML"])

    monkeypatch.setattr(
        "src.factor_data.web.DataReader",
        lambda *args, **kwargs: sample,
    )

    with pytest.raises(ValueError, match="Required factor columns"):
        load_fama_french_factors(
            "2025-01-01", "2025-01-31"
        )


def test_empty_factor_data(monkeypatch):
    """Empty datasets should raise an error."""
    sample = make_sample_data()
    sample[0] = sample[0].iloc[0:0]

    monkeypatch.setattr(
        "src.factor_data.web.DataReader",
        lambda *args, **kwargs: sample,
    )

    with pytest.raises(ValueError, match="No factor data"):
        load_fama_french_factors(
            "2025-01-01", "2025-01-31"
        )



def test_five_factor_data_conversion(monkeypatch):
    """Check five-factor values and date conversion."""

    dates = pd.period_range("2025-01-02", periods=3, freq="D")

    sample = pd.DataFrame(
        {
            "Mkt-RF": [1.0, -2.0, 0.5],
            "SMB": [0.2, -0.1, 0.3],
            "HML": [0.4, 0.2, -0.2],
            "RMW": [0.6, -0.3, 0.1],
            "CMA": [-0.5, 0.4, 0.2],
            "RF": [0.02, 0.02, 0.02],
        },
        index=dates,
    )

    monkeypatch.setattr(
        "src.factor_data.web.DataReader",
        lambda *args, **kwargs: {0: sample},
    )

    result = load_fama_french_five_factors(
        "2025-01-01", "2025-01-31"
    )

    assert result.iloc[0]["Mkt-RF"] == pytest.approx(0.01)
    assert result.iloc[0]["RMW"] == pytest.approx(0.006)
    assert result.iloc[0]["CMA"] == pytest.approx(-0.005)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_five_factor_missing_column(monkeypatch):
    """Reject five-factor datasets missing a column."""

    sample = pd.DataFrame(
        {
            "Mkt-RF": [1.0],
            "SMB": [0.2],
            "HML": [0.4],
            "RMW": [0.6],
            "RF": [0.02],
        }
    )

    monkeypatch.setattr(
        "src.factor_data.web.DataReader",
        lambda *args, **kwargs: {0: sample},
    )

    with pytest.raises(ValueError, match="five-factor columns"):
        load_fama_french_five_factors(
            "2025-01-01", "2025-01-31"
        )


def test_five_factor_invalid_dates():
    """Reject an invalid date range."""

    with pytest.raises(ValueError, match="before end date"):
        load_fama_french_five_factors(
            "2025-02-01", "2025-01-01"
        )
