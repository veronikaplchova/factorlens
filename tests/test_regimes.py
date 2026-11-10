
"""Tests for market regime detection."""

import numpy as np
import pandas as pd
import pytest

from src.regimes import (
    calculate_rolling_volatility,
    classify_market_regimes,
)

def test_rolling_volatility():
    """Verify annualized rolling market volatility."""

    returns = pd.Series([0.01, -0.02, 0.03, -0.01])

    result = calculate_rolling_volatility(
        returns,
        window=3,
    )

    expected = (
        returns.iloc[:3].std() * np.sqrt(252)
    )

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(expected)


@pytest.mark.parametrize("invalid_window", [0, 1, -5])
def test_rolling_volatility_invalid_window(invalid_window):
    """Reject invalid rolling volatility window sizes."""

    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError, match="at least 2"):
        calculate_rolling_volatility(
            returns,
            window=invalid_window,
        )


def test_rolling_volatility_empty_data():
    """Reject an empty market-return dataset."""

    returns = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        calculate_rolling_volatility(
            returns,
            window=21,
        )


def test_classify_market_regimes():
    """Classify volatility using only previous observations."""

    volatility = pd.Series(
        [0.10, 0.20, 0.30, 0.15, 0.40, 0.10],
        index=pd.date_range("2020-01-01", periods=6),
    )

    result = classify_market_regimes(
        volatility,
        threshold_window=3,
    )

    # First three observations lack sufficient history.
    assert result.iloc[:3].isna().all()

    # Historical median of 0.10, 0.20, 0.30 is 0.20.
    assert result.iloc[3] == "Low Volatility"

    # Historical median of 0.20, 0.30, 0.15 is 0.20.
    assert result.iloc[4] == "High Volatility"

    # Historical median of 0.30, 0.15, 0.40 is 0.30.
    assert result.iloc[5] == "Low Volatility"


@pytest.mark.parametrize("invalid_window", [0, 1, -5])
def test_classify_market_regimes_invalid_window(invalid_window):
    """Reject invalid historical threshold windows."""

    volatility = pd.Series([0.10, 0.20, 0.30, 0.15])

    with pytest.raises(ValueError, match="at least 2"):
        classify_market_regimes(
            volatility,
            threshold_window=invalid_window,
        )


def test_classify_market_regimes_empty():
    """Reject empty rolling-volatility data."""

    volatility = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        classify_market_regimes(
            volatility,
            threshold_window=126,
        )



def test_classify_market_regimes_missing_values():
    """Leave regimes unclassified when volatility data is missing."""

    volatility = pd.Series(
        [0.10, 0.20, np.nan, 0.30, 0.15, 0.40, 0.10],
        index=pd.date_range("2020-01-01", periods=7),
    )

    result = classify_market_regimes(
        volatility,
        threshold_window=2,
    )

    
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[4])
    assert result.iloc[5] == "High Volatility"
    assert result.iloc[6] == "Low Volatility"

