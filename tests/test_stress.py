
"""Tests for historical portfolio stress calculations."""

import numpy as np
import pandas as pd
import pytest

from src.stress import calculate_stress_period


def test_stress_total_return():
    dates = pd.date_range("2020-03-01", periods=3)
    returns = pd.Series([0.10, -0.10, 0.05], index=dates)

    result = calculate_stress_period(
        returns, "2020-03-01", "2020-03-03"
    )

    assert result["total_return"] == pytest.approx(0.0395)
    assert result["n_observations"] == 3


def test_stress_max_drawdown():
    dates = pd.date_range("2020-03-01", periods=3)
    returns = pd.Series([0.10, -0.20, 0.05], index=dates)

    result = calculate_stress_period(
        returns, "2020-03-01", "2020-03-03"
    )

    assert result["max_drawdown"] == pytest.approx(-0.20)


def test_stress_volatility():
    dates = pd.date_range("2020-03-01", periods=4)
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01],
        index=dates,
    )

    result = calculate_stress_period(
        returns, "2020-03-01", "2020-03-04"
    )

    expected = returns.std() * np.sqrt(252)

    assert result["annualized_volatility"] == pytest.approx(
        expected
    )


def test_stress_invalid_dates():
    dates = pd.date_range("2020-03-01", periods=3)
    returns = pd.Series([0.01, 0.02, -0.01], index=dates)

    with pytest.raises(ValueError, match="before end date"):
        calculate_stress_period(
            returns, "2020-04-01", "2020-03-01"
        )


def test_stress_no_data():
    dates = pd.date_range("2025-01-01", periods=3)
    returns = pd.Series([0.01, 0.02, -0.01], index=dates)

    with pytest.raises(ValueError, match="No returns"):
        calculate_stress_period(
            returns, "2020-03-01", "2020-03-31"
        )



def test_stress_first_day_loss():
    """Capture a drawdown that starts on the first day."""

    dates = pd.date_range("2020-03-01", periods=3)
    returns = pd.Series(
        [-0.10, 0.05, 0.02],
        index=dates,
    )

    result = calculate_stress_period(
        returns, "2020-03-01", "2020-03-03"
    )

    assert result["max_drawdown"] == pytest.approx(-0.10)
