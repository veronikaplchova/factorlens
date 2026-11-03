
"""Tests for historical portfolio stress calculations."""

import numpy as np
import pandas as pd
import pytest

from src.stress import calculate_stress_period, validate_stress_prices


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


def test_stress_benchmark_date_alignment():
    """Compare portfolio and SPY only on shared dates."""

    portfolio = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime([
            "2020-03-02", "2020-03-03", "2020-03-04"
        ]),
    )

    benchmark = pd.Series(
        [-0.01, 0.02, 0.01],
        index=pd.to_datetime([
            "2020-03-03", "2020-03-04", "2020-03-05"
        ]),
    )

    common_dates = portfolio.index.intersection(benchmark.index)

    aligned_portfolio = portfolio.loc[common_dates]
    aligned_benchmark = benchmark.loc[common_dates]

    assert len(common_dates) == 2
    assert aligned_portfolio.index.equals(aligned_benchmark.index)
    assert list(common_dates) == list(pd.to_datetime([
        "2020-03-03", "2020-03-04"
    ]))

def test_validate_stress_prices_missing_values():
    """Reject historical prices containing missing values."""

    prices = pd.DataFrame({
        "AAPL": [100.0, np.nan, 105.0],
        "MSFT": [200.0, 202.0, 204.0],
    })

    with pytest.raises(ValueError, match="missing values"):
        validate_stress_prices(prices)


@pytest.mark.parametrize("invalid_price", [0.0, -50.0])
def test_validate_stress_prices_invalid_price(invalid_price):
    """Reject zero and negative stock prices."""

    prices = pd.DataFrame({
        "AAPL": [100.0, invalid_price, 105.0],
        "MSFT": [200.0, 202.0, 204.0],
    })

    with pytest.raises(ValueError, match="must be positive"):
        validate_stress_prices(prices)


def test_validate_stress_prices_empty():
    """Reject an empty historical price dataset."""

    prices = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        validate_stress_prices(prices)


@pytest.mark.parametrize("invalid_price", [np.inf, -np.inf])
def test_validate_stress_prices_infinite(invalid_price):
    """Reject positive and negative infinite prices."""

    prices = pd.DataFrame({
        "AAPL": [100.0, invalid_price, 105.0],
        "MSFT": [200.0, 202.0, 204.0],
    })

    with pytest.raises(ValueError, match="nonfinite"):
        validate_stress_prices(prices)
