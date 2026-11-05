
"""Tests for historical portfolio stress calculations."""

import numpy as np
import pandas as pd
import pytest

from src.stress import (
    calculate_stress_period,
    validate_stress_prices,
    validate_stress_coverage,
    interpret_stress_performance,
    calculate_stress_contributions,
)

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


def test_validate_stress_prices_valid():
    """Accept complete, positive historical stock prices."""

    prices = pd.DataFrame({
        "AAPL": [100.0, 102.0, 105.0],
        "MSFT": [200.0, 198.0, 204.0],
        "SPY": [400.0, 405.0, 403.0],
    })

    assert validate_stress_prices(prices) is None

def test_validate_stress_coverage_missing_dates():
    """Reject portfolio data with missing benchmark trading dates."""

    portfolio_dates = pd.to_datetime([
        "2020-03-02",
        "2020-03-04",
    ])

    benchmark_dates = pd.to_datetime([
        "2020-03-02",
        "2020-03-03",
        "2020-03-04",
    ])

    with pytest.raises(ValueError, match="missing 1 trading days"):
        validate_stress_coverage(
            portfolio_dates,
            benchmark_dates,
        )

def test_validate_stress_coverage_complete():
    """Accept portfolio data covering all benchmark trading dates."""

    portfolio_dates = pd.to_datetime([
        "2020-03-02",
        "2020-03-03",
        "2020-03-04",
    ])

    benchmark_dates = pd.to_datetime([
        "2020-03-02",
        "2020-03-03",
        "2020-03-04",
    ])

    assert validate_stress_coverage(
        portfolio_dates,
        benchmark_dates,
    ) is None


@pytest.mark.parametrize(
    "outperformance, expected",
    [
        (0.09, "outperformed"),
        (-0.03, "underperformed"),
        (0.0, "matched"),
    ],
)
def test_interpret_stress_performance(outperformance, expected):
    """Correctly classify performance relative to SPY."""

    assert interpret_stress_performance(outperformance) == expected


def test_calculate_stress_contributions():
    """Verify individual stocks' daily return contributions."""

    asset_returns = pd.DataFrame({
        "AAPL": [0.10, -0.05],
        "MSFT": [-0.02, 0.04],
    })

    weights = [0.60, 0.40]

    result = calculate_stress_contributions(
        asset_returns,
        weights,
    )

    assert result.loc[0, "AAPL"] == pytest.approx(0.06)
    assert result.loc[0, "MSFT"] == pytest.approx(-0.008)
    assert result.loc[1, "AAPL"] == pytest.approx(-0.03)
    assert result.loc[1, "MSFT"] == pytest.approx(0.016)


def test_stress_contributions_invalid_weights():
    """Reject weights that do not match the number of assets."""

    asset_returns = pd.DataFrame({
        "AAPL": [0.02, -0.01],
        "MSFT": [0.03, 0.01],
    })

    weights = [1.0]

    with pytest.raises(ValueError, match="Weights must match"):
        calculate_stress_contributions(
            asset_returns,
            weights,
        )


def test_stress_contributions_empty_data():
    """Reject empty historical asset returns."""

    asset_returns = pd.DataFrame()
    weights = [0.5, 0.5]

    with pytest.raises(ValueError, match="must not be empty"):
        calculate_stress_contributions(
            asset_returns,
            weights,
        )


def test_stress_contributions_weights_sum():
    """Reject portfolio weights that do not sum to 100%."""

    asset_returns = pd.DataFrame({
        "AAPL": [0.02, -0.01],
        "MSFT": [0.03, 0.01],
    })

    weights = [0.60, 0.60]

    with pytest.raises(ValueError, match="sum to 100%"):
        calculate_stress_contributions(
            asset_returns,
            weights,
        )

def test_stress_contributions_negative_weights():
    """Reject negative portfolio allocations."""

    asset_returns = pd.DataFrame({
        "AAPL": [0.02, -0.01],
        "MSFT": [0.03, 0.01],
    })

    weights = [1.20, -0.20]

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_stress_contributions(
            asset_returns,
            weights,
        )

@pytest.mark.parametrize("invalid_weight", [np.nan, np.inf, -np.inf])
def test_stress_contributions_nonfinite_weights(invalid_weight):
    """Reject missing or infinite portfolio weights."""

    asset_returns = pd.DataFrame({
        "AAPL": [0.02, -0.01],
        "MSFT": [0.03, 0.01],
    })

    weights = [invalid_weight, 0.40]

    with pytest.raises(ValueError, match="must be finite"):
        calculate_stress_contributions(
            asset_returns,
            weights,
        )
