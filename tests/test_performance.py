
"""Tests for portfolio performance calculations."""

import numpy as np
import pandas as pd
import pytest



from src.performance import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_growth,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
)



def test_cumulative_growth():
    returns = pd.Series([0.10, -0.05, 0.02])

    result = calculate_cumulative_growth(
        returns, initial_investment=1000
    )

    expected = [1100.0, 1045.0, 1065.9]

    np.testing.assert_allclose(result.values, expected)


def test_total_return():
    returns = pd.Series([0.10, -0.05, 0.02])

    result = calculate_total_return(returns)

    assert result == pytest.approx(0.0659)


def test_annualized_return():
    returns = pd.Series([0.01] * 12)

    result = calculate_annualized_return(
        returns, periods_per_year=12
    )

    expected = (1.01 ** 12) - 1

    assert result == pytest.approx(expected)


def test_invalid_returns():
    returns = pd.Series([0.05, np.nan])

    with pytest.raises(ValueError):
        calculate_total_return(returns)


def test_invalid_initial_investment():
    returns = pd.Series([0.05, 0.02])

    with pytest.raises(ValueError):
        calculate_cumulative_growth(
            returns, initial_investment=-1000
        )


def test_invalid_periods_per_year():
    returns = pd.Series([0.05, 0.02])

    with pytest.raises(ValueError):
        calculate_annualized_return(
            returns, periods_per_year=0
        )



def test_annualized_volatility():
    returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    result = calculate_annualized_volatility(
        returns, periods_per_year=252
    )

    expected = returns.std(ddof=1) * np.sqrt(252)

    assert result == pytest.approx(expected)


def test_volatility_requires_two_returns():
    returns = pd.Series([0.05])

    with pytest.raises(ValueError):
        calculate_annualized_volatility(returns)




def test_sharpe_ratio():
    returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    result = calculate_sharpe_ratio(
        returns, risk_free_rate=0.0, periods_per_year=252
    )

    expected = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(252)
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_zero_volatility():
    returns = pd.Series([0.01, 0.01, 0.01])

    with pytest.raises(ValueError):
        calculate_sharpe_ratio(returns)


def test_sharpe_ratio_invalid_risk_free_rate():
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError):
        calculate_sharpe_ratio(
            returns, risk_free_rate=-1.5
        )



def test_calculate_drawdown():
    returns = pd.Series([0.20, -0.25, 0.10])

    result = calculate_drawdown(returns)

    expected = [0.0, -0.25, -0.175]

    np.testing.assert_allclose(result.values, expected)


def test_calculate_max_drawdown():
    returns = pd.Series([0.20, -0.25, 0.10])

    result = calculate_max_drawdown(returns)

    assert result == pytest.approx(-0.25)


def test_drawdown_no_losses():
    returns = pd.Series([0.05, 0.10, 0.03])

    result = calculate_max_drawdown(returns)

    assert result == pytest.approx(0.0)
