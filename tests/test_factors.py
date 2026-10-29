
"""Tests for CAPM factor regression calculations."""

import numpy as np
import pandas as pd
import pytest

from src.factors import calculate_capm


def test_capm_known_alpha_and_beta():
    """Recover known regression coefficients."""
    dates = pd.date_range("2026-01-01", periods=6)
    market = pd.Series(
        [0.01, -0.02, 0.03, 0.005, -0.01, 0.02],
        index=dates,
    )
    portfolio = 0.001 + 1.5 * market

    result = calculate_capm(portfolio, market)

    assert result["alpha"] == pytest.approx(0.001)
    assert result["beta"] == pytest.approx(1.5)
    assert result["r_squared"] == pytest.approx(1.0)


def test_capm_aligns_dates():
    """Use only dates shared by both return series."""
    dates = pd.date_range("2026-01-01", periods=5)
    market = pd.Series(
        [0.01, 0.02, -0.01, 0.03, 0.005],
        index=dates,
    )
    portfolio = 0.002 + 0.8 * market
    portfolio = portfolio.iloc[1:]

    result = calculate_capm(portfolio, market)

    assert result["beta"] == pytest.approx(0.8)
    assert result["alpha"] == pytest.approx(0.002)


def test_capm_rejects_insufficient_observations():
    """Regression requires at least three shared observations."""
    portfolio = pd.Series([0.01, 0.02])
    market = pd.Series([0.02, 0.03])

    with pytest.raises(ValueError, match="three observations"):
        calculate_capm(portfolio, market)


def test_capm_rejects_constant_market():
    """Beta cannot be estimated from constant market returns."""
    portfolio = pd.Series([0.01, -0.02, 0.03])
    market = pd.Series([0.02, 0.02, 0.02])

    with pytest.raises(ValueError, match="must vary"):
        calculate_capm(portfolio, market)


def test_capm_rejects_infinite_values():
    """Infinite returns are invalid inputs."""
    portfolio = pd.Series([0.01, np.inf, 0.03])
    market = pd.Series([0.01, -0.02, 0.02])

    with pytest.raises(ValueError, match="finite"):
        calculate_capm(portfolio, market)
