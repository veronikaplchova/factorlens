
"""Tests for CAPM factor regression calculations."""

import numpy as np
import pandas as pd
import pytest

from src.factors import calculate_capm, calculate_ff3


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


def test_ff3_known_coefficients():
    """Recover known three-factor regression coefficients."""

    dates = pd.date_range("2025-01-01", periods=10)

    factors = pd.DataFrame(
        {
            "Mkt-RF": [
                0.01, -0.02, 0.03, 0.005, -0.01,
                0.02, -0.015, 0.025, 0.008, -0.005,
            ],
            "SMB": [
                0.003, -0.004, 0.002, 0.006, -0.002,
                0.001, 0.005, -0.003, 0.004, -0.001,
            ],
            "HML": [
                -0.002, 0.005, -0.003, 0.001, 0.004,
                -0.005, 0.002, 0.003, -0.004, 0.006,
            ],
            "RF": [0.0001] * 10,
        },
        index=dates,
    )

    portfolio = (
        factors["RF"]
        + 0.0005
        + 1.2 * factors["Mkt-RF"]
        + 0.4 * factors["SMB"]
        - 0.3 * factors["HML"]
    )

    result = calculate_ff3(portfolio, factors)

    assert result["alpha"] == pytest.approx(0.0005)
    assert result["market_beta"] == pytest.approx(1.2)
    assert result["smb_beta"] == pytest.approx(0.4)
    assert result["hml_beta"] == pytest.approx(-0.3)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["n_observations"] == 10


def test_ff3_missing_factor():
    """Reject datasets missing a required factor."""

    dates = pd.date_range("2025-01-01", periods=6)

    factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01] * 6,
            "SMB": [0.002] * 6,
            "RF": [0.0001] * 6,
        },
        index=dates,
    )

    portfolio = pd.Series([0.01] * 6, index=dates)

    with pytest.raises(ValueError, match="Required Fama-French"):
        calculate_ff3(portfolio, factors)


def test_ff3_insufficient_observations():
    """Reject datasets with too few observations."""

    dates = pd.date_range("2025-01-01", periods=3)

    factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02, -0.01],
            "SMB": [0.002, -0.003, 0.001],
            "HML": [0.003, 0.001, -0.002],
            "RF": [0.0001] * 3,
        },
        index=dates,
    )

    portfolio = pd.Series(
        [0.01, 0.02, -0.01],
        index=dates,
    )

    with pytest.raises(ValueError, match="five observations"):
        calculate_ff3(portfolio, factors)
