
"""Tests for CAPM factor regression calculations."""

import numpy as np
import pandas as pd
import pytest

from src.factors import calculate_capm, calculate_ff3, calculate_ff5


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


def test_ff5_known_coefficients():
    """Check whether FF5 recovers known factor loadings."""

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=30)

    factors = pd.DataFrame(
        rng.normal(0, 0.01, size=(30, 5)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    )
    factors["RF"] = 0.0001

    portfolio = (
        factors["RF"]
        + 0.0005
        + 1.2 * factors["Mkt-RF"]
        + 0.4 * factors["SMB"]
        - 0.3 * factors["HML"]
        + 0.6 * factors["RMW"]
        - 0.2 * factors["CMA"]
    )

    result = calculate_ff5(portfolio, factors)

    assert result["alpha"] == pytest.approx(0.0005)
    assert result["market_beta"] == pytest.approx(1.2)
    assert result["smb_beta"] == pytest.approx(0.4)
    assert result["hml_beta"] == pytest.approx(-0.3)
    assert result["rmw_beta"] == pytest.approx(0.6)
    assert result["cma_beta"] == pytest.approx(-0.2)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["n_observations"] == 30


def test_ff5_missing_factor():
    """Check that missing factor columns are rejected."""

    dates = pd.bdate_range("2025-01-01", periods=10)

    factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01] * 10,
            "SMB": [0.002] * 10,
            "HML": [0.003] * 10,
            "RMW": [0.004] * 10,
            "RF": [0.0001] * 10,
        },
        index=dates,
    )

    portfolio = pd.Series([0.01] * 10, index=dates)

    with pytest.raises(ValueError, match="five-factor columns"):
        calculate_ff5(portfolio, factors)


def test_ff5_insufficient_observations():
    """Check that fewer than seven observations are rejected."""

    dates = pd.bdate_range("2025-01-01", periods=5)

    factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01, -0.02, 0.03, 0.01, -0.01],
            "SMB": [0.002, 0.001, -0.003, 0.004, -0.002],
            "HML": [0.003, -0.002, 0.001, 0.002, -0.001],
            "RMW": [0.004, 0.001, -0.002, 0.003, -0.003],
            "CMA": [0.001, -0.003, 0.002, -0.002, 0.004],
            "RF": [0.0001] * 5,
        },
        index=dates,
    )

    portfolio = pd.Series([0.01] * 5, index=dates)

    with pytest.raises(ValueError, match="seven observations"):
        calculate_ff5(portfolio, factors)


def test_ff5_r_squared_at_least_ff3():
    """Adding factors cannot reduce in-sample OLS R-squared."""

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=100)

    factors = pd.DataFrame(
        rng.normal(0, 0.01, size=(100, 5)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    )
    factors["RF"] = 0.0001

    portfolio = (
        factors["RF"]
        + 0.0003
        + 1.1 * factors["Mkt-RF"]
        + 0.3 * factors["SMB"]
        - 0.2 * factors["HML"]
        + 0.5 * factors["RMW"]
        - 0.4 * factors["CMA"]
        + rng.normal(0, 0.002, size=100)
    )

    ff3 = calculate_ff3(portfolio, factors)
    ff5 = calculate_ff5(portfolio, factors)

    assert ff3["n_observations"] == ff5["n_observations"]
    assert ff5["r_squared"] >= ff3["r_squared"] - 1e-10
