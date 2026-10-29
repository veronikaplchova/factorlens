
"""Tests for factor regression diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import calculate_factor_diagnostics


def make_sample_data():
    """Create reproducible factor and portfolio returns."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=120)

    factors = pd.DataFrame(
        rng.normal(0, 0.01, size=(120, 3)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML"],
    )
    factors["RF"] = 0.0001

    noise = rng.normal(0, 0.002, size=120)

    portfolio = (
        factors["RF"]
        + 0.0005
        + 1.2 * factors["Mkt-RF"]
        + 0.4 * factors["SMB"]
        - 0.3 * factors["HML"]
        + noise
    )

    return portfolio, factors


def test_diagnostics_columns():
    """Check that all inference statistics are returned."""
    portfolio, factors = make_sample_data()

    result = calculate_factor_diagnostics(
        portfolio, factors, ["Mkt-RF", "SMB", "HML"]
    )

    assert list(result.columns) == [
        "Coefficient",
        "Standard Error",
        "t-statistic",
        "p-value",
        "CI Lower",
        "CI Upper",
    ]
    assert list(result.index) == [
        "const", "Mkt-RF", "SMB", "HML"
    ]


def test_diagnostics_coefficients():
    """Check that estimates recover the simulated exposures."""
    portfolio, factors = make_sample_data()

    result = calculate_factor_diagnostics(
        portfolio, factors, ["Mkt-RF", "SMB", "HML"]
    )

    assert result.loc["Mkt-RF", "Coefficient"] == pytest.approx(
        1.2, abs=0.05
    )
    assert result.loc["SMB", "Coefficient"] == pytest.approx(
        0.4, abs=0.05
    )
    assert result.loc["HML", "Coefficient"] == pytest.approx(
        -0.3, abs=0.05
    )


def test_diagnostics_confidence_intervals():
    """Confidence intervals should contain their estimates."""
    portfolio, factors = make_sample_data()

    result = calculate_factor_diagnostics(
        portfolio, factors, ["Mkt-RF", "SMB", "HML"]
    )

    assert (result["CI Lower"] <= result["Coefficient"]).all()
    assert (result["Coefficient"] <= result["CI Upper"]).all()
    assert result["p-value"].between(0, 1).all()
    assert (result["Standard Error"] >= 0).all()


def test_diagnostics_missing_factor():
    """Reject datasets missing a requested factor."""
    portfolio, factors = make_sample_data()

    with pytest.raises(ValueError, match="Required factor"):
        calculate_factor_diagnostics(
            portfolio, factors, ["RMW"]
        )


def test_diagnostics_invalid_lags():
    """Reject negative HAC lag values."""
    portfolio, factors = make_sample_data()

    with pytest.raises(ValueError, match="max_lags"):
        calculate_factor_diagnostics(
            portfolio,
            factors,
            ["Mkt-RF", "SMB", "HML"],
            max_lags=-1,
        )
