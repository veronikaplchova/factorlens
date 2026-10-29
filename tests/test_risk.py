
"""Tests for portfolio risk contribution calculations."""

import numpy as np
import pandas as pd
import pytest

from src.risk import calculate_risk_contributions


def test_equal_risk_contributions():
    """Identical assets with equal weights contribute equally."""
    returns = pd.DataFrame({
        "AAPL": [0.01, -0.02, 0.03, -0.01],
        "MSFT": [0.01, -0.02, 0.03, -0.01],
    })

    result = calculate_risk_contributions(
        returns, weights=[0.5, 0.5]
    )

    np.testing.assert_allclose(result.values, [50.0, 50.0])


def test_risk_contributions_sum_to_100():
    """All contributions together must equal 100%."""
    returns = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03],
        "MSFT": [-0.01, 0.03, 0.01, -0.02],
        "JPM": [0.02, -0.01, 0.02, 0.01],
    })

    result = calculate_risk_contributions(
        returns, weights=[0.4, 0.4, 0.2]
    )

    assert result.sum() == pytest.approx(100.0)


def test_invalid_weights_are_rejected():
    """Portfolio weights must add up to 100%."""
    returns = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01],
        "MSFT": [0.02, -0.01, 0.03],
    })

    with pytest.raises(ValueError, match="add up to 1"):
        calculate_risk_contributions(
            returns, weights=[0.6, 0.6]
        )


def test_missing_returns_are_rejected():
    """Missing return observations should cause an error."""
    returns = pd.DataFrame({
        "AAPL": [0.01, np.nan, 0.02],
        "MSFT": [0.02, 0.01, -0.01],
    })

    with pytest.raises(ValueError, match="missing values"):
        calculate_risk_contributions(
            returns, weights=[0.5, 0.5]
        )
