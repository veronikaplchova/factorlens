
"""Tests for market regime detection."""

import numpy as np
import pandas as pd
import pytest

from src.regimes import calculate_rolling_volatility


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
