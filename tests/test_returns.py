"""Tests for asset and portfolio return calculations."""

import numpy as np
import pandas as pd
import pytest

from src.returns import (
    calculate_daily_returns,
    calculate_monthly_returns,
    calculate_portfolio_returns,
)


def test_calculate_daily_returns():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [200.0, 180.0, 198.0],
        },
        index=pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03"]
        ),
    )

    result = calculate_daily_returns(prices)

    np.testing.assert_allclose(result["AAPL"], [0.10, 0.10])
    np.testing.assert_allclose(result["MSFT"], [-0.10, 0.10])


def test_calculate_monthly_returns():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [200.0, 180.0, 200.0],
        },
        index=pd.to_datetime(
            ["2026-01-31", "2026-02-28", "2026-03-31"]
        ),
    )

    result = calculate_monthly_returns(prices)

    np.testing.assert_allclose(result["AAPL"], [0.10, 0.10])
    np.testing.assert_allclose(
        result["MSFT"],
        [-0.10, 200 / 180 - 1],
    )


def test_calculate_portfolio_returns():
    asset_returns = pd.DataFrame(
        {
            "AAPL": [0.10, -0.02],
            "MSFT": [0.05, 0.03],
        }
    )

    result = calculate_portfolio_returns(
        asset_returns,
        weights=[0.60, 0.40],
    )

    np.testing.assert_allclose(result, [0.08, 0.00])
    assert result.name == "Portfolio"


def test_portfolio_return_requires_complete_asset_data():
    asset_returns = pd.DataFrame(
        {
            "AAPL": [0.10, np.nan],
            "MSFT": [0.05, 0.03],
        }
    )

    result = calculate_portfolio_returns(
        asset_returns,
        weights=[0.60, 0.40],
    )

    assert np.isnan(result.iloc[1])


def test_daily_returns_reject_empty_prices():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_daily_returns(pd.DataFrame())


def test_portfolio_returns_reject_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_portfolio_returns(
            pd.DataFrame(),
            weights=[],
        )