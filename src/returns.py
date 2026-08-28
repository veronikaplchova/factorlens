"""Calculate asset and portfolio returns."""

from collections.abc import Sequence

import pandas as pd

from src.validation import validate_weights


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns from a table of prices."""
    if prices.empty:
        raise ValueError("The price table cannot be empty.")

    daily_returns = prices.pct_change(fill_method=None)
    daily_returns = daily_returns.dropna(how="all")

    return daily_returns


def calculate_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly returns using each month's final available price."""
    if prices.empty:
        raise ValueError("The price table cannot be empty.")

    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    monthly_returns = monthly_returns.dropna(how="all")

    return monthly_returns


def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """Calculate portfolio returns from asset returns and portfolio weights."""
    if asset_returns.empty:
        raise ValueError("The asset-return table cannot be empty.")

    validated_weights = validate_weights(
        weights,
        number_of_assets=asset_returns.shape[1],
    )

    weighted_returns = asset_returns.mul(
        validated_weights,
        axis="columns",
    )

    portfolio_returns = weighted_returns.sum(
        axis="columns",
        min_count=asset_returns.shape[1],
    )

    portfolio_returns.name = "Portfolio"

    return portfolio_returns