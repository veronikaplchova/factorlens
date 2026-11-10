
"""Market regime detection and analysis."""

import numpy as np
import pandas as pd


def calculate_rolling_volatility(
    market_returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Calculate annualized rolling market volatility."""

    if window < 2:
        raise ValueError("Rolling window must be at least 2.")

    if market_returns.empty:
        raise ValueError("Market returns must not be empty.")

    rolling_volatility = (
        market_returns.rolling(window=window).std()
        * np.sqrt(252)
    )

    return rolling_volatility




def classify_market_regimes(
    rolling_volatility: pd.Series,
    threshold_window: int = 126,
) -> pd.Series:
    """Classify market volatility using a historical median."""

    if threshold_window < 2:
        raise ValueError("Threshold window must be at least 2.")

    if rolling_volatility.empty:
        raise ValueError("Rolling volatility must not be empty.")

    # Use only previous observations to estimate the threshold.
    historical_threshold = (
        rolling_volatility.shift(1)
        .rolling(window=threshold_window)
        .median()
    )

    regimes = pd.Series(
        pd.NA,
        index=rolling_volatility.index,
        dtype="object",
    )

    valid_dates = (
        rolling_volatility.notna()
        & historical_threshold.notna()
    )

    regimes.loc[valid_dates] = np.where(
        rolling_volatility.loc[valid_dates]
        > historical_threshold.loc[valid_dates],
        "High Volatility",
        "Low Volatility",
    )

    return regimes
