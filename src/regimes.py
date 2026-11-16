
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



def calculate_regime_performance(
    regime_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize portfolio returns across market regimes."""

    required_columns = {"Portfolio Return", "Market Regime"}

    if not required_columns.issubset(regime_data.columns):
        raise ValueError("Required regime analysis columns are missing.")

    if regime_data.empty:
        raise ValueError("Regime analysis data must not be empty.")

    summary = (
        regime_data.groupby("Market Regime")["Portfolio Return"]
        .agg(
            Average_Daily_Return="mean",
            Annualized_Volatility=lambda x: x.std() * np.sqrt(252),
            Observations="count",
        )
    )

    return summary

