
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
