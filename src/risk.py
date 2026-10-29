
"""Portfolio risk and diversification calculations."""

import numpy as np
import pandas as pd


def calculate_risk_contributions(
    asset_returns: pd.DataFrame,
    weights: list[float],
) -> pd.Series:
    """Calculate each asset's contribution to portfolio volatility."""

    if asset_returns.empty or len(asset_returns) < 2:
        raise ValueError("At least two observations are required.")

    if asset_returns.isna().any().any():
        raise ValueError("Returns cannot contain missing values.")

    if not np.isfinite(asset_returns.to_numpy()).all():
        raise ValueError("Returns must be finite.")

    w = np.asarray(weights, dtype=float)

    if len(w) != asset_returns.shape[1]:
        raise ValueError("Weights must match the number of assets.")

    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")

    if not np.isclose(w.sum(), 1.0):
        raise ValueError("Weights must add up to 1.")

    covariance = asset_returns.cov().to_numpy()
    portfolio_variance = float(w @ covariance @ w)

    if portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive.")

    contributions = w * (covariance @ w) / portfolio_variance

    return pd.Series(
        contributions * 100,
        index=asset_returns.columns,
        name="Risk Contribution (%)",
    )
