
"""Factor models for portfolio return analysis."""

import numpy as np
import pandas as pd


def calculate_capm(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """Estimate CAPM alpha, beta, and R-squared using OLS."""

    data = pd.concat(
        [portfolio_returns, market_returns],
        axis=1,
        join="inner",
    ).dropna()

    if len(data) < 3:
        raise ValueError("At least three observations are required.")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Returns must be finite.")

    y = data.iloc[:, 0].to_numpy(dtype=float) - risk_free_rate
    x = data.iloc[:, 1].to_numpy(dtype=float) - risk_free_rate

    if np.var(x) <= 0:
        raise ValueError("Market returns must vary.")

    beta, alpha = np.polyfit(x, y, 1)

    predicted = alpha + beta * x
    residuals = y - predicted

    total_ss = np.sum((y - y.mean()) ** 2)
    residual_ss = np.sum(residuals ** 2)

    r_squared = (
        1 - residual_ss / total_ss
        if total_ss > 0
        else np.nan
    )

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "r_squared": float(r_squared),
    }


def calculate_ff3(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> dict:
    """Estimate Fama-French three-factor regression using OLS."""

    required = ["Mkt-RF", "SMB", "HML", "RF"]

    if not set(required).issubset(factor_data.columns):
        raise ValueError("Required Fama-French factors are missing.")

    data = pd.concat(
        [
            portfolio_returns.rename("Portfolio"),
            factor_data[required],
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(data) < 5:
        raise ValueError("At least five observations are required.")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Returns and factors must be finite.")

    y = (
        data["Portfolio"] - data["RF"]
    ).to_numpy(dtype=float)

    x = data[["Mkt-RF", "SMB", "HML"]].to_numpy(dtype=float)

    design_matrix = np.column_stack(
        [np.ones(len(x)), x]
    )

    if np.linalg.matrix_rank(design_matrix) < 4:
        raise ValueError("Factors are linearly dependent.")

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        y,
        rcond=None,
    )

    predicted = design_matrix @ coefficients

    total_ss = np.sum((y - y.mean()) ** 2)
    residual_ss = np.sum((y - predicted) ** 2)

    r_squared = (
        1 - residual_ss / total_ss
        if total_ss > 0
        else np.nan
    )

    return {
        "alpha": float(coefficients[0]),
        "market_beta": float(coefficients[1]),
        "smb_beta": float(coefficients[2]),
        "hml_beta": float(coefficients[3]),
        "r_squared": float(r_squared),
        "n_observations": len(data),
    }



def calculate_ff5(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
) -> dict:
    """Estimate Fama-French five-factor regression using OLS."""

    required = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

    if not set(required).issubset(factor_data.columns):
        raise ValueError("Required five-factor columns are missing.")

    data = pd.concat(
        [portfolio_returns.rename("Portfolio"), factor_data[required]],
        axis=1,
        join="inner",
    ).dropna()

    if len(data) < 7:
        raise ValueError("At least seven observations are required.")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Returns and factors must be finite.")

    y = (data["Portfolio"] - data["RF"]).to_numpy(dtype=float)
    x = data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].to_numpy(dtype=float)

    design_matrix = np.column_stack([np.ones(len(x)), x])

    if np.linalg.matrix_rank(design_matrix) < 6:
        raise ValueError("Factors are linearly dependent.")

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix, y, rcond=None
    )

    predicted = design_matrix @ coefficients
    total_ss = np.sum((y - y.mean()) ** 2)
    residual_ss = np.sum((y - predicted) ** 2)

    r_squared = (
        1 - residual_ss / total_ss
        if total_ss > 0
        else np.nan
    )

    return {
        "alpha": float(coefficients[0]),
        "market_beta": float(coefficients[1]),
        "smb_beta": float(coefficients[2]),
        "hml_beta": float(coefficients[3]),
        "rmw_beta": float(coefficients[4]),
        "cma_beta": float(coefficients[5]),
        "r_squared": float(r_squared),
        "n_observations": len(data),
    }
