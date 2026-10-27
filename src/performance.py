
"""Calculate portfolio performance metrics."""

import numpy as np
import pandas as pd


def validate_returns(returns: pd.Series) -> None:
    """Check that returns are complete, finite, and valid."""
    if returns.empty:
        raise ValueError("The return series cannot be empty.")

    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns must not contain missing or infinite values.")

    if (returns < -1).any():
        raise ValueError("Simple returns cannot be below -100%.")


def calculate_cumulative_growth(
    returns: pd.Series,
    initial_investment: float = 10_000.0,
) -> pd.Series:
    """Calculate investment value after each period's return."""
    validate_returns(returns)

    if not np.isfinite(initial_investment) or initial_investment <= 0:
        raise ValueError("Initial investment must be finite and positive.")

    portfolio_value = initial_investment * (1 + returns).cumprod()
    portfolio_value.name = "Portfolio Value"

    return portfolio_value


def calculate_total_return(returns: pd.Series) -> float:
    """Calculate the compounded return over the entire period."""
    validate_returns(returns)

    return float((1 + returns).prod() - 1)




def calculate_annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized compounded return."""
    validate_returns(returns)

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    total_growth = (1 + returns).prod()
    number_of_periods = len(returns)

    return float(total_growth ** (periods_per_year / number_of_periods) - 1)




def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility of portfolio returns."""
    validate_returns(returns)

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    if len(returns) < 2:
        raise ValueError("At least two returns are needed.")

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))



def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sharpe ratio."""
    validate_returns(returns)

    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("Risk-free rate must be finite and above -100%.")

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    if len(returns) < 2:
        raise ValueError("At least two returns are needed.")

    risk_free_per_period = (
        (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    )

    excess_returns = returns - risk_free_per_period
    volatility = returns.std(ddof=1)

    if np.isclose(volatility, 0):
        raise ValueError("Sharpe ratio is undefined for zero volatility.")

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )




def calculate_drawdown(
    returns: pd.Series,
) -> pd.Series:
    """Calculate drawdown from the previous portfolio peak."""
    validate_returns(returns)

    growth = (1 + returns).cumprod()

    running_peak = growth.cummax().clip(lower=1.0)

    drawdown = growth / running_peak - 1
    drawdown.name = "Drawdown"

    return drawdown


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    """Calculate the largest peak-to-trough portfolio decline."""
    drawdown = calculate_drawdown(returns)

    return float(drawdown.min())
