
"""Historical stress-testing calculations for portfolios."""

import numpy as np
import pandas as pd


STRESS_PERIODS = {
    "Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Market Downturn": ("2022-01-01", "2022-12-31"),
}


def calculate_stress_period(
    portfolio_returns: pd.Series,
    start_date: str,
    end_date: str,
) -> dict:
    """Calculate portfolio performance during a historical period."""

    if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise ValueError("Start date must be before end date.")

    period_returns = portfolio_returns.loc[
        start_date:end_date
    ].dropna()

    if period_returns.empty:
        raise ValueError("No returns available for stress period.")

    if not np.isfinite(period_returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns must be finite.")

    cumulative = (1 + period_returns).cumprod()

    total_return = cumulative.iloc[-1] - 1

    
    # Include initial portfolio wealth of 1.0 as a possible peak.
    running_peak = cumulative.cummax().clip(lower=1.0)
    drawdowns = cumulative / running_peak - 1

    max_drawdown = drawdowns.min()

    volatility = period_returns.std() * np.sqrt(252)

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "annualized_volatility": float(volatility),
        "n_observations": len(period_returns),
    }


def validate_stress_prices(prices: pd.DataFrame) -> None:
    """Check that historical stress-test prices are valid."""

    if prices.empty:
        raise ValueError("Historical price data is empty.")

    if prices.isna().any().any():
        raise ValueError("Historical prices contain missing values.")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("Historical prices contain nonfinite values.")

    if (prices <= 0).any().any():
        raise ValueError("Historical prices must be positive.")

def validate_stress_coverage(
    portfolio_dates: pd.DatetimeIndex,
    benchmark_dates: pd.DatetimeIndex,
) -> None:
    """Ensure portfolio data covers all benchmark trading dates."""

    missing_dates = benchmark_dates.difference(portfolio_dates)

    if len(missing_dates) > 0:
        raise ValueError(
            f"Portfolio data is missing {len(missing_dates)} "
            "trading days available in the S&P 500 benchmark."
        )

def interpret_stress_performance(outperformance: float) -> str:
    """Describe portfolio performance relative to the benchmark."""

    if not np.isfinite(outperformance):
        raise ValueError("Outperformance must be finite.")

    if outperformance > 0:
        return "outperformed"

    if outperformance < 0:
        return "underperformed"

    return "matched"
