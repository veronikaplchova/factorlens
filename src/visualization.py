
"""Create visualizations for portfolio analysis."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_portfolio_growth(
    portfolio_values: pd.Series,
) -> None:
    """Plot the value of a portfolio over time."""

    if portfolio_values.empty:
        raise ValueError("Portfolio values cannot be empty.")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        portfolio_values.index,
        portfolio_values.values,
        linewidth=2,
        label="Portfolio value",
    )

    ax.set_title("Portfolio Growth Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_drawdown(
    drawdown: pd.Series,
) -> None:
    """Plot portfolio drawdown over time."""

    if drawdown.empty:
        raise ValueError("Drawdown series cannot be empty.")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(
        drawdown.index,
        drawdown.values * 100,
        0,
        alpha=0.4,
        label="Drawdown",
    )

    ax.plot(
        drawdown.index,
        drawdown.values * 100,
        linewidth=1.5,
    )

    ax.set_title("Portfolio Drawdown Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_portfolio_vs_benchmark(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
) -> None:
    """Compare portfolio growth against a market benchmark."""

    if portfolio_values.empty or benchmark_values.empty:
        raise ValueError("Portfolio and benchmark cannot be empty.")

    if not portfolio_values.index.equals(benchmark_values.index):
        raise ValueError("Portfolio and benchmark dates must match.")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        portfolio_values.index,
        portfolio_values.values,
        linewidth=2,
        label="Our Portfolio",
    )

    ax.plot(
        benchmark_values.index,
        benchmark_values.values,
        linewidth=2,
        label="S&P 500 (SPY)",
    )

    ax.set_title("Portfolio vs. S&P 500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Investment Value ($)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    plt.close(fig)
