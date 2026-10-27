
"""Run a portfolio example using downloaded stock prices."""

from src.data_loader import download_prices
from src.returns import (
    calculate_daily_returns,
    calculate_portfolio_returns,
)
from src.performance import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_growth,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
)


def main():
    tickers = ["AAPL", "MSFT", "JPM"]
    weights = [0.40, 0.40, 0.20]

    prices = download_prices(
        tickers=tickers,
        start="2025-01-01",
    )

    # Keep asset columns aligned with their weights.
    prices = prices.loc[:, tickers]

    daily_returns = calculate_daily_returns(prices)

    portfolio_returns = calculate_portfolio_returns(
        daily_returns,
        weights=weights,
    )

    if portfolio_returns.isna().any():
        raise ValueError(
            "Portfolio returns contain missing values."
        )

    initial_investment = 10_000.0

    portfolio_value = calculate_cumulative_growth(
        portfolio_returns,
        initial_investment=initial_investment,
    )

    total_return = calculate_total_return(portfolio_returns)

    annualized_return = calculate_annualized_return(
        portfolio_returns
    )

    annualized_volatility = calculate_annualized_volatility(
        portfolio_returns
    )

    sharpe_ratio = calculate_sharpe_ratio(
        portfolio_returns,
        risk_free_rate=0.0,
    )

    max_drawdown = calculate_max_drawdown(
        portfolio_returns
    )

    print("\n=== PORTFOLIO PERFORMANCE REPORT ===")

    print(f"\nInitial investment: ${initial_investment:,.2f}")
    print(f"Final value: ${portfolio_value.iloc[-1]:,.2f}")

    print(f"\nTotal return: {total_return:.2%}")
    print(f"Annualized return: {annualized_return:.2%}")
    print(f"Annualized volatility: {annualized_volatility:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    print("\nLatest portfolio values ($):")
    print(portfolio_value.tail().round(2))


if __name__ == "__main__":
    main()
