"""Run a simple portfolio example using downloaded stock prices."""

from src.data_loader import download_prices
from src.returns import (
    calculate_daily_returns,
    calculate_portfolio_returns,
)


def main():
    tickers = ["AAPL", "MSFT", "JPM"]
    weights = [0.40, 0.40, 0.20]

    prices = download_prices(
        tickers=tickers,
        start="2025-01-01",
    )

    # Keep columns in the same order as the portfolio weights.
    prices = prices.loc[:, tickers]

    daily_returns = calculate_daily_returns(prices)

    portfolio_returns = calculate_portfolio_returns(
        daily_returns,
        weights=weights,
    )

    print("\nLatest daily asset returns (%):")
    print((daily_returns.tail() * 100).round(2))

    print("\nLatest daily portfolio returns (%):")
    print((portfolio_returns.tail() * 100).round(2))




    if portfolio_returns.isna().any():
        raise ValueError(
            "Portfolio returns contain missing values; "
            "resolve these before calculating growth."
        )

    initial_investment = 10_000.0
    portfolio_value = (
        initial_investment
        * (1 + portfolio_returns).cumprod()
    )

    total_return = (
        portfolio_value.iloc[-1] / initial_investment - 1
    )

    print("\nLatest portfolio values ($):")
    print(portfolio_value.tail().round(2))

    print(f"\nInitial investment: ${initial_investment:,.2f}")
    print(f"Final value: ${portfolio_value.iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")

if __name__ == "__main__":
    main()
    