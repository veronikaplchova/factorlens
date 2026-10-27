
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
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
)




from src.visualization import (
    plot_drawdown,
    plot_portfolio_growth,
    plot_portfolio_vs_benchmark,
)





def main():
    tickers = ["AAPL", "MSFT", "JPM"]
    weights = [0.40, 0.40, 0.20]

    prices = download_prices(
        tickers=tickers,
        start="2025-01-01",
    )


    benchmark_prices = download_prices(
        tickers=["SPY"],
        start="2025-01-01",
    )


    # Keep asset columns aligned with their weights.
    prices = prices.loc[:, tickers]

    daily_returns = calculate_daily_returns(prices)

    portfolio_returns = calculate_portfolio_returns(
        daily_returns,
        weights=weights,
    )


    benchmark_returns = calculate_daily_returns(
        benchmark_prices
    )["SPY"]

    # Compare only dates available for both investments.
    common_dates = portfolio_returns.index.intersection(
        benchmark_returns.index
    )

    portfolio_returns = portfolio_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]



    if portfolio_returns.isna().any():
        raise ValueError(
            "Portfolio returns contain missing values."
        )

    initial_investment = 10_000.0

    portfolio_value = calculate_cumulative_growth(
        portfolio_returns,
        initial_investment=initial_investment,
    )


    benchmark_value = calculate_cumulative_growth(
        benchmark_returns,
        initial_investment=initial_investment,
    )

    benchmark_total_return = calculate_total_return(
        benchmark_returns
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
    print(f"Benchmark (SPY) return: {benchmark_total_return:.2%}")

    outperformance = total_return - benchmark_total_return

    print(f"Outperformance vs SPY: {outperformance:+.2%}")
    print(f"Annualized return: {annualized_return:.2%}")
    print(f"Annualized volatility: {annualized_volatility:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    print("\nLatest portfolio values ($):")
    print(portfolio_value.tail().round(2))

    print("\nDisplaying portfolio growth chart...")
    plot_portfolio_growth(portfolio_value)


    drawdown = calculate_drawdown(portfolio_returns)

    print("\nDisplaying portfolio drawdown chart...")
    plot_drawdown(drawdown)


    print("\nDisplaying portfolio vs. benchmark chart...")

    plot_portfolio_vs_benchmark(
        portfolio_value,
        benchmark_value,
    )


if __name__ == "__main__":
    main()
