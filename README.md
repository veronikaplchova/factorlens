# FactorLens

A Python portfolio-analysis project exploring investment performance, diversification, and the factors that drive stock returns.

The project currently provides data-loading and return-calculation functions, plus a runnable portfolio example. An interactive Streamlit dashboard, factor models, and stress tests are planned.

## Current Features

- Stock-price downloads using `yfinance`
- Ticker and price-data validation
- Portfolio-weight validation
- Daily and monthly asset returns
- Weighted portfolio returns
- Compounded investment growth and total return
- Unit tests for return calculations and validation

## Getting Started

Run the following commands from the project folder.

### Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If you already have a virtual environment, only run the activation command.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the tests

```bash
python -m pytest
```

### Run the portfolio example

```bash
python -m src.demo_portfolio
```

The example downloads prices starting from January 2025 and analyses a portfolio with these weights:

| Stock | Ticker | Weight |
|---|---|---|
| Apple | AAPL | 40% |
| Microsoft | MSFT | 40% |
| JPMorgan Chase | JPM | 20% |

It prints recent asset and portfolio returns, the growth of an initial $10,000 investment, and the total return over the available period.

To change the example, edit `tickers`, `weights`, and the start date in `src/demo_portfolio.py`. Weights must follow the ticker order.

## Project Structure

| File | Purpose |
|---|---|
| `src/data_loader.py` | Download stock-price data |
| `src/validation.py` | Validate inputs and price data |
| `src/returns.py` | Calculate asset and portfolio returns |
| `src/demo_portfolio.py` | Run the portfolio example |
| `tests/test_returns.py` | Test return calculations |
| `tests/test_validation.py` | Test validation functions |
| `requirements.txt` | List Python dependencies |

## Methodology and Assumptions

Asset returns are calculated as simple percentage changes:

```text
Return = Current price / Previous price - 1
```

Portfolio returns are the weighted sum of asset returns:

```text
Portfolio return = Sum of (Asset weight × Asset return)
```

Investment growth compounds successive portfolio returns:

```text
Portfolio value = Initial investment × Product of (1 + Daily return)
```

The current example:

- Assumes daily rebalancing to the specified portfolio weights.
- Reports USD performance without currency conversion.
- Excludes transaction costs, taxes, and slippage.
- Requires all asset returns to be available for each portfolio return.
- Stops the growth calculation if portfolio returns contain missing values.
- Uses each month's final available price for monthly asset returns.

The first monthly return requires a preceding month-end price. A month still in progress may represent a partial month.

Downloaded data may be revised or unavailable. Results depend on the selected assets, dates, and data quality.

## Development Roadmap

1. **Foundation:** data loading, validation, return calculations, and tests.
2. **Performance dashboard:** reusable performance metrics, growth and drawdown charts, benchmark comparison, and initial Streamlit pages.
3. **Risk analysis:** correlations, rolling volatility and beta, Value at Risk, Expected Shortfall, and risk contributions.
4. **Factor modelling:** CAPM and Fama–French models, robust standard errors, diagnostics, and exposure interpretation.
5. **Regimes and stress tests:** market-regime comparisons, historical crises, and hypothetical scenarios.
6. **Professional polish:** documentation, broader testing, dashboard refinement, deployment, and a version 1.0 release.

## License

This project is licensed under the MIT License. See `LICENSE` for details.