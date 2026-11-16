# FactorLens

**Quantitative Portfolio Intelligence | Python · Streamlit · Asset Pricing · Risk Analytics**

FactorLens is an interactive quantitative finance dashboard for constructing equity portfolios, evaluating historical performance and diversification, estimating asset-pricing factor exposures, and exploring portfolio behavior during market stress. It brings portfolio analytics and statistical modelling together in a transparent, visual research workflow.

**Stack:** Python · Streamlit · Pandas · NumPy · Plotly · statsmodels · yfinance · pandas-datareader · pytest

> **Status:** Implemented and tested locally. Public Streamlit deployment and dashboard screenshots are upcoming.

## At a glance

| Dashboard | What it provides |
| --- | --- |
| **Overview** | Total and annualized returns, annualized volatility, Sharpe ratio, maximum drawdown, effective holdings, and cumulative portfolio value versus SPY. |
| **Portfolio Builder** | Choose from 10 predefined US stocks, edit long-only weights, reset to equal weights, and view stock and sector allocations. |
| **Risk Analytics** | Volatility contribution by holding, correlation heatmap, Herfindahl–Hirschman Index (HHI), drawdown history, and rule-based risk insights. |
| **Factor Analysis** | CAPM, Fama–French 3-factor (FF3) and 5-factor (FF5) regression, adjusted-R² model comparison, and FF5 inference with HAC standard errors. |
| **Stress Testing** | Replay three historical downturns using selected holdings and weights, compare with SPY, and attribute compounded crisis returns to individual stocks. |
| **Market Regimes** | Classify historical market conditions as high or low volatility and compare portfolio returns and volatility across the two regimes. |

## Dashboard preview

*Screenshots will be added in the next project phase.*

## Methodology and key concepts

### 1. Portfolio construction and performance

The user chooses holdings from a predefined universe (AAPL, MSFT, NVDA, JPM, GS, XOM, JNJ, WMT, CAT, TSLA) and assigns **nonnegative weights summing to 100%**. Changing one weight redistributes the remaining allocation across other selected holdings; equal weighting can be restored with one click.

For a given trading day, portfolio return is computed as the weighted sum of individual stock returns:

**Portfolio return:** `r_p,t = Σ_i w_i × r_i,t`

Returns are based on **auto-adjusted daily closing prices** obtained via `yfinance`. The application compounds daily portfolio returns to produce cumulative value (illustrated using an initial investment of **$10,000**) and benchmarks the result against **SPY**, an ETF tracking the S&P 500.

Key performance measures:

- **Total return:** the compounded percentage gain or loss over the selected interval.
- **Annualized return:** compound growth expressed on a 252-trading-day-year basis.
- **Annualized volatility:** standard deviation of daily returns multiplied by `√252`; it measures historical return variability.
- **Sharpe ratio:** average excess daily return divided by daily return volatility and scaled by `√252`; the dashboard assumes a **0% annual risk-free rate** for this metric.
- **Maximum drawdown:** the largest peak-to-trough decline in compounded portfolio value over the selected period.

The weighted daily-return engine describes a **constant-weight return series**, not a transaction-level backtest with trading costs or an execution schedule.

### 2. Diversification and risk contributions

The Portfolio Builder summarizes allocations by stock and sector. Risk Analytics then considers whether the allocation is *actually diversified*:

- **Correlation:** measures how two stocks' daily returns co-move (from −1 to +1). The dashboard displays a correlation heatmap and identifies the strongest positive pair.
- **HHI (concentration index):** `Σ_i w_i²`. Higher values mean greater concentration in a smaller number of positions.
- **Effective holdings:** `1 / HHI`. This expresses weight concentration as an equivalent number of equally weighted holdings; it is **not** the number of independent risk factors.
- **Percentage risk contribution:** `100 × w_i(Σw)_i / (wᵀΣw)`, where `Σ` is the sample covariance matrix. This measures each position's share of total portfolio volatility; contributions sum to 100% and can be negative when a position offsets other risks.

This distinction is crucial: **a stock with a 20% portfolio weight can contribute more—or less—than 20% of total portfolio risk** because volatility and covariance both matter. The app compares position weights with estimated risk contributions and highlights disproportionate exposures.

### 3. Asset-pricing regressions: CAPM, FF3 and FF5

The **Factor Analysis** page has five sub-tabs: *CAPM*, *Fama–French 3*, *Fama–French 5*, *Model Comparison*, and *Statistical Tests*.

**CAPM (Capital Asset Pricing Model)** relates the portfolio's daily return to SPY's daily return:

`r_p,t − r_f = α + β_m (r_SPY,t − r_f) + ε_t`

- **Beta (`β`):** sensitivity to market moves; a beta above 1 indicates stronger historical responsiveness than the proxy market.
- **Alpha (`α`):** the fitted daily intercept; by itself, it does **not** demonstrate manager skill or statistically significant outperformance.
- **R²:** the share of variation in historical daily portfolio returns explained by the regression.

The dashboard's CAPM calculation assumes `r_f = 0`. Its benchmark is **SPY**, not the S&P 500 index return itself.

**Fama–French 3-factor model (FF3)** extends market exposure to include **size** and **value**:

`r_p,t − RF_t = α + β_m(Mkt−RF)_t + β_s SMB_t + β_h HML_t + ε_t`

**Fama–French 5-factor model (FF5)** adds **profitability** and **investment**:

`r_p,t − RF_t = α + β_m(Mkt−RF)_t + β_s SMB_t + β_h HML_t + β_r RMW_t + β_c CMA_t + ε_t`

| Factor | Meaning | Interpretation of a positive loading |
| --- | --- | --- |
| **Mkt−RF** | Market return minus the risk-free rate | Greater exposure to market excess returns |
| **SMB** | Small Minus Big (size) | Greater sensitivity to small-cap versus large-cap returns |
| **HML** | High Minus Low (value) | Greater sensitivity to value versus growth returns |
| **RMW** | Robust Minus Weak (profitability) | Greater sensitivity to robust-profitability stocks |
| **CMA** | Conservative Minus Aggressive (investment) | Greater sensitivity to conservative-investment stocks |

FF3 and FF5 use **daily factor returns from the Kenneth R. French Data Library**, converted from percentage to decimal units. Regressions subtract the **factor dataset's daily risk-free-rate series** from portfolio returns. These models estimate *historical statistical exposures*, not a decomposition of the actual portfolio holdings.

### 4. Model comparison and statistical inference

FF3 and FF5 are estimated on the **same overlapping trading days**, making their fit statistics directly comparable. FactorLens reports **R² and adjusted R²**, with the latter penalizing models for adding explanatory variables. A higher adjusted R² means a better **in-sample adjusted fit**, not necessarily stronger future predictions.

For **FF5**, the Statistical Tests tab reports coefficient estimates, t-statistics, p-values, 95% confidence intervals, and **heteroskedasticity-and-autocorrelation-consistent (HAC) standard errors** using `statsmodels` with **5 lags**. HAC inference is designed to be more robust to heteroskedasticity and serial correlation than conventional OLS standard errors.

At the displayed **5% significance level**, a p-value below 0.05 indicates statistical evidence against the corresponding zero-coefficient null hypothesis, under the model's assumptions. **Statistical significance is not the same as economic importance, causality, or predictive power.** The app also visualizes confidence intervals for individual FF5 factor loadings.

### 5. Historical stress testing

The Stress Testing page evaluates selected holdings with the **current weights**, but uses prices from predefined historical windows:

| Historical episode | Window in the application |
| --- | --- |
| Global Financial Crisis | 2008-09-01 to 2009-03-31 |
| COVID-19 Crash | 2020-02-19 to 2020-03-23 |
| 2022 Market Downturn | 2022-01-01 to 2022-12-31 |

The analysis shows crisis-period portfolio return, maximum drawdown, annualized volatility, SPY performance, portfolio outperformance in **percentage points**, and individual holdings' contributions to the **compounded** portfolio return. An internal reconciliation check verifies that holding contributions sum to the reported crisis return.

Historical coverage is checked before results are displayed. **Some current holdings were not publicly traded in 2008**, so certain portfolio/scenario combinations may be unavailable rather than producing misleading numbers. These exercises are **retrospective historical reconstructions, not hypothetical shocks or forecasts**.

### 6. Market regime detection

FactorLens uses a transparent rule rather than a machine-learning regime model:

1. Calculate **21-trading-day rolling annualized volatility** of SPY daily returns.
2. Compute the **median of the previous 126 rolling-volatility observations**, excluding the current observation via a one-day shift.
3. Label the current day **High Volatility** if volatility is above its historical threshold; otherwise label it **Low Volatility**.
4. Compare portfolio **average daily return**, **annualized volatility**, and **number of observations** across the classified regimes.

The rule is designed to avoid using the current day's volatility when estimating its threshold. It is a **relative historical classification**, not a prediction of future markets or a bull/bear detector. It requires enough observations to establish the rolling windows.

## Data and implementation

| Component | Implementation |
| --- | --- |
| Equity prices | `yfinance` downloads auto-adjusted daily closing prices |
| Market benchmark | `SPY` ETF return series |
| Factor data | Kenneth French daily five-factor data via `pandas-datareader` |
| Portfolio and risk calculations | Pandas and NumPy; sample covariance for risk attribution |
| Regressions | NumPy least squares for CAPM/FF3/FF5 estimates; `statsmodels` for FF5 HAC inference |
| Interface and visuals | Streamlit and Plotly; dark navy UI with purple/blue charts |
| Caching | Streamlit data caching with a one-hour TTL for core downloads |
| Automated testing | `pytest` (102 tests reported as passing in local development) |

**Data availability:** Equity and factor data are downloaded live from third-party sources and can be delayed, temporarily unavailable, or incompatible with a chosen date range. Factor-based views require **at least 30 shared trading observations** and otherwise show an explanatory message; independent dashboard sections remain available.

## Repository structure

```text
FactorLens/
├── app.py                 # Streamlit application and six dashboard tabs
├── README.md
├── requirements.txt       # Dependency list
├── src/
│   ├── data_loader.py     # Market price retrieval
│   ├── validation.py      # Input and data checks
│   ├── returns.py         # Daily/monthly and weighted portfolio returns
│   ├── performance.py     # Returns, Sharpe, volatility, drawdowns
│   ├── risk.py            # Covariance-based risk contributions
│   ├── factor_data.py     # Fama–French data retrieval and conversion
│   ├── factors.py         # CAPM, FF3, FF5 regression estimates
│   ├── diagnostics.py     # FF5 HAC standard errors and confidence intervals
│   ├── stress.py          # Historical crises and return attribution
│   ├── regimes.py         # Rolling volatility and regime classification
│   ├── visualization.py   # Matplotlib helpers for demo script
│   └── demo_portfolio.py  # Standalone example
└── tests/                 # Automated tests (reported separately)
```

## Run the dashboard locally

With Python installed, clone this repository and open a terminal at its root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit normally opens at **http://localhost:8501** on the same computer. An internet connection is required to retrieve market and factor data.

**Packaging note:** The currently uploaded `requirements.txt` needs a minor correction before these instructions will work on a fresh installation: add the `streamlit` dependency and remove the command `python -m streamlit run app.py` from the dependency file. This will be addressed before deployment.

## Testing and quality controls

The development version has **102 automated tests reported as passing**, alongside manual checks of portfolio weight changes, date-range selection, historical crises, and unavailable Fama–French data. Run the suite from the project root using:

```bash
python -m pytest
```

The analytical modules contain input validation for invalid weights, insufficient historical data, missing prices, regression sample sizes, and unavailable stress-test coverage. The stated test count is based on the locally reported result; the test files were not included in the supplied review materials.

## Assumptions and limitations

- **Investable universe:** only 10 predefined US equities are selectable; users cannot enter arbitrary tickers in the current interface.
- **Long-only, fully invested weights:** no short selling or leverage; return aggregation uses fixed weights at each daily step.
- **No trading frictions:** commissions, slippage, taxes, and actual rebalancing execution are not modelled.
- **Risk-free conventions:** the dashboard Sharpe ratio and CAPM use a zero risk-free rate, whereas FF3 and FF5 use the French library's daily `RF` series.
- **Annualization:** 252 trading days per year; SPY is an ETF benchmark proxy, not the index itself.
- **History dependence:** regressions, crisis tests and regimes describe selected historical samples and depend on market-data coverage; they do not provide live trade signals or forecasts.
- **Data and estimation:** external sources can revise data; in-sample fit and statistical significance do not establish out-of-sample performance.

## Project purpose

FactorLens was developed as a hands-on quantitative finance and Python project to combine **financial theory, statistical inference, software engineering, and interactive communication**. The goal is to make portfolio behaviour explainable: not only *how a portfolio performed*, but *where its risks come from, which systematic factors its returns load on, and how it behaved under different historical conditions*.

## Disclaimer

FactorLens is an educational and research-oriented application. Historical estimates and model outputs are **not investment advice**, and past performance does not guarantee future results.
