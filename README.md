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

The user chooses holdings from a predefined universe (AAPL, MSFT, NVDA, JPM, GS, XOM, JNJ, WMT, CAT, TSLA) and assigns **nonnegative weights summing to 100%**. Users can manually adjust individual stock weights, with the total portfolio allocation required to equal 100%. An equal-weight allocation can be restored with a single click.

For a given trading day, portfolio return is computed as the weighted sum of individual stock returns:



**Portfolio return:**


FactorLens calculates how a portfolio performs by combining the daily returns of individual stocks according to their portfolio weights.

For example, if you invest **60% in Apple and 40% in Microsoft**, and their stock prices increase by 2% and 1% respectively, your portfolio gains:

**(60% × 2%) + (40% × 1%) = 1.6%**

This calculation follows the formula:

$$
r_{p,t} = \sum_{i=1}^{N} w_i r_{i,t}
$$

Where:
- **rₚ,ₜ** = Total portfolio return on a given day
- **wᵢ** = Percentage of the portfolio invested in stock i
- **rᵢ,ₜ** = Daily return of stock i
- **N** = Number of stocks in the portfolio

FactorLens uses historical stock prices from Yahoo Finance (via `yfinance`), adjusted for stock splits and dividends.

The dashboard then shows how an initial investment of **$10,000** would have grown over the selected period and compares that performance with the **S&P 500**, a widely used benchmark representing large U.S. companies (tracked through the SPY ETF).


### Key Performance Measures

- **Total Return:** The overall percentage gained or lost by the portfolio during the selected period. For example, if a $10,000 investment grows to $11,500, the total return is 15%.

- **Annualized Return:** The portfolio's average yearly growth rate, accounting for compounding. This makes it easier to compare investments held for different lengths of time.

- **Annualized Volatility:** Measures how much the portfolio's returns fluctuate over time. Higher volatility means larger price movements and generally greater uncertainty. FactorLens calculates this using daily returns and assumes 252 trading days per year.

- **Sharpe Ratio:** Measures how much return the portfolio generates relative to the risk taken. A higher Sharpe ratio generally indicates better historical risk-adjusted performance. FactorLens assumes a 0% risk-free rate.

- **Maximum Drawdown:** The largest percentage decline in portfolio value from a previous peak to a subsequent low. For example, if a portfolio falls from $12,000 to $9,000 before recovering, its maximum drawdown is 25%.

**Important assumption:** FactorLens calculates portfolio returns using the same target stock weights every day. This represents a portfolio that is continuously rebalanced to maintain its allocation. The calculations do not include transaction costs, taxes, or the practical details of buying and selling stocks.



## 2. Diversification and Risk Contributions

Diversification means spreading investments across different stocks and sectors to reduce risk. However, owning multiple stocks does not automatically make a portfolio well diversified. If those stocks behave similarly, they may still expose the portfolio to the same risks.

FactorLens helps users understand diversification through several measures:

### Correlation

Correlation measures how closely two stocks' returns move together. It ranges from -1 to +1:

- **+1:** The stocks move perfectly together in percentage-return terms, offering limited diversification benefits.
- **0:** There is no linear relationship between their returns.
- **-1:** The stocks move in exactly opposite directions in percentage-return terms, potentially providing strong diversification benefits.

For example, two technology stocks may have high positive correlation because they respond similarly to market conditions.

FactorLens displays a **correlation heatmap** showing the relationships between selected stocks and identifies the pair with the strongest positive correlation.

### Portfolio Concentration (HHI)

The Herfindahl-Hirschman Index (HHI) measures how concentrated a portfolio is in a small number of stocks.

$$
HHI = \sum_{i=1}^{N} w_i^2
$$

Here, **w** represents each stock's share of the portfolio, expressed as a decimal.

- **Higher HHI:** More money is concentrated in fewer stocks.
- **Lower HHI:** Investments are distributed more evenly.

For example, a portfolio invested entirely in one stock has an HHI of 1, while a portfolio split equally between four stocks has an HHI of 0.25.

### Effective Number of Holdings

This measure translates portfolio concentration into an easier-to-understand number:

$$
N_{\text{eff}} = \frac{1}{HHI}
$$

For example:
- A portfolio equally divided among 5 stocks has **5 effective holdings**.
- A portfolio with most of its money concentrated in one stock will have fewer effective holdings, even if it owns many different stocks.

**Important:** Effective holdings measure how evenly money is distributed, not how independently the stocks behave.

### Risk Contributions

A stock's portfolio weight tells us **how much money is invested in it**, while its risk contribution tells us **how much it contributes to the portfolio's overall volatility**.

These two percentages are not necessarily equal.

For example:
- Stock A represents **20% of the portfolio's investment**.
- However, because it is highly volatile and moves closely with other holdings, it might contribute **35% of the portfolio's total risk**.

FactorLens calculates risk contributions using historical stock returns, their volatility, and how they move relative to each other.

The dashboard then compares each stock's **portfolio weight against its risk contribution**, helping users identify holdings that contribute disproportionately to overall portfolio risk.

Risk contributions add up to 100% and can occasionally be negative when a stock helps offset fluctuations in other holdings.

### Why This Matters

A portfolio may appear diversified because it contains several different companies, but its risk could still be dominated by only a few stocks.

FactorLens helps users identify these hidden concentrations and better understand the relationship between **portfolio allocation, diversification, and risk**.



## 3. Asset-Pricing Models: CAPM, FF3 and FF5

FactorLens uses asset-pricing models to understand **what drives a portfolio's historical returns**.

For example, is the portfolio performing well simply because the overall stock market is rising? Or is its performance also related to the types of companies it invests in, such as smaller businesses or companies with strong profitability?

The **Factor Analysis** section contains five sub-tabs: CAPM, Fama–French 3, Fama–French 5, Model Comparison, and Statistical Tests.

### Capital Asset Pricing Model (CAPM)

CAPM examines how sensitive a portfolio is to movements in the overall stock market.

FactorLens uses the **SPY ETF**, which tracks the S&P 500, as its market benchmark.

The model is expressed as:

$$
r_{p,t} - r_f = \alpha + \beta_m(r_{SPY,t} - r_f) + \varepsilon_t
$$

The three main results are:

- **Beta (β):** Measures how strongly the portfolio moves with the market. For example, a beta of 1.2 suggests that when SPY rises by 1%, the portfolio tends to rise by approximately 1.2%, on average, based on the historical relationship. The same applies to market declines.

- **Alpha (α):** Measures the part of the portfolio's average daily return not explained by market movements in the regression. A positive alpha does not automatically mean the portfolio consistently outperforms the market.

- **R-squared (R²):** Measures how much of the portfolio's historical return variation is explained by market movements. For example, an R² of 80% means that approximately 80% of the variation in daily portfolio returns is explained by the model.

**In FactorLens:** The CAPM tab displays alpha, beta, R², and a scatter plot showing the historical relationship between portfolio and SPY returns, together with the fitted regression line.

**Note:** The CAPM calculation assumes a 0% risk-free rate.

### Fama–French Three-Factor Model (FF3)

While CAPM focuses only on the overall market, the Fama–French Three-Factor Model considers two additional factors that may help explain portfolio returns: **company size and valuation**.

The model is:

$$
r_{p,t} - RF_t = \alpha + \beta_m(Mkt-RF)_t + \beta_s SMB_t + \beta_h HML_t + \varepsilon_t
$$

The three factors are:

- **Market (Mkt-RF):** Measures the stock market's return above the risk-free rate. It captures how sensitive the portfolio is to overall market movements.

- **Size (SMB — Small Minus Big):** Captures the difference in returns between small and large companies. A positive exposure suggests the portfolio behaves more like smaller companies, while a negative exposure suggests a tendency toward larger companies.

- **Value (HML — High Minus Low):** Captures the difference in returns between value stocks (companies with high book-to-market ratios) and growth stocks (companies with low book-to-market ratios). A positive exposure indicates a value tilt, while a negative exposure suggests a growth tilt.

**Example:** A portfolio containing mostly large technology companies might have negative SMB and HML coefficients, indicating that its returns historically resemble those of large-cap and growth-oriented stocks.

**In FactorLens:** The FF3 tab displays estimated alpha, R², and a bar chart of the portfolio's exposure to market, size, and value factors.

### Fama–French Five-Factor Model (FF5)

The Fama–French Five-Factor Model extends FF3 by adding two more factors: **profitability and investment behavior**.

The model is:

$$
\begin{aligned}
r_{p,t} - RF_t ={}& \alpha + \beta_m(Mkt-RF)_t + \beta_s SMB_t \\
&+ \beta_h HML_t + \beta_r RMW_t + \beta_c CMA_t + \varepsilon_t
\end{aligned}
$$

In addition to the three factors above, FF5 includes:

- **Profitability (RMW — Robust Minus Weak):** Measures the difference in returns between companies with strong and weak operating profitability. A positive exposure suggests that the portfolio's returns behave more like those of highly profitable companies.

- **Investment (CMA — Conservative Minus Aggressive):** Measures the difference in returns between companies that invest relatively conservatively and those that expand their assets more aggressively. A positive exposure suggests greater sensitivity to companies with conservative investment behavior.

**Example:** A portfolio with positive RMW and CMA coefficients may historically behave more like companies with strong profitability and relatively conservative investment strategies.

**In FactorLens:** The FF5 tab displays estimated alpha, R², and a bar chart showing exposure to all five factors.

### Understanding Factor Exposures

Factor exposures, also called **factor loadings**, describe how strongly portfolio returns respond to different sources of market risk.

- **Positive loading:** Portfolio returns tend to move in the same direction as the factor, holding the other factors constant.
- **Negative loading:** Portfolio returns tend to move in the opposite direction to the factor, holding the other factors constant.
- **Loading close to zero:** The portfolio has relatively little estimated sensitivity to that factor.

These estimates describe historical statistical relationships. They do not directly measure the portfolio's actual stock holdings or guarantee future performance.

### Data Sources

FactorLens obtains daily Fama–French factor returns from the **Kenneth R. French Data Library**.

The data is converted from percentages into decimal values to match portfolio returns. FF3 and FF5 regressions also subtract the dataset's daily risk-free rate from portfolio returns.

Both models use historical daily observations to estimate the portfolio's factor exposures.

#
## 4. Model Comparison and Statistical Inference

FactorLens not only estimates the Fama–French 3-factor (FF3) and 5-factor (FF5) models, but also compares how well they explain portfolio returns and evaluates whether the estimated factor exposures are statistically significant.

### Model Comparison: FF3 vs. FF5

The FF3 model uses three factors (market, size, and value), while FF5 adds two more (profitability and investment).

The **Model Comparison** tab helps users understand whether adding these two factors improves how well the model explains historical portfolio returns.

FactorLens compares the models using:

- **R-squared (R²):** Measures how much of the variation in portfolio returns a model explains. For example, an R² of 90% means the model explains approximately 90% of the historical variation in daily returns.

- **Adjusted R-squared:** Similar to R², but accounts for the number of factors included in the model. Adding more factors can increase R² even if they contribute very little useful information. Adjusted R² helps determine whether the additional factors genuinely improve the model's fit.

**Example:** Suppose FF3 has an adjusted R² of 85% and FF5 has an adjusted R² of 89%. This suggests FF5 provides a better explanation of the portfolio's historical returns, even after accounting for its two additional factors.

Both models are estimated using the **same trading days**, ensuring that their results can be compared fairly.

**In FactorLens:** The Model Comparison tab displays R², adjusted R², and daily alpha for both models, and identifies which provides the better adjusted fit.

**Important:** A higher adjusted R² indicates a better fit to historical data, but does not guarantee more accurate predictions of future returns.

### Statistical Significance Testing

Estimating a factor exposure does not necessarily mean that the relationship is statistically reliable.

For example, a portfolio might have a positive estimated exposure to the profitability factor (RMW), but the relationship could be too uncertain to distinguish from zero.

FactorLens uses several statistical measures to evaluate the estimates from the FF5 model:

- **Coefficient:** The estimated strength and direction of the relationship between portfolio returns and a factor, holding the other factors constant.

- **Standard Error:** Measures the uncertainty surrounding an estimated coefficient. Smaller standard errors generally indicate more precise estimates.

- **t-statistic:** Compares the estimated coefficient with its standard error to help assess whether it is different from zero.

- **p-value:** Helps evaluate the evidence against the hypothesis that a factor has no relationship with portfolio returns, after accounting for the other factors.

- **95% Confidence Interval:** Provides a range of plausible values for the true factor coefficient, based on the model and its assumptions.

### Understanding p-values and Confidence Intervals

FactorLens uses a **5% significance level**, meaning:

- **p-value below 0.05:** The factor exposure is considered statistically significant, providing evidence that its coefficient differs from zero.
- **p-value of 0.05 or higher:** There is insufficient statistical evidence to conclude that the coefficient differs from zero.

For example, if the estimated SMB coefficient is 0.30 with a p-value of 0.01, there is statistical evidence of a positive size-factor exposure.

Confidence intervals provide another way to understand this uncertainty.

For example, if a factor has an estimated coefficient of 0.30 and a 95% confidence interval of [0.10, 0.50], the interval excludes zero, supporting statistical significance at the 5% level.

**In FactorLens:** The Statistical Tests tab presents a table of estimated FF5 coefficients, standard errors, t-statistics, p-values, confidence intervals, and significance indicators. It also displays a chart visualizing the confidence intervals for all five factors.

### HAC Standard Errors

Financial returns often exhibit changing volatility over time, and regression errors may be correlated across trading days.

These patterns can make conventional statistical standard errors unreliable.

To address this, FactorLens uses **Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors**, also known as Newey–West-type standard errors.

This method adjusts statistical inference to account for:

- **Heteroskedasticity:** Changes in the variability of regression errors over time.
- **Autocorrelation:** Relationships between regression errors across different trading days.

FactorLens implements HAC estimation using Python's `statsmodels` library with **5 lags**, producing more robust standard errors, hypothesis tests, and confidence intervals under the relevant assumptions.

### Why This Matters

A factor model may appear to explain portfolio returns well, but that does not automatically mean every estimated factor exposure is statistically meaningful.

By combining **model comparison and statistical significance testing**, FactorLens helps users distinguish between models that fit historical data well and factor exposures supported by stronger statistical evidence.

Statistical significance does not necessarily imply economic importance, causation, or future predictive performance.



## 5. Historical Stress Testing

Historical stress testing helps investors understand **how a portfolio would have performed during past financial crises**.

FactorLens takes the stocks and portfolio weights currently selected by the user and applies them to historical market data from three major downturns.

This helps answer questions such as: *How would my portfolio have performed during the COVID-19 crash? Would it have lost more or less value than the overall market? Which stocks would have contributed most to the losses?*

### Available Historical Scenarios

| Historical Event | Period Analyzed |
|---|---|
| **2008 Global Financial Crisis** | September 1, 2008 – March 31, 2009 |
| **COVID-19 Market Crash** | February 19, 2020 – March 23, 2020 |
| **2022 Market Downturn** | January 1, 2022 – December 31, 2022 |

### What FactorLens Measures

For each selected crisis, FactorLens calculates:

- **Portfolio Return:** The total percentage gained or lost by the portfolio during the crisis.

- **Maximum Drawdown:** The largest decline in portfolio value from a previous peak during the selected period.

- **Annualized Volatility:** How much the portfolio's daily returns fluctuated during the crisis, expressed on a yearly basis.

- **S&P 500 Comparison:** Compares the portfolio's performance with SPY, an ETF tracking the S&P 500.

- **Portfolio Outperformance:** The difference between the portfolio's return and SPY's return, measured in percentage points. For example, if the portfolio loses 15% while SPY loses 20%, the portfolio outperforms the benchmark by 5 percentage points.

- **Individual Stock Contributions:** Shows how much each holding contributed to the portfolio's overall gain or loss during the crisis.

### Understanding Stock Contributions

Not every stock affects portfolio performance equally.

For example, suppose a portfolio contains two stocks with equal weights during a market downturn:

- **Stock A** loses 20%.
- **Stock B** loses 10%.

Over a single day, Stock A contributes -10 percentage points to the portfolio's return, while Stock B contributes -5 percentage points.

The total portfolio return is therefore **-15%**.

FactorLens uses a more detailed calculation across the entire crisis period, accounting for how daily portfolio returns compound over time.

The application also checks that the individual stock contributions add up to the total reported portfolio return.

### How FactorLens Handles Missing Historical Data

Not every company has stock-price data going back to earlier financial crises.

For example, some companies included in today's stock market were not publicly traded during the 2008 financial crisis.

FactorLens checks whether sufficient historical data is available for every selected holding before calculating results.

If the required data is unavailable, the application displays a warning rather than presenting incomplete or potentially misleading calculations.

### Important Limitations

Historical stress testing uses real market data from past events, but it **does not predict how a portfolio will perform during future crises**.

The analysis also assumes that the selected portfolio weights remain constant throughout each historical scenario and does not include transaction costs or taxes.

### Why This Matters

Historical stress testing helps investors explore potential weaknesses in their portfolios by examining how different stocks behaved during periods of significant market uncertainty.

By comparing portfolio losses with the overall market and identifying the stocks responsible for those losses, FactorLens provides a clearer understanding of **historical downside risk and portfolio resilience**.


## 6. Market Regime Detection

Financial markets experience both calm periods and periods of greater uncertainty. FactorLens classifies these conditions as **High Volatility** or **Low Volatility**, helping users understand how their portfolios behave in different market environments.

### How It Works

1. **Recent Volatility:** Calculates how much SPY (the S&P 500 ETF) has fluctuated over the previous 21 trading days, expressed as annualized volatility.

2. **Historical Benchmark:** Calculates the median volatility from the previous 126 observations (approximately six months), excluding the current observation.

3. **Market Classification:** If recent volatility is above the historical median, the market is classified as **High Volatility**. Otherwise, it is classified as **Low Volatility**.

**Example:** If recent market volatility is 25% and the historical median is 18%, FactorLens classifies the market as High Volatility.

### What FactorLens Displays

- A chart comparing recent volatility with its historical benchmark.
- The latest available market regime classification.
- Average daily portfolio returns and annualized volatility in each regime.
- Insights showing which regime had higher average daily returns.

**Important:** These classifications describe historical market conditions, not future predictions. High volatility does not necessarily mean stock prices are falling.

---

## Data and Implementation

FactorLens is built in **Python**, combining financial data processing, quantitative models, and interactive visualization.

| Component | Implementation |
|---|---|
| **Market Data** | `yfinance` for historical adjusted stock prices and SPY benchmark data |
| **Factor Data** | Kenneth R. French Data Library via `pandas-datareader` |
| **Financial Calculations** | Pandas and NumPy |
| **Regression Analysis** | NumPy for CAPM, FF3, and FF5; `statsmodels` for statistical inference |
| **Dashboard** | Streamlit and Plotly |
| **Testing** | `pytest` (102 automated tests passed during local development) |

**Data reliability:** FactorLens validates portfolio weights and historical data, checks for missing observations, and handles unavailable Fama–French data. It also uses Streamlit caching to reduce repeated downloads.



## Repository Structure

FactorLens follows a modular Python project structure, separating the interactive dashboard from the financial calculations, data processing, and automated tests.

```text
FactorLens/
├── app.py                  # Main Streamlit dashboard
├── app_backup.py           # Backup version of the application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── LICENSE                 # Project license
├── .gitignore              # Files excluded from Git
│
├── src/                    # Financial analysis modules
│   ├── __init__.py
│   ├── data_loader.py      # Historical stock price downloads
│   ├── validation.py       # Input and data validation
│   ├── returns.py          # Stock and portfolio returns
│   ├── performance.py      # Performance and risk-adjusted metrics
│   ├── risk.py             # Portfolio risk contributions
│   ├── factor_data.py      # Fama-French data retrieval
│   ├── factors.py          # CAPM, FF3, and FF5 regressions
│   ├── diagnostics.py      # Statistical tests and HAC inference
│   ├── stress.py           # Historical crisis stress testing
│   ├── regimes.py          # Market volatility classification
│   ├── visualization.py    # Visualization functions for demo
│   └── demo_portfolio.py   # Standalone portfolio analysis example
│
└── tests/                  # Automated unit tests
    ├── __init__.py
    ├── test_diagnostics.py
    ├── test_factor_data.py
    ├── test_factors.py
    ├── test_performance.py
    ├── test_regimes.py
    ├── test_returns.py
    ├── test_risk.py
    ├── test_stress.py
    └── test_validation.py
```

**Project organization:**

- **`app.py`** connects the financial calculations to the interactive Streamlit dashboard.
- **`src/`** contains reusable Python functions for financial analysis, statistical modelling, and data processing.
- **`tests/`** contains automated tests used to verify the correctness of the calculations.

This separation makes the project easier to maintain, test, and extend.



## Run the Dashboard Locally

To run FactorLens on your own computer, make sure Python and Git are installed.

### 1. Clone the Repository

Download the project from GitHub and navigate to the project folder.

```bash
git clone https://github.com/YOUR-USERNAME/FactorLens.git
cd FactorLens
```

### 2. Set Up the Python Environment

Create and activate a virtual environment (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

### 4. Launch FactorLens

Start the interactive Streamlit dashboard:

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

**http://localhost:8501**

An internet connection is required to download historical stock prices and Fama–French factor data.



## Testing and Quality Controls

FactorLens uses automated testing and input validation to ensure reliable financial calculations.

**102 automated tests passed** during local development using the `pytest` framework. The tests cover key components, including:

- Portfolio returns and performance metrics
- Risk contributions and diversification calculations
- CAPM and Fama–French factor models
- Statistical diagnostics and confidence intervals
- Historical stress testing and market regime classification
- Data validation and handling of invalid inputs

The application was also manually tested by adjusting portfolio weights, changing analysis periods, selecting historical crisis scenarios, and checking how the dashboard handles unavailable Fama–French data.

### Running the Tests

To run the automated tests, open a terminal in the project's root directory and execute:

```bash
python -m pytest
```

FactorLens also includes safeguards against invalid portfolio weights, missing market data, insufficient regression observations, and incomplete historical stress-test data.



## Assumptions and Limitations

FactorLens is designed for educational and analytical purposes. Its calculations rely on several assumptions:

- **Stock Selection:** Users can build portfolios using 10 predefined U.S. stocks. The current version does not support entering additional stock tickers.

- **Portfolio Weights:** All portfolio weights must be positive or zero and add up to 100%. Short selling (betting against a stock) and borrowing money to invest are not supported. The calculations assume that portfolio weights remain constant each day.

- **Trading Costs:** The analysis does not account for transaction fees, taxes, or other costs associated with buying and selling stocks.

- **Risk-Free Rate:** The Sharpe ratio and CAPM assume a 0% risk-free rate, while the Fama–French models use historical daily risk-free rates from the Kenneth R. French Data Library.

- **Annualization and Benchmark:** Annualized calculations assume 252 trading days per year. The SPY ETF, which tracks the S&P 500, serves as the market benchmark.

- **Historical Analysis:** Factor models, stress tests, and market regimes are based on historical data. Their results describe past relationships and do not predict future market performance.

- **Data Availability:** Historical prices and factor data come from external sources and may be incomplete, delayed, or revised. Statistical relationships found in historical data may not remain valid in the future.

**Disclaimer:** FactorLens is a personal educational project and does not provide financial advice or investment recommendations.



## Project Purpose

FactorLens was developed as a personal Python and quantitative finance project to apply financial theory and statistical methods to real market data.

The goal is to go beyond simply measuring portfolio returns and help users understand **what drives portfolio performance, where investment risks come from, and how portfolios behave under different market conditions**.

Through this project, I aimed to combine **financial analysis, econometrics, Python programming, and interactive data visualization** into an accessible portfolio intelligence dashboard.


## Disclaimer

FactorLens is an educational and research-oriented application. Historical estimates and model outputs are **not investment advice**, and past performance does not guarantee future results.
