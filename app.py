
"""FactorLens — interactive portfolio analytics and risk dashboard."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st
import altair as alt

from src.data_loader import download_prices
from src.performance import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_growth,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
)
from src.returns import calculate_daily_returns, calculate_portfolio_returns


st.set_page_config(page_title="FactorLens", page_icon="📊", layout="wide")

COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "The Goldman Sachs Group, Inc.",
    "XOM": "Exxon Mobil Corporation",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "CAT": "Caterpillar Inc.",
    "TSLA": "Tesla, Inc.",
}

SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "JPM": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "JNJ": "Health Care",
    "WMT": "Consumer Staples",
    "CAT": "Industrials",
    "TSLA": "Consumer Discretionary",
}


@st.cache_data(ttl=3600)


def load_stock_data(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Download and cache adjusted stock-price data for one hour."""
    return download_prices(tickers=list(tickers), start=start, end=end)


def equal_weights_percent(count: int) -> list[float]:
    """Return percentage weights that add up to exactly 100%."""
    base = round(100.0 / count, 2)
    return [base] * (count - 1) + [round(100.0 - base * (count - 1), 2)]


def reset_equal_weights(tickers: list[str]) -> None:
    """Reset the selected stocks to equal weights."""
    for ticker, weight in zip(tickers, equal_weights_percent(len(tickers))):
        st.session_state[f"weight_{ticker}"] = weight


def on_tickers_changed():
    """Reset weights when stocks are added or removed."""
    selected = st.session_state["selected_tickers"]

    if selected:
        reset_equal_weights(selected)


def on_weight_changed(changed_ticker):
    """Rebalance other stocks when one weight changes."""
    selected = st.session_state["selected_tickers"]
    others = [t for t in selected if t != changed_ticker]

    if not others:
        st.session_state[f"weight_{changed_ticker}"] = 100.0
        return

    changed_weight = st.session_state[f"weight_{changed_ticker}"]
    remaining = round(100.0 - changed_weight, 2)

    equal_share = round(remaining / len(others), 2)

    for ticker in others[:-1]:
        st.session_state[f"weight_{ticker}"] = equal_share

    st.session_state[f"weight_{others[-1]}"] = round(
        remaining - equal_share * (len(others) - 1),
        2,
    )



st.title(" FactorLens")
st.caption("Portfolio Analytics & Risk Management")

st.write(
    "Build a portfolio, explore historical performance, "
    "and understand investment risk."
)

st.divider()

builder_tab, performance_tab, risk_tab = st.tabs(
    [" Portfolio Builder", " Performance", " Risk Analysis"]
)


with builder_tab:
    st.header("Portfolio Builder")
    st.write("Select your companies, analysis period, and portfolio weights.")

    
    tickers = st.multiselect(
        "Select companies",
        options=list(COMPANIES),
        default=["AAPL", "MSFT", "JPM"],
        format_func=lambda ticker: f"{COMPANIES[ticker]} ({ticker})",
        key="selected_tickers",
        on_change=on_tickers_changed,
    )

    
    if not tickers:
        st.warning("Select at least one company to continue.")
        st.stop()


    st.subheader("Analysis Period")

    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "Start date",
            value=date(2025, 1, 1),
            max_value=date.today(),
        )

    with date_col2:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            max_value=date.today(),
        )

    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        st.stop()

    st.subheader("Portfolio Allocation")

    st.button(
        "Reset to Equal Weights",
        on_click=reset_equal_weights,
        args=(tickers,),
    )

    defaults = equal_weights_percent(len(tickers))
    weights_pct = []

    for i, ticker in enumerate(tickers):
        key = f"weight_{ticker}"

        if key not in st.session_state:
            st.session_state[key] = defaults[i]

        
        
        weight = st.number_input(
            f"{COMPANIES[ticker]} ({ticker}) — Weight (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key=key,
            on_change=on_weight_changed,
            args=(ticker,),
        )



        weights_pct.append(weight)

    total_weight = sum(weights_pct)

    st.write(f"Total portfolio weight: {total_weight:.2f}%")

    if abs(total_weight - 100.0) > 1e-6:
        st.warning("Portfolio weights must add up to 100%.")
        st.stop()

    weights = [weight / 100.0 for weight in weights_pct]


    st.subheader("Portfolio Composition")

    composition = pd.DataFrame({
        "Company": [COMPANIES[t] for t in tickers],
        "Ticker": tickers,
        "Sector": [SECTORS[t] for t in tickers],
        "Weight (%)": weights_pct,
    })

    st.dataframe(
        composition,
        hide_index=True,
        width="stretch",
    )

    st.subheader("Sector Allocation")

    sector_allocation = (
        composition.groupby("Sector")["Weight (%)"]
        .sum()
        .sort_values()
    )

    
    
    sector_chart_data = sector_allocation.reset_index()
    sector_chart_data.columns = ["Sector", "Weight"]

    sector_chart = (
        alt.Chart(sector_chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(
                "Weight:Q",
                title="Portfolio Weight (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "Sector:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelLimit=250),
            ),
            tooltip=[
                alt.Tooltip("Sector:N"),
                alt.Tooltip("Weight:Q", format=".2f"),
            ],
        )
        .properties(height=180)
    )

    st.altair_chart(
        sector_chart,
        width="stretch",
    )



    hhi = sum(weight ** 2 for weight in weights)
    effective_holdings = 1 / hhi

    st.subheader("Portfolio Concentration")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Concentration (HHI)", f"{hhi:.4f}")

    with c2:
        st.metric("Effective Holdings", f"{effective_holdings:.2f}")


    st.caption(
        "**Concentration (HHI):** Measures how concentrated your "
        "portfolio is across individual holdings. A higher value "
        "indicates greater concentration."
    )

    st.caption(
        "**Effective Holdings:** The equivalent number of equally "
        "weighted investments. A higher value indicates that your "
        "capital is spread more evenly across holdings."
    )


# Download and process market data.
prices = load_stock_data(
    tuple(tickers),
    start_date.isoformat(),
    end_date.isoformat(),
)

prices = prices.loc[:, tickers]
daily_returns = calculate_daily_returns(prices)
portfolio_returns = calculate_portfolio_returns(
    daily_returns, weights=weights
)


# Calculate portfolio performance metrics.
total_return = calculate_total_return(portfolio_returns)
annualized_return = calculate_annualized_return(portfolio_returns)
annualized_volatility = calculate_annualized_volatility(portfolio_returns)
sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
max_drawdown = calculate_max_drawdown(portfolio_returns)

portfolio_value = calculate_cumulative_growth(
    portfolio_returns, initial_investment=10_000.0
)

# Prepare the S&P 500 benchmark.
benchmark_prices = load_stock_data(
    ("SPY",),
    start_date.isoformat(),
    end_date.isoformat(),
)

benchmark_returns = calculate_daily_returns(
    benchmark_prices
)["SPY"]

common_dates = portfolio_returns.index.intersection(
    benchmark_returns.index
)

aligned_portfolio = portfolio_returns.loc[common_dates]
aligned_benchmark = benchmark_returns.loc[common_dates]

comparison = pd.DataFrame({
    "Our Portfolio": calculate_cumulative_growth(
        aligned_portfolio, initial_investment=10_000.0
    ),
    "S&P 500 (SPY)": calculate_cumulative_growth(
        aligned_benchmark, initial_investment=10_000.0
    ),
})

with performance_tab:
    st.header("Portfolio Performance")
    st.caption("Historical investment performance and benchmark comparison.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Return", f"{total_return:.2%}")

    with c2:
        st.metric("Annualized Return", f"{annualized_return:.2%}")

    with c3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")


    st.subheader("Portfolio vs. S&P 500")

    st.line_chart(
        comparison,
        x_label="Date",
        y_label="Investment Value ($)",
    )

    
    st.subheader("Portfolio Growth Over Time")

    st.line_chart(
        portfolio_value,
        x_label="Date",
        y_label="Portfolio Value ($)",
        height=350,
    )



with risk_tab:
    st.header("Risk Analysis")
    st.caption("Understand portfolio volatility, losses, and concentration.")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Annualized Volatility",
            f"{annualized_volatility:.2%}",
        )

    with r2:
        st.metric(
            "Maximum Drawdown",
            f"{max_drawdown:.2%}",
        )

    with r3:
        st.metric(
            "Effective Holdings",
            f"{effective_holdings:.2f}",
        )

    st.subheader("Portfolio Drawdown")

    drawdown = calculate_drawdown(portfolio_returns)

    st.area_chart(
        drawdown * 100,
        x_label="Date",
        y_label="Drawdown (%)",
    )

    st.info(
        "Concentration measures how portfolio weights are distributed. "
        "It does not account for correlations between assets."
    )

