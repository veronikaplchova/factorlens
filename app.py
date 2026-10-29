
"""FactorLens | portfolio intelligence dashboard."""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
from src.risk import calculate_risk_contributions

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
PURPLE = "#9B87F5"
BLUE = "#4CB8F4"
RED = "#FB7185"
GREEN = "#2DD4BF"
BG = "#0B1220"
PANEL = "#151F33"


@st.cache_data(ttl=3600)
def load_stock_data(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Reuse downloads when the ticker list and dates have not changed."""
    return download_prices(tickers=list(tickers), start=start, end=end)


def equal_weights_percent(count: int) -> list[float]:
    if count < 1:
        return []
    base = round(100.0 / count, 2)
    return [base] * (count - 1) + [round(100 - base * (count - 1), 2)]


def reset_equal_weights(tickers: list[str]) -> None:
    for ticker, weight in zip(tickers, equal_weights_percent(len(tickers))):
        st.session_state[f"weight_{ticker}"] = weight


def on_tickers_changed() -> None:
    selected = st.session_state["selected_tickers"]
    if selected:
        reset_equal_weights(selected)


def on_weight_changed(changed_ticker: str) -> None:
    selected = st.session_state["selected_tickers"]
    others = [ticker for ticker in selected if ticker != changed_ticker]
    if not others:
        st.session_state[f"weight_{changed_ticker}"] = 100.0
        return
    changed_weight = st.session_state[f"weight_{changed_ticker}"]
    remaining = round(100.0 - changed_weight, 2)
    share = round(remaining / len(others), 2)
    for ticker in others[:-1]:
        st.session_state[f"weight_{ticker}"] = share
    st.session_state[f"weight_{others[-1]}"] = round(
        remaining - share * (len(others) - 1), 2
    )



def style_figure(fig: go.Figure, height: int = 350) -> go.Figure:
    """Apply the FactorLens dark chart theme."""
    fig.update_layout(
        height=height,
        paper_bgcolor="#151F33",
        plot_bgcolor="#151F33",
        font=dict(
    family="Arial, sans-serif",
    size=13,
    color="#E8EEFA",
),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="#29354B",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#29354B",
            zeroline=False,
        ),
        
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
            font=dict(
                color="#E8EEFA",
                size=13,
            ),
            bgcolor="rgba(0,0,0,0)",
        ),

    )
    return fig



# FactorLens visual theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1220;
        color: #E8EEFA;
    }

    [data-testid="stSidebar"] {
        background-color: #101A2D;
        border-right: 1px solid #26344D;
    }

    [data-testid="stHeader"] {
        background-color: #0B1220;
    }

    h1, h2, h3 {
        color: #F5F7FF !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stMetric"] {
        background: #151F33;
        border: 1px solid #26344D;
        border-radius: 16px;
        padding: 22px;
    }

    [data-testid="stMetricLabel"] {
        color: #9FB0CE;
    }

    [data-testid="stMetricValue"] {
        color: #F5F7FF;
    }

    div.stButton > button {
        border-radius: 12px;
        background-color: #25365A;
        color: #E8EEFA;
        border: 1px solid #394C74;
    }

    div.stButton > button:hover {
        background-color: #394C74;
        color: white;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #26344D;
        border-radius: 14px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .factorlens-subtitle {
        color: #9FB0CE;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(" FactorLens")
st.markdown(
    '<p class="factorlens-subtitle">'
    'PORTFOLIO INTELLIGENCE · Performance · Risk · Insights'
    '</p>',
    unsafe_allow_html=True,
)

st.divider()


# Portfolio configuration sidebar
with st.sidebar:
    st.markdown("##  Portfolio Settings")
    st.caption("Customize your investment portfolio.")

    tickers = st.multiselect(
        "Select companies",
        options=list(COMPANIES),
        default=["AAPL", "MSFT", "JPM"],
        format_func=lambda t: f"{COMPANIES[t]} ({t})",
        key="selected_tickers",
        on_change=on_tickers_changed,
    )

    if not tickers:
        st.warning("Select at least one company.")
        st.stop()

    st.divider()
    st.markdown("### Analysis Period")

    start_date = st.date_input(
        "Start date",
        value=date(2025, 1, 1),
        max_value=date.today(),
    )

    end_date = st.date_input(
        "End date",
        value=date.today(),
        max_value=date.today(),
    )

    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        st.stop()

    st.divider()
    st.markdown("### Portfolio Allocation")

    st.button(
        "Reset to Equal Weights",
        on_click=reset_equal_weights,
        args=(tickers,),
        use_container_width=True,
    )

    defaults = equal_weights_percent(len(tickers))
    weights_pct = []

    for i, ticker in enumerate(tickers):
        key = f"weight_{ticker}"

        if key not in st.session_state:
            st.session_state[key] = defaults[i]

        weight = st.number_input(
            f"{COMPANIES[ticker]} ({ticker})",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key=key,
            on_change=on_weight_changed,
            args=(ticker,),
        )

        weights_pct.append(weight)

    total_weight = sum(weights_pct)
    st.caption(f"Total allocation: {total_weight:.2f}%")

    if abs(total_weight - 100.0) > 1e-6:
        st.warning("Portfolio weights must total 100%.")
        st.stop()

    weights = [w / 100.0 for w in weights_pct]

    st.divider()
    st.caption("FactorLens · Portfolio Intelligence")

# Portfolio composition dataset
composition = pd.DataFrame({
    "Company": [COMPANIES[t] for t in tickers],
    "Ticker": tickers,
    "Sector": [SECTORS[t] for t in tickers],
    "Weight (%)": weights_pct,
})

sector_allocation = (
    composition.groupby("Sector")["Weight (%)"]
    .sum()
    .sort_values(ascending=False)
)

hhi = sum(w ** 2 for w in weights)
effective_holdings = 1.0 / hhi


# Download historical prices and calculate portfolio returns.
prices = load_stock_data(
    tuple(tickers),
    start_date.isoformat(),
    (end_date + timedelta(days=1)).isoformat(),
)

prices = prices.loc[:, tickers]
daily_returns = calculate_daily_returns(prices)
portfolio_returns = calculate_portfolio_returns(
    daily_returns, weights=weights
)
risk_contributions = calculate_risk_contributions(
    daily_returns, weights=weights
)

# Calculate portfolio performance.
total_return = calculate_total_return(portfolio_returns)
annualized_return = calculate_annualized_return(portfolio_returns)
annualized_volatility = calculate_annualized_volatility(portfolio_returns)
sharpe_ratio = calculate_sharpe_ratio(
    portfolio_returns, risk_free_rate=0.0
)
max_drawdown = calculate_max_drawdown(portfolio_returns)

portfolio_value = calculate_cumulative_growth(
    portfolio_returns, initial_investment=10_000.0
)
drawdown = calculate_drawdown(portfolio_returns)

# Download the S&P 500 benchmark.
benchmark_prices = load_stock_data(
    ("SPY",),
    start_date.isoformat(),
    (end_date + timedelta(days=1)).isoformat(),
)
benchmark_returns = calculate_daily_returns(benchmark_prices)["SPY"]

common_dates = portfolio_returns.index.intersection(
    benchmark_returns.index
)
aligned_portfolio = portfolio_returns.loc[common_dates]
aligned_benchmark = benchmark_returns.loc[common_dates]

comparison = pd.DataFrame({
    "Portfolio": calculate_cumulative_growth(
        aligned_portfolio, initial_investment=10_000.0
    ),
    "S&P 500": calculate_cumulative_growth(
        aligned_benchmark, initial_investment=10_000.0
    ),
})

# Build the main dashboard navigation.
overview_tab, builder_tab, risk_tab = st.tabs(
    [" Overview", " Portfolio Builder", " Risk Analytics"]
)

with overview_tab:
    st.subheader("Your Portfolio at a Glance")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Return", f"{total_return:+.2%}")
    with c2:
        st.metric("Annualized Return", f"{annualized_return:.2%}")
    with c3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Annualized Volatility", f"{annualized_volatility:.2%}")
    with c5:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
    with c6:
        st.metric("Effective Holdings", f"{effective_holdings:.2f}")

    st.subheader("Portfolio vs. S&P 500")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison.index, y=comparison["Portfolio"],
        name="Your Portfolio", line=dict(color=PURPLE, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=comparison.index, y=comparison["S&P 500"],
        name="S&P 500", line=dict(color=BLUE, width=2),
    ))
    fig.update_yaxes(title_text="Investment Value ($)")
    st.plotly_chart(style_figure(fig, 430), width="stretch")


with builder_tab:
    st.subheader("Portfolio Composition")
    st.caption("Understand how your investments are allocated.")

    left, right = st.columns(2)

    with left:
        
        fig = px.pie(
            composition,
            names="Ticker",
            values="Weight (%)",
            hole=0.68,
            color_discrete_sequence=[
                PURPLE, BLUE, GREEN, "#F59E0B",
                RED, "#C4B5FD", "#38BDF8",
                "#FBBF24", "#34D399", "#F472B6",
            ],
        )

        fig.update_traces(
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=13, color="#FFFFFF"),
            hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
            sort=False,
        )

        fig.update_layout(
            showlegend=False,
            annotations=[
                dict(
                    text=f"<b>{len(tickers)}</b><br>Holdings",
                    x=0.5,
                    y=0.5,
                    font=dict(size=20, color="#F5F7FF"),
                    showarrow=False,
                )
            ],
        )

        st.plotly_chart(
            style_figure(fig, 340),
            width="stretch",
        )

        fig.update_traces(
            textinfo="percent",
            hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
        )
        

    with right:
        st.markdown("#### Selected Holdings")
        st.dataframe(
            composition,
            hide_index=True,
            width="stretch",
        )

    st.subheader("Sector Allocation")

    fig = px.bar(
        sector_allocation.reset_index(),
        x="Weight (%)",
        y="Sector",
        orientation="h",
        color_discrete_sequence=[PURPLE],
    )
    fig.update_xaxes(range=[0, 100])
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(style_figure(fig, 300), width="stretch")


with risk_tab:
    st.subheader("Portfolio Risk Analytics")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Volatility", f"{annualized_volatility:.2%}")
    with r2:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    with r3:
        st.metric("Concentration (HHI)", f"{hhi:.4f}")

    st.subheader("Portfolio Weight vs. Risk Contribution")

    risk_comparison = pd.DataFrame({
        "Ticker": tickers,
        "Portfolio Weight": weights_pct,
        "Risk Contribution": risk_contributions.reindex(tickers).values,
    })

    risk_long = risk_comparison.melt(
        id_vars="Ticker",
        var_name="Metric",
        value_name="Percentage",
    )

    fig = px.bar(
        risk_long,
        x="Ticker",
        y="Percentage",
        color="Metric",
        barmode="group",
        color_discrete_map={
            "Portfolio Weight": BLUE,
            "Risk Contribution": PURPLE,
        },
    )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Percentage (%)",
    )

    st.plotly_chart(
        style_figure(fig, 370),
        width="stretch",
    )

    st.caption(
        "Portfolio weight measures invested capital. Risk contribution "
        "measures each holding's share of total portfolio volatility. "
        "Risk contributions can be negative when a holding reduces risk."
    )

    st.subheader("Portfolio Drawdown")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color=RED, width=2),
        fillcolor="rgba(251,113,133,0.18)",
    ))
    fig.update_yaxes(title_text="Drawdown (%)")
    st.plotly_chart(style_figure(fig, 350), width="stretch")

    st.subheader("Asset Correlation Matrix")
    if len(tickers) > 1:
        correlation = daily_returns.corr()

        # Find the pair of stocks with the highest correlation.
        correlation_pairs = []

        for i, ticker_a in enumerate(tickers):
            for ticker_b in tickers[i + 1:]:
                correlation_pairs.append(
                    (
                        ticker_a,
                        ticker_b,
                        correlation.loc[ticker_a, ticker_b],
                    )
                )

        valid_pairs = [
            pair for pair in correlation_pairs
            if pd.notna(pair[2])
        ]

        if valid_pairs:
            strongest_pair = max(
                valid_pairs,
                key=lambda pair: pair[2],
            )

            stock_a, stock_b, correlation_value = strongest_pair


            st.success(
                f"**Strongest positive correlation:** "
                f"{COMPANIES[stock_a]} ({stock_a}) and "
                f"{COMPANIES[stock_b]} ({stock_b}) "
                f"have a historical return correlation of "
                f"**{correlation_value:.2f}**."
            )

            if correlation_value >= 0.7:
                st.caption(
                    "These stocks have a strong positive historical "
                    "correlation, which may limit diversification benefits."
                )
            elif correlation_value >= 0.3:
                st.caption(
                    "These stocks have a moderate positive historical "
                    "correlation and may provide some diversification."
                )
            else:
                st.caption(
                    "This pair has relatively low historical correlation, "
                    "which may offer diversification benefits."
                )

        fig = px.imshow(
            correlation,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        st.plotly_chart(style_figure(fig, 370), width="stretch")
    else:
        st.info("Select at least two stocks to view correlations.")

    st.subheader("Portfolio Insights")

    largest_sector = sector_allocation.idxmax()
    largest_weight = sector_allocation.max()

    st.info(
        f"**Sector exposure:** {largest_sector} represents "
        f"{largest_weight:.1f}% of your portfolio."
    )

    st.info(
        f"**Diversification:** Your portfolio contains {len(tickers)} "
        f"stocks, with an effective number of "
        f"{effective_holdings:.2f} equally weighted holdings."
    )

    st.info(
        f"**Downside risk:** Your portfolio experienced a maximum "
        f"historical drawdown of {max_drawdown:.2%} "
        f"during the selected period."
    )

# Improve text contrast across the dark dashboard.
st.markdown(
    """
    <style>
    /* Main text and captions */
    .stApp {
        color: #F1F5FF;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"] p {
        color: #C0CBE0;
    }

    /* Sidebar labels and descriptions */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #E2E8F5 !important;
    }

    /* Date and number inputs */
    [data-testid="stSidebar"] input {
        color: #172033 !important;
    }

    /* Navigation tabs */
    button[data-baseweb="tab"] {
        color: #B8C6DE !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #A995FF !important;
        font-weight: 600;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] p {
        color: #B8C6DE !important;
    }

    /* Metric numbers */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    /* Dataframe background */
    [data-testid="stDataFrame"] {
        background-color: #151F33;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
