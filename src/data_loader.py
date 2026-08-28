"""Download and prepare historical market-price data."""

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "JPM",
    "GS",
    "XOM",
    "JNJ",
    "WMT",
    "CAT",
    "TSLA",
]

DEFAULT_BENCHMARK = "SPY"


def download_prices(
    tickers: Sequence[str],
    start: str | date | datetime,
    end: str | date | datetime | None = None,
) -> pd.DataFrame:
    """Download adjusted closing prices for one or more assets."""

    normalized_tickers = [
        ticker.strip().upper()
        for ticker in tickers
        if ticker.strip()
    ]

    if not normalized_tickers:
        raise ValueError("Please provide at least one ticker.")

    data = yf.download(
        normalized_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No price data was downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data["Close"].to_frame(name=normalized_tickers[0])

    prices = prices.dropna(how="all").sort_index()

    return prices


if __name__ == "__main__":
    sample_prices = download_prices(
        tickers=["AAPL", "MSFT", "JPM"],
        start="2025-01-01",
    )

    print(sample_prices.tail())