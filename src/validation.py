"""Validate portfolio inputs before performing calculations."""

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd


DateInput = str | date | datetime | pd.Timestamp


def clean_tickers(tickers: Sequence[str]) -> list[str]:
    """Clean ticker symbols and reject empty or duplicate inputs."""
    normalized_tickers = [
        ticker.strip().upper()
        for ticker in tickers
        if ticker.strip()
    ]

    if not normalized_tickers:
        raise ValueError("Please provide at least one ticker.")

    seen = set()
    duplicates = set()

    for ticker in normalized_tickers:
        if ticker in seen:
            duplicates.add(ticker)
        seen.add(ticker)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate tickers detected: {duplicate_list}")

    return normalized_tickers


def validate_date_range(
    start: DateInput,
    end: DateInput,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Check that the requested start and end dates are logically valid."""
    try:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
    except (TypeError, ValueError) as error:
        raise ValueError("Start and end must be valid dates.") from error

    if start_date >= end_date:
        raise ValueError("The start date must be earlier than the end date.")

    return start_date, end_date


def validate_weights(
    weights: Sequence[float],
    number_of_assets: int,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate long-only portfolio weights expressed as decimals."""
    weight_array = np.asarray(weights, dtype=float)

    if weight_array.ndim != 1:
        raise ValueError("Portfolio weights must be one-dimensional.")

    if len(weight_array) != number_of_assets:
        raise ValueError(
            "The number of weights must match the number of assets."
        )

    if not np.all(np.isfinite(weight_array)):
        raise ValueError("Portfolio weights must contain finite numbers.")

    if np.any(weight_array < 0):
        raise ValueError("Portfolio weights cannot be negative.")

    if not np.isclose(weight_array.sum(), 1.0, atol=tolerance):
        raise ValueError("Portfolio weights must add up to 1.0.")

    return weight_array