"""Tests for portfolio input validation."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    clean_tickers,
    validate_date_range,
    validate_weights,
)


def test_clean_tickers_normalizes_inputs():
    result = clean_tickers([" aapl ", "msft", " JPM"])

    assert result == ["AAPL", "MSFT", "JPM"]


def test_clean_tickers_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate tickers"):
        clean_tickers(["AAPL", "aapl"])


def test_clean_tickers_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one ticker"):
        clean_tickers(["", "   "])


def test_validate_date_range_accepts_valid_dates():
    start, end = validate_date_range("2020-01-01", "2025-01-01")

    assert start == pd.Timestamp("2020-01-01")
    assert end == pd.Timestamp("2025-01-01")


def test_validate_date_range_rejects_reversed_dates():
    with pytest.raises(ValueError, match="earlier than"):
        validate_date_range("2025-01-01", "2020-01-01")


def test_validate_weights_accepts_valid_portfolio():
    result = validate_weights([0.50, 0.30, 0.20], number_of_assets=3)

    np.testing.assert_allclose(result, [0.50, 0.30, 0.20])


def test_validate_weights_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError, match="number of weights"):
        validate_weights([0.60, 0.40], number_of_assets=3)


def test_validate_weights_rejects_negative_weights():
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_weights([0.80, 0.30, -0.10], number_of_assets=3)


def test_validate_weights_rejects_incorrect_total():
    with pytest.raises(ValueError, match="add up to 1.0"):
        validate_weights([0.50, 0.30, 0.10], number_of_assets=3)