
"""Download and prepare Fama-French factor data."""

import pandas as pd
from pandas_datareader import data as web


FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]


def load_fama_french_factors(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily Fama-French three-factor data."""

    if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise ValueError("Start date must be before end date.")

    dataset = web.DataReader(
        "F-F_Research_Data_Factors_daily",
        "famafrench",
        start=start_date,
        end=end_date,
    )

    factors = dataset[0].copy()

    if not set(FACTOR_COLUMNS).issubset(factors.columns):
        raise ValueError("Required factor columns are missing.")

    factors = factors[FACTOR_COLUMNS]

    # French's data is reported in percent.
    # Convert to decimals to match stock returns.
    factors = factors.astype(float) / 100.0

    
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    else:
        factors.index = pd.to_datetime(factors.index)

    factors = factors.sort_index()

    if factors.empty:
        raise ValueError("No factor data available for this period.")

    if factors.isna().any().any():
        raise ValueError("Factor data contains missing values.")

    return factors
