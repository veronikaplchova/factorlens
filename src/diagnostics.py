
"""Statistical inference for portfolio factor regressions."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_factor_diagnostics(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
    factor_columns: list[str],
    max_lags: int = 5,
) -> pd.DataFrame:
    """Estimate factor coefficients with HAC standard errors."""

    if not factor_columns:
        raise ValueError("At least one factor is required.")

    required = factor_columns + ["RF"]

    if not set(required).issubset(factor_data.columns):
        raise ValueError("Required factor columns are missing.")

    if not isinstance(max_lags, int) or max_lags < 0:
        raise ValueError("max_lags must be a nonnegative integer.")

    data = pd.concat(
        [portfolio_returns.rename("Portfolio"), factor_data[required]],
        axis=1,
        join="inner",
    ).dropna()

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Returns and factors must be finite.")

    if len(data) <= len(factor_columns) + 1:
        raise ValueError("Insufficient regression observations.")

    y = data["Portfolio"] - data["RF"]
    X = sm.add_constant(data[factor_columns], has_constant="add")

    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("Factors are linearly dependent.")

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )

    confidence = model.conf_int(alpha=0.05)

    return pd.DataFrame(
        {
            "Coefficient": model.params,
            "Standard Error": model.bse,
            "t-statistic": model.tvalues,
            "p-value": model.pvalues,
            "CI Lower": confidence.iloc[:, 0],
            "CI Upper": confidence.iloc[:, 1],
        }
    )
