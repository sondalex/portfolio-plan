from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


def generate_stock_series(
    symbols: list[str],
    start_date: date,
    end_date: date,
    initial_price: float = 100.0,
    volatility: float = 5.0,
    seed: Optional[int] = 1,
) -> pd.Series:
    """
    Generate financial time series data using a Wiener process:

    For t ∈ [0,1], the discrete Wiener process is defined as:
        W_n(t) = (1/√n) ∑_{1≤k≤⌊nt⌋} ξ_k

    The price process is modeled as:
        P(t) = P₀ + σ·W_n(t)

    where:
        P₀ = initial_price (base price)
        σ = volatility (scaling factor)

    Example
    -------

    >>> from datetime import date
    >>> generate_stock_series(
            ["a", "b"],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

    """
    np.random.seed(seed)
    n_symbols = len(symbols)

    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(business_days)

    n = 1000

    xi = np.random.normal(0, 1, size=(n_symbols, n))

    price_matrix = np.zeros((n_symbols, n_days))

    for i in range(n_days):
        t = (i + 1) / n_days

        nt_floor = int(np.floor(n * t))

        for j in range(n_symbols):
            price_matrix[j, i] = (1 / np.sqrt(n)) * np.sum(xi[j, :nt_floor])

    price_matrix = initial_price + volatility * price_matrix

    symbol_idx = np.repeat(np.arange(n_symbols), n_days)
    date_idx = np.tile(np.arange(n_days), n_symbols)

    index = pd.MultiIndex.from_arrays(
        [np.array(symbols)[symbol_idx], business_days[date_idx]],
        names=["symbol", "date"],
    )

    result = pd.Series(
        price_matrix.flatten("C"),
        index=index,
    )

    return result


def generate_stock_data(
    symbols: list[str],
    start_date: date,
    end_date: date,
    open_initial_price: float = 100.0,
    close_initial_price: float = 200.0,
    low_initial_price: float = 300.0,
    high_initial_price: float = 400.0,
    volatility: float = 5.0,
    seed: Optional[int] = 1,
):
    open = generate_stock_series(
        symbols, start_date, end_date, open_initial_price, volatility, seed
    )
    open.name = "open"
    close = generate_stock_series(
        symbols, start_date, end_date, close_initial_price, volatility, seed
    )
    close.name = "close"
    low = generate_stock_series(
        symbols, start_date, end_date, low_initial_price, volatility, seed
    )
    low.name = "low"

    high = generate_stock_series(
        symbols, start_date, end_date, high_initial_price, volatility, seed
    )
    high.name = "high"
    return pd.concat([open, close, low, high], axis=1)
