from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from empyrical import excess_sharpe

if TYPE_CHECKING:
    from portfolio_plan.core import Plan

from portfolio_plan.financialseries import Returns
from portfolio_plan.utils import add_singleton_level


class DifferentIndexError(ValueError):
    pass


def _validate_information_ratio_input(
    returns: pd.DataFrame, benchmark_returns: pd.DataFrame
):
    if isinstance(returns.index, pd.MultiIndex):
        raise ValueError("Expected single index")
    if isinstance(benchmark_returns.index, pd.MultiIndex):
        raise ValueError("Expected single index")
    if not benchmark_returns.index.equals(returns.index):
        raise DifferentIndexError(
            "returns and benchmark_returns do not share same index values"
        )
    assert returns.index.is_monotonic_increasing


def information_ratio(
    returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    name: str = "",
    keep_index: bool = False,
) -> pd.DataFrame:
    """
    Compute the non-annualized information ratio for all OHLC columns


    Parameters
    ----------
    returns: pd.DataFrame
        OHLC returns of one asset
    benchmark_returns: pd.DataFrame
        OHLC returns of one benchmark asset

    Returns
    -------
    A single index DataFrame with the IR for each dimension of OHLC
    """
    # NOTE: empyrical.excess_sharpe is an information ratio with no annualization
    try:
        _validate_information_ratio_input(returns, benchmark_returns)
    except DifferentIndexError as e:
        if returns.index.shape[0] > benchmark_returns.index.shape[0]:
            # TODO: check what are the values that are present in the other
            if not set(benchmark_returns.index).issubset(returns.index):
                raise e
            benchmark_returns = benchmark_returns.reindex(returns.index)
        elif returns.index.shape[0] < benchmark_returns.index.shape[0]:
            if not set(returns.index).issubset(returns.index):
                raise e
            returns = returns.reindex(benchmark_returns.index)
        else:
            raise e

    if keep_index:
        nrow = max(1, returns.shape[0])
        index = returns.index
    else:
        nrow = 1
        max_date = returns.index.get_level_values("date").max()
        index = pd.Index([max_date])

    out = np.empty((nrow, 4), dtype=np.float64)
    out[:] = np.nan
    columns = ["open", "low", "high", "close"]
    for i, dim in enumerate(columns):
        r = returns[dim].to_numpy()
        b = benchmark_returns[dim].to_numpy()
        excess_sharpe(r, b, out=out[-1:, i])

    return pd.DataFrame(out, columns=columns, index=index)


class InformationRatio:
    def __init__(self, reference: Plan, benchmark: Plan):
        self._reference = reference
        self._benchmark = benchmark

    def period_returns(self, keep_index: bool = False) -> Returns:
        """
        Parameters
        ---------
        keep_index: bool
            See :py:func:`information_ratio`
        Returns
        -------
            Returns, a singleton return
        """
        portfolio_returns = self._reference.portfolio_returns()
        benchmark_returns = self._benchmark.portfolio_returns()
        assert portfolio_returns.is_unique_symbol
        assert benchmark_returns.is_unique_symbol
        pdata = portfolio_returns.data.droplevel(level="symbol")
        bdata = benchmark_returns.data.droplevel(level="symbol")

        ir = information_ratio(pdata, bdata, keep_index=keep_index)
        portfolio_returns.data.index.get_level_values("symbol")
        name = f"IR: {portfolio_returns.name}-{benchmark_returns.name}"
        add_singleton_level(ir, name)
        ir.name = "IR"
        return Returns(ir, "IR")
