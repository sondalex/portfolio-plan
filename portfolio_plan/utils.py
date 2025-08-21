import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Literal, Optional, cast

import empyrical as ep
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_grid,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_rect,
    geom_text,
    ggplot,
    position_dodge,
    scale_x_datetime,
    theme,
    ylab,
    ylim,
)

from portfolio_plan._properties import Frequency, Weights
from portfolio_plan.visualisation import (
    THEME_ROSE_PINE_BASE_SIZE,
    scale_alphabet_fill_discrete,
    scale_brewer_fill_discrete,
    scale_rose_pine_discrete,
    scale_rose_pine_fill_discrete,
    theme_rose_pine,
)


def validate_dataframe(data: pd.DataFrame, check_frequency: bool = True):
    assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
    assert not data.empty, "DataFrame should not be empty"
    columns = data.columns.tolist()
    expected_columns = ["open", "close", "high", "low"]
    assert set(columns) == set(expected_columns), (
        f"DataFrame must have exactly columns {expected_columns}. Got {columns}"
    )

    assert isinstance(data.index, pd.MultiIndex), "Index must be a MultiIndex"
    expected_index_names = ["symbol", "date"]
    index_names = data.index.names
    assert index_names == expected_index_names, (
        f"Index names must be {expected_index_names} got {index_names}"
    )

    row_has_nan = data.isna().any(axis=1)
    row_all_nan = data.isna().all(axis=1)
    rows_with_partial_nan = row_has_nan & ~row_all_nan

    if rows_with_partial_nan.any():
        first_bad_row = rows_with_partial_nan[rows_with_partial_nan].index[0]
        symbol, date = first_bad_row
        values = data.loc[first_bad_row]

        raise AssertionError(
            f"Invalid NaN pattern found for symbol {symbol} at date {date}. "
            f"If a row contains NaN, all values in that row must be NaN. "
            f"Got values: {values.to_dict()}"
        )
    for symbol, df in data.groupby(level="symbol"):
        date_index: pd.DatetimeIndex = df.index.get_level_values("date")  # type: ignore

        assert (
            date_index.is_monotonic_increasing
        ), f"""Dates must be sorted for symbol {symbol}.
            Are dates unique ? Got {date_index}
            """
        if check_frequency:
            _validate_frequency(date_index)


def _validate_frequency(index: pd.DatetimeIndex, symbol: Optional[str] = None):
    if index.shape[0] > 1:
        inferred_freq = pd.infer_freq(index)
        if symbol is not None:
            assert inferred_freq is not None, (
                f"Date index must have an inferrable frequency for symbol {symbol}. Found freq=None"
            )
        else:
            assert inferred_freq is not None, (
                "Date index must have an inferrable frequency for current symbol. Found freq=None"
            )

        return
    # else, the index is a singleton


def _returns_from_prices(prices: pd.DataFrame):
    """
    Calculate the returns

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC price data

    Returns
    -------
    pd.DataFrame
        Returns derived from prices
    """
    # TODO: check that NaS are equivalent in all OHLC dimension
    index = prices.index

    mask = prices.isna().all(axis=1).to_numpy()
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    pct_change = prices[~mask].pct_change().to_numpy()
    out[~mask] = pct_change
    returns = pd.DataFrame(out, index=index, columns=prices.columns)
    return returns


def returns_from_prices(
    data: pd.DataFrame, check_frequency: bool = True
) -> pd.DataFrame:
    """
    Parameters
    ==========
    data: pd.DataFrame
        Take an input DataFrame, transform it.
    check_frequency: bool
        See :py:meth:`validate_dataframe`
        Defaults to True
    """
    validate_dataframe(data, check_frequency=check_frequency)
    symbols = data.index.get_level_values("symbol")
    first_symbol = symbols[0]
    freq = data.loc[first_symbol].index.freq
    prices = data.unstack(level="symbol")  # Creates multiindex column
    prices = cast(pd.DataFrame, prices)
    returns = _returns_from_prices(prices)
    if freq is not None:
        returns = returns.asfreq(freq)
    stacked_returns = returns.stack(dropna=False).reorder_levels([1, 0])
    stacked_returns.columns.name = "Returns"

    return stacked_returns


# TODO: test with singleton
def cumulative_from_returns(
    data: pd.DataFrame, starting_value: float = 1.0, keepna: bool = True
) -> pd.DataFrame:
    # TODO: Implement validation of data
    assert isinstance(data, pd.DataFrame)
    usymbols = data.index.get_level_values("symbol").unique()
    newreturns: List[pd.DataFrame] = []
    for symbol in usymbols:
        returns = data.loc[symbol, :]
        if keepna:
            not_a_nan_row = ~returns.isna().any(axis=1)
            if not_a_nan_row[0] == np.False_:
                not_a_nan_row[0] = np.True_
            out = np.empty(returns.shape, dtype=np.float64)
            out[:] = np.nan
            returns_without_nan = returns.loc[not_a_nan_row, :]

            # NOTE: Probably the element causing the discrepency
            out[not_a_nan_row] = ep.cum_returns(
                returns_without_nan.values,
                starting_value=starting_value,
                out=out[not_a_nan_row, :],
            )
            cumulative_returns = pd.DataFrame(
                out, columns=returns.columns, index=returns.index
            )
        else:
            cumulative_returns = ep.cum_returns(returns, starting_value=starting_value)
        cumulative_returns.name = symbol
        newreturns.append(cumulative_returns)

    newreturns_data: pd.DataFrame = pd.concat(  # type: ignore
        newreturns,
        axis=0,
        keys=[df.name for df in newreturns],
        names=["symbol", "date"],
    )
    newreturns_data.columns.name = "CumulativeReturns"
    return newreturns_data


def _find_gap_lines(data: pd.Series):
    """
    Find gaps (even multiple missing points) and build dashed lines to connect
    valid data.
    The function connects the last valid point before a gap to the
    first valid point after the gap.
    """
    name = data.name

    if data.empty:
        raise ValueError("Empty Series, cannot find gaps in empty data.")

    data = data.sort_index()
    records = []
    # TODO: ignore blocks of NAN which start at index 0 (there's no before)
    # & Ignore blocks of NaN that end at index -1 (there's no after)
    for symbol, group in data.groupby(level="symbol"):
        series = group.droplevel("symbol")

        is_na = series.isna()
        if not is_na.any():
            continue

        gap_start_idx = None
        last_valid_value = None
        last_valid_date = None

        for idx, (date, value) in enumerate(series.items()):
            if pd.isna(value):
                if idx == 0:
                    continue
                # If we find a NaN and we haven't started a gap yet
                if gap_start_idx is None and last_valid_value is not None:
                    gap_start_idx = idx
            else:
                # If we have a valid value and we were in a gap
                if gap_start_idx is not None:
                    # Connect the last valid point before the gap to this point
                    records.append(
                        {
                            "symbol": symbol,
                            "date_start": last_valid_date,
                            "value_start": last_valid_value,
                            "date_end": date,
                            "value_end": value,
                        }
                    )
                    gap_start_idx = None

                # Update the last valid point
                last_valid_value = value
                last_valid_date = date

    if not records:
        return pd.DataFrame()

    gap_lines = pd.DataFrame(records)

    # Flatten into long format
    gap_lines_long = pd.DataFrame(
        {
            "symbol": np.ravel(
                np.column_stack(
                    [gap_lines["symbol"].to_numpy(), gap_lines["symbol"].to_numpy()]
                )
            ),
            "date": np.ravel(
                np.column_stack(
                    [
                        gap_lines["date_start"].to_numpy(),
                        gap_lines["date_end"].to_numpy(),
                    ]
                )
            ),
            name: np.ravel(
                np.column_stack(
                    [
                        gap_lines["value_start"].to_numpy(),
                        gap_lines["value_end"].to_numpy(),
                    ]
                )
            ),
            # Add the group column back to ensure proper line segments
            "group": np.repeat(np.arange(len(gap_lines)), 2),
        }
    )

    return gap_lines_long


@dataclass
class Period:
    """"""

    """The start of the period"""
    start: datetime
    """The end of the period"""
    end: datetime
    """The frequency of data within the period"""
    frequency: Frequency | None


def _geom_lines(
    data: pd.Series, periods: List[Period] | None = None
) -> List[geom_line | geom_point]:
    name = data.name

    if data.empty:
        return []

    data_subsets = []
    gap_lines_list = []

    if periods:
        dates = data.index.get_level_values("date")
        for period in periods:
            subset = data.loc[(dates >= period.start) & (dates <= period.end)]
            gap_lines_list.append(_find_gap_lines(subset))
            data_subsets.append(subset)
    else:
        # NOTE: Append a single item
        data_subsets.append(data)
        gap_lines_list.append(_find_gap_lines(data))

    geoms: List[geom_line | geom_point] = []
    for gap_lines_long, subset in zip(gap_lines_list, data_subsets):
        geoms.append(
            # Straight line
            geom_line(
                data=subset.reset_index(),
                mapping=aes(x="date", y=name, group="symbol", color="symbol"),
                # TODO: IMPLEMENT COLOR MAPPING FUNCTION
            )
        )

        if not gap_lines_long.empty:
            geoms.append(
                # Dashed line
                geom_line(
                    data=gap_lines_long,
                    mapping=aes(x="date", y=name, group="group", color="symbol"),
                    linetype="dashed",
                )
            )
        group_sizes = subset.groupby(level="symbol").size()
        singleton_groups = group_sizes[group_sizes == 1].index
        singleton_df = subset.loc[singleton_groups, :]
        if not singleton_df.empty:
            geoms.append(
                geom_point(
                    data=singleton_df.reset_index(),
                    mapping=aes(x="date", y=name, group="symbol", color="symbol"),
                )
            )

    return geoms


def _scale_date(dates: pd.DatetimeIndex, periods: List[Period] | None = None):
    if periods:
        assert all(periods[i] != periods[i - 1] for i in range(1, len(periods)))
        freq = periods[0].frequency
    else:
        try:
            inferred = pd.infer_freq(dates)
            if inferred is None:
                raise ValueError("Failed to infer frequency")
            freq = Frequency.from_pandas(inferred)
        except ValueError:
            freq = None

    breaks = "1 year"
    labels = "%Y"

    if freq is not None:
        match freq:
            case Frequency.MINUTE:
                breaks = "1 minute"
                labels = "%H:%M"
            case Frequency.DAILY:
                breaks = "1 day"
                labels = "%Y-%m-%d"
            case Frequency.WEEKLY:
                breaks = "1 week"
                labels = "%Y-%m-%d"
            case Frequency.MONTHLY:
                breaks = "1 month"
                labels = "%Y-%m"
            case _:
                breaks = "1 year"
                labels = "%Y"
    else:
        warnings.warn(
            f"Failed to infer frequency, defaulting to x scale with date_breaks={breaks} and date_labels={labels}"
        )

    return scale_x_datetime(date_breaks=breaks, date_labels=labels)


def plot_lines(
    data: pd.Series,
    scale_x: bool = True,
    periods: List[Period] | None = None,
    variant: Literal["main", "moon", "dawn"] = "dawn",
) -> ggplot:
    """
    Parameters
    ----------
    data: pd.Series
        Data in long form, with multiindex, first level "symbol", second level "date".
    scale_x: bool
        Whether to include a date scale on the x axis
        For data which has uneven frequency, it is recommend to set to False and define
        the scale manually.
        Defaults to True
    periods: List[Period] | None
        If a list of Period is provided, transparent rectangle will be drawn on
        each period.start, period.end
        Defaults to None
    variant: Literal["main", "moon", "dawn"]
        Refer to :py:class:`portfolio_plan.visualisation.theme_rose_pine`
    """
    base_plot = ggplot() + scale_rose_pine_discrete(variant=variant)
    for geom in _geom_lines(data, periods):
        base_plot += geom
    if scale_x:
        symbols = data.index.get_level_values("symbol")
        first_symbol = symbols[0]

        subset = data.loc[first_symbol]
        dates: pd.DatetimeIndex = subset.index.get_level_values("date")  # type: ignore
        base_plot = base_plot + _scale_date(dates, periods)
    if periods is not None:
        periods_data = pd.DataFrame.from_records(
            {
                "start_date": period.start,
                "end_date": period.end,
                "max_value": data.max(),
                "min_value": data.min(),
            }
            for period in periods
        )
        base_plot = base_plot + geom_rect(
            data=periods_data,
            mapping=aes(
                xmin="start_date",
                xmax="end_date",
                ymin="min_value",
                ymax="max_value",
            ),
            alpha=0.2,
        )
    return (
        base_plot
        + theme_rose_pine(variant=variant)
        + theme(axis_text_x=element_text(angle=90))
    )


def plot_prices(data: pd.Series, scale_x: bool = True) -> ggplot:
    """
    A wrapper for :py:func:`plot_lines`
    """
    return plot_lines(data, scale_x)


def plot_returns(data: pd.Series, scale_x: bool = True) -> ggplot:
    """
    A wrapper for :py:func:`plot_lines`, adds a facet layer.
    Each symbol is facetted in order to improve readability of the chart.
    """
    return plot_lines(data) + facet_grid("symbol ~ .")


def plot_cumulative_returns(
    data: pd.Series, scale_x: bool = True, periods=List[Period] | None
) -> ggplot:
    """
    A wrapper for :py:func:`plot_lines`
    """
    return plot_lines(data, scale_x, periods=periods)


def add_singleton_level(data: pd.DataFrame, value: Any) -> None:
    """
    Warning
    -------
    Mutates the original DataFrame
    """
    new_level = [value] * data.shape[0]
    data.index = pd.MultiIndex.from_arrays(
        [new_level, data.index],
        names=["symbol", data.index.name or "date"],
    )


def _create_str_stack(title: str, values: List[str]):
    max_length = max(len(title), *(len(value) for value in values))
    line = "-" * (max_length + 4)

    header = f"{line}\n| {title.ljust(max_length)} |\n{line}"

    content = ""
    for value in values:
        content += f"\n| {value.ljust(max_length)} |\n{line}"

    return header + content


def _geom_text(data: pd.Series, text: str) -> geom_text:
    name = data.name
    labels = np.empty(data.shape[0], dtype="object")
    labels.fill(pd.NA)
    labels[-1] = text
    dtype = pd.StringDtype()
    data = data.reset_index(drop=False)  # type: ignore
    data["label"] = pd.Series(labels, dtype=dtype)

    return geom_text(data=data, mapping=aes(x="date", y=name, label="label"))


def plot_allocations(
    weights: List[Weights] | Weights,
    periods: List[Period] | None = None,
    variant: Literal["main", "moon", "dawn"] = "dawn",
    base_size: int = THEME_ROSE_PINE_BASE_SIZE,
) -> ggplot:
    """
    Plot allocations over multiple periods.

    Parameters
    ----------
    weights : List[Weights] | Weights
        A list of weight dictionaries, where each dictionary represents
        the allocation of weights for a specific period, or a single Weights object.
    periods : List[Period] | None
        A list of Period objects corresponding to the weights, or None for a single allocation.
    variant : Literal["main", "moon", "dawn"]
        The color variant to use for the plot.

    Returns
    -------
    ggplot
        A ggplot object representing the allocations.
    """
    if not isinstance(weights, (list, Weights)):
        raise TypeError("weights must be a List[Weights] or a single Weights object.")
    if periods is not None and not isinstance(periods, list):
        raise TypeError("periods must be a List[Period] or None.")

    if periods is None:
        if not isinstance(weights, Weights):
            raise ValueError(
                "When periods=None, weights must be a single Weights object."
            )
        if weights.empty:
            raise ValueError("Weights can not be empty")
        data = [
            {"symbol": symbol, "weight": weight} for symbol, weight in weights.items()
        ]
        df = pd.DataFrame(data)
    else:
        if len(weights) != len(periods):
            raise ValueError("The number of weights must match the number of periods.")
        data = []
        for w, period in zip(weights, periods):
            if w.empty:
                raise ValueError("Weights can not be empty")
            period_label = (
                f"{period.start.strftime('%b %Y')} - {period.end.strftime('%b %Y')}"
            )
            for symbol, weight in w.items():
                data.append(
                    {
                        "symbol": symbol,
                        "weight": weight,
                        "period_label": period_label,
                    }
                )
        df = pd.DataFrame(data)

    base = (
        ggplot(df, aes(x="symbol", y="weight", fill="symbol"))
        + geom_col()
        + ylim(-0.0, 1.02)
        + theme_rose_pine(variant=variant, base_size=base_size)
        + ylab("Allocation")
    )
    n_symbols = df["symbol"].unique().shape[0]
    if n_symbols <= 6:
        base += scale_rose_pine_fill_discrete(variant=variant)
    elif n_symbols <= 12:
        base += scale_brewer_fill_discrete()
    else:
        base += scale_alphabet_fill_discrete()

    if periods is not None:
        base += facet_wrap("~period_label", scales="free_y")

    return base


def plot_allocations2(
    weights: List[Weights] | Weights,
    periods: List[Period] | None = None,
    variant: Literal["main", "moon", "dawn"] = "dawn",
    base_size: int = THEME_ROSE_PINE_BASE_SIZE,
    text_size: float = 0.8 * THEME_ROSE_PINE_BASE_SIZE,
    text_angle: float = 45,
) -> ggplot:
    """
    Plot allocations over multiple periods.

    Parameters
    ----------
    weights : List[Weights] | Weights
        A list of weight dictionaries, where each dictionary represents
        the allocation of weights for a specific period, or a single Weights object.
    periods : List[Period] | None
        A list of Period objects corresponding to the weights, or None for a single allocation.
    variant : Literal["main", "moon", "dawn"]
        The color variant to use for the plot.

    Returns
    -------
    ggplot
        A ggplot object representing the allocations.
    """
    if not isinstance(weights, (list, Weights)):
        raise TypeError("weights must be a List[Weights] or a single Weights object.")
    if periods is not None and not isinstance(periods, list):
        raise TypeError("periods must be a List[Period] or None.")

    if periods is None:
        if not isinstance(weights, Weights):
            raise ValueError(
                "When periods=None, weights must be a single Weights object."
            )
        data = [
            {"symbol": symbol, "weight": weight} for symbol, weight in weights.items()
        ]
        df = pd.DataFrame(data)
        df["symbol"] = pd.Categorical(df["symbol"], ordered=True)
    else:
        if len(weights) != len(periods):
            raise ValueError("The number of weights must match the number of periods.")
        data = [
            {
                "symbol": symbol,
                "weight": weight,
                "start_date": period.start,
                "end_date": period.end,
                "period_label": f"{period.start.strftime('%b %Y')} - {period.end.strftime('%b %Y')}",
            }
            for w, period in zip(weights, periods)
            for symbol, weight in w.items()
        ]
        df = pd.DataFrame(data)
        df["symbol"] = pd.Categorical(df["symbol"], ordered=True)
        df["period_label"] = pd.Categorical(df["period_label"], ordered=True)
    dodge_text = position_dodge(width=0.9)
    base = (
        ggplot(df, aes(x="period_label", y="weight", fill="symbol"))
        + geom_col(stat="identity", position="dodge", show_legend=False)
        + geom_text(
            aes(y=-0.01, label="symbol"),  # , color="symbol"),
            position=dodge_text,
            size=text_size,
            angle=text_angle,
            va="top",
        )
        + ylim(-0.08, 1.02)
        + theme_rose_pine(variant=variant, base_size=base_size)
        + theme(axis_title_x=element_blank())
        + ylab("Allocation")
    )
    n_symbols = df["symbol"].unique().shape[0]
    if n_symbols <= 6:
        base += scale_rose_pine_fill_discrete(variant=variant)
    elif n_symbols <= 12:
        base += scale_brewer_fill_discrete()
    else:
        base += scale_alphabet_fill_discrete()

    return base


def plot_allocation(
    weights: Weights,
    variant: Literal["main", "moon", "dawn"] = "dawn",
    base_size: int = THEME_ROSE_PINE_BASE_SIZE,
) -> ggplot:
    """
    Plot a single allocation.

    Parameters
    ----------
    weights : Weights
        A dictionary representing the allocation of weights.
    variant : Literal["main", "moon", "dawn"]
        The color variant to use for the plot.

    Returns
    -------
    ggplot
        A ggplot object representing the allocation.
    """
    return plot_allocations(
        weights=weights, periods=None, variant=variant, base_size=base_size
    )
