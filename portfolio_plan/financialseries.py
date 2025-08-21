from typing import List, Literal

import numpy as np
import pandas as pd
from plotnine.ggplot import ggplot

from portfolio_plan import const
from portfolio_plan._properties import Frequency, Weights
from portfolio_plan.abc import FinancialSeries
from portfolio_plan.errors import ValidationError
from portfolio_plan.utils import (
    Period,
    _geom_lines,
    _geom_text,
    add_singleton_level,
    cumulative_from_returns,
    plot_cumulative_returns,
    plot_prices,
    plot_returns,
    validate_dataframe,
)


class Prices(FinancialSeries):
    def __init__(self, data: pd.DataFrame, name: str):
        """
        Parameters
        ----------
        data: pd.DataFrame
            The data including financial prices.
        name: str
            The name of this set of financial prices.

        Note
        ----
        It is recommended to derive `Prices` instance from a resource rather
        than from direct instantiation.
        A resource, can be a file, an in memory parquet buffer, a network resource.
        Refer to :ref:`resource API`
        """

        super().__init__(data, name)

    def validate_data(self, data: pd.DataFrame):
        validate_dataframe(data)

    def resample_period(self, frequency: Frequency):
        ohlc_dict = {
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
        }
        usymbols = self.data.index.get_level_values("symbol").unique()
        frequencies = [
            pd.infer_freq(self.data.loc[symbol].index) for symbol in usymbols
        ]
        assert all(
            frequencies[i] == frequencies[i - 1] for i in range(1, len(frequencies))
        )
        current_freq = frequencies[0]
        if current_freq is None:
            raise ValueError("Cannot infer current frequency of the data.")

        def freq_to_timedelta(freq):
            match freq:
                case "1B":
                    approx_timedelta = pd.Timedelta(days=1.4) * 5
                    return approx_timedelta
                case "1D" | "1W" | "1M" | "1T":
                    pass
                    return pd.to_timedelta(freq)
                case _:
                    raise ValueError(f"Upsampling of {current_freq} is not supported.")

        current_td = freq_to_timedelta("1" + current_freq)

        target_td = freq_to_timedelta(frequency.to_pandas())
        target_freq = frequency.to_pandas()
        if target_td < current_td:
            raise ValueError(
                f"""Sampling from {current_freq} to {target_freq} is not allowed.
                Down sampling is not supported"""
            )

        resampled = (
            self._data.groupby(level="symbol")
            .resample(target_freq, level="date")
            .agg(ohlc_dict)  # type: ignore
        )
        resampled.index = pd.MultiIndex.from_tuples(
            resampled.index.to_flat_index(), names=["symbol", "date"]
        )
        return self.__class__(data=resampled, name=self.name)

    def plot_low(self, scale_x: bool = True) -> ggplot:
        """
        Plot all prices low price

        Example
        -------

        >>> from portfolio_plan.financialseries import Prices, Frequency
        >>> from portfolio_plan.resource import File
        >>> from portfolio_plan.data import example_prices_path


        >>> resource = File(path=example_prices_path(), frequency=Frequency.BDAILY)
        >>> prices = resource.fetch(
        ...     ["A", "B"],
        ...     start_date="2025-01-01",
        ... end_date="2025-01-31"
        )
        >>> price.plot_low()
        """
        return plot_prices(self._data["low"], scale_x)

    def plot_high(self, scale_x: bool = True) -> ggplot:
        """
        Plot all prices high price
        """
        return plot_prices(self._data["high"], scale_x)

    def plot_open(self, scale_x: bool = True) -> ggplot:
        """
        Plot all prices open price
        """

        return plot_prices(self._data["open"], scale_x)

    def plot_close(self, scale_x: bool = True) -> ggplot:
        """
        Plot all price close price
        """
        return plot_prices(self._data["close"], scale_x)

    def plot(self, scale_x: bool = True):
        """
        Plot all prices candle chart
        """
        raise NotImplementedError()


class CumulativeReturns(FinancialSeries):
    def __init__(
        self,
        data: pd.DataFrame,
        name: str,
        check_base: bool = True,
        periods: List[Period] | None = None,
        check_frequency: bool = True,
    ):
        """
        Parameters
        ----------
        data: pd.DataFrame
            A dataframe containing cumulative OHLC returns
        name: str
            Name of the cumulative returns
        check_base: bool
            Whether to enforce first value to be equal across open, close, high, low.
            Defaults to True
        periods: List[Period] | None
            See :py:meth:`portfolio_plan.utils.Period`
        """
        self._check_frequency = check_frequency
        super().__init__(data, name)
        subset = data.groupby("symbol").nth(0)
        if check_base:
            if not np.all(subset.values == subset.values[0, 0]):
                raise ValidationError(
                    "Cumulative Returns should always start with same base value across symbols."
                )
        self._periods = periods

    def validate_data(self, data: pd.DataFrame):
        validate_dataframe(data, check_frequency=self._check_frequency)

    def plot_low(self, scale_x: bool = True) -> ggplot:
        """
        Plot all cumulative returns high price
        """
        return plot_cumulative_returns(self._data["low"], scale_x, periods=self.periods)

    def plot_high(self, scale_x: bool = True) -> ggplot:
        """
        Plot all cumulative returns high price
        """
        return plot_cumulative_returns(
            self._data["high"], scale_x, periods=self.periods
        )

    def plot_open(self, scale_x: bool = True) -> ggplot:
        """
        Plot all cumulative returns open price
        """

        return plot_cumulative_returns(
            self._data["open"], scale_x, periods=self.periods
        )

    def plot_close(self, scale_x: bool = True) -> ggplot:
        """
        Plot all cumulative returns close price
        """
        return plot_cumulative_returns(
            self._data["close"], scale_x, periods=self.periods
        )

    def plot(self, scale_x: bool = True):
        """
        Plot all cumulative returns candle chart
        """
        raise NotImplementedError()

    @property
    def periods(self) -> List[Period] | None:
        return self._periods


class Returns(FinancialSeries):
    def __init__(self, data: pd.DataFrame, name: str):
        """
        Parameters
        ----------
        data: pd.DataFrame
            A dataframe containing the OHLC returns
        """
        super().__init__(data, name)
        subset = data.groupby(level="symbol").nth(0)
        if not self.is_singleton:
            if not subset.isna().all(axis=1).all():  # type: ignore
                raise ValidationError(
                    "Each symbols returns are expected to have first row made of NaNs"
                )

    def validate_data(self, data: pd.DataFrame):
        validate_dataframe(data)

    def avg(
        self, weights: Weights, portfolio_name: str = const.DEFAULT_PORTFOLIO_NAME
    ) -> "Returns":
        """
        Compute the average return by date
        """
        weighted = []
        if not len(weights):
            weighted_avg = self._data.groupby(level="date").mean()
            add_singleton_level(weighted_avg, portfolio_name)
            return self.__class__(weighted_avg, name=self.name)

        for symbol, weight in weights.items():
            w = self._data.loc[symbol, :] * weight
            w.name = symbol
            weighted.append(w)

        weighted_data = pd.concat(
            weighted,
            axis=0,
            keys=[df.name for df in weighted],
            names=["symbol", "date"],
        )
        namask = weighted_data.xs(symbol, level="symbol").isna().all(axis=1)
        weighted_avg = weighted_data.groupby(level="date").sum()
        weighted_avg[namask] = np.nan

        add_singleton_level(weighted_avg, portfolio_name)
        return self.__class__(weighted_avg, name=self.name)

    def cumulative(
        self,
        name: str,
        starting_value: float | List[float] = 1.0,
        keepna: bool = True,
        check_base: bool = True,
        periods: List[Period] | None = None,
    ) -> CumulativeReturns:
        """
        Parameters
        ----------
        name: str
            See :py:meth:`portfolio_plan.financialseries.CumulativeReturns`
        starting_value: int | List[int]
            See :py:func:`empyrical.cum_returns`
            If starting_value is a list, it must be of length 4.
            Each element is mapped (in order) to:
            - open
            - close
            - high
            - low

        Warning
        -------
        I recommend setting keepna to True. When False, inconsistencies between
        the frequencies of the scale of the plotted dates and the actual frequency
        of the data could lead to misinterpretation.
        """
        if not isinstance(starting_value, list):
            newreturns: pd.DataFrame = cumulative_from_returns(
                self._data, starting_value, keepna
            )
        else:
            if len(starting_value) != 4:
                raise ValueError("Expected starting_value to be of length 4")
            columns = ("open", "close", "high", "low")
            newreturnss: List[pd.DataFrame] = []
            for i, value in enumerate(starting_value):
                column = columns[i]
                newreturnss.append(
                    cumulative_from_returns(self._data[[column]], value, keepna)
                )
            newreturns = pd.concat(newreturnss, axis=1)

        return CumulativeReturns(
            newreturns, name=name, check_base=check_base, periods=periods
        )

    def plot_low(self, scale_x: bool = True) -> ggplot:
        """
        Plot all returns low price

        Example
        -------

        >>> from portfolio_plan.financialseries import Returns, Frequency
        >>> from portfolio_plan.resource import File
        >>> from portfolio_plan.data import example_prices_path
        >>> from portfolio_plan.utils import returns_from_prices

        >>> resource = File(path=example_prices_path(), frequency=Frequency.BDAILY)
        >>> prices = resource.fetch(
        ...     ["A", "B"],
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31"
        ... )
        >>> returns = Returns(returns_from_prices(prices.data))
        >>> returns.plot_low()

        >>> returns.cumulative().plot_low()
        """
        return plot_returns(self._data["low"], scale_x)

    def plot_high(self, scale_x: bool = True) -> ggplot:
        """
        Plot all returns high price
        """
        return plot_returns(self._data["high"], scale_x)

    def plot_open(self, scale_x: bool = True) -> ggplot:
        """
        Plot all returns open price
        """

        return plot_returns(self._data["open"], scale_x)

    def plot_close(self, scale_x: bool = True) -> ggplot:
        """
        Plot all returns close price
        """
        return plot_returns(self._data["close"], scale_x)

    def plot(self, scale_x: bool = True):
        """
        Plot all returns candle chart
        """
        raise NotImplementedError()

    @property
    def is_unique_symbol(self) -> bool:
        data = self.data
        usymbols = data.index.get_level_values("symbol").unique()
        return usymbols.shape[0] == 1

    @property
    def is_singleton(self):
        return self.data.shape[0] == 1


class Comparison:
    def __init__(self, objs: List[FinancialSeries]):
        """ """
        if not isinstance(objs, list):
            raise TypeError("Expected a list")
        if not objs:
            raise ValueError("At least one element required")
        if not isinstance(objs[0], (FinancialSeries)):
            raise TypeError("Elements must be a concrete type of FinancialSeries")
        for i in range(1, len(objs)):
            current = objs[i]
            previous = objs[i - 1]
            if not isinstance(current, type(previous)):
                raise TypeError("All elements should be of same type")

        self._objs = objs

    def _plot_price_dimension(
        self, column: Literal["open", "low", "high", "close"], scale_x: bool
    ):
        objs = self._objs
        match column:
            case "open":
                previous = objs[0].plot_open(scale_x=scale_x)
            case "close":
                previous = objs[0].plot_close(scale_x=scale_x)
            case "high":
                previous = objs[0].plot_high(scale_x=scale_x)
            case "low":
                previous = objs[0].plot_low(scale_x=scale_x)
            case _:
                raise ValueError("not one of accepted column values")
        if objs[0].name:
            text = _geom_text(data=objs[0].data[column], text=objs[0].name)
            previous += text

        if len(objs) == 1:
            return previous

        current = previous
        for i in range(1, len(objs)):
            current_obj = objs[i]
            if isinstance(current_obj, CumulativeReturns):
                lines = _geom_lines(
                    data=current_obj.data[column], periods=current_obj.periods
                )
            else:
                lines = _geom_lines(data=current_obj.data[column])
            for geom in lines:
                current += geom
            if current_obj.name:
                text = _geom_text(data=current_obj.data[column], text=current_obj.name)
                current += text
        return current

    def plot_open(self, scale_x: bool = True) -> ggplot:
        return self._plot_price_dimension("open", scale_x=scale_x)

    def plot_close(self, scale_x: bool = True) -> ggplot:
        return self._plot_price_dimension("close", scale_x=scale_x)

    def plot_low(self, scale_x: bool = True) -> ggplot:
        return self._plot_price_dimension("low", scale_x=scale_x)

    def plot_high(self, scale_x: bool = True):
        return self._plot_price_dimension("high", scale_x=scale_x)

    def plot(self, scale_x: bool = True):
        raise NotImplementedError()


def vstack(prices_list: List[Prices] | List[Returns]) -> pd.DataFrame:
    dfs = [el.data for el in prices_list]
    return pd.concat(dfs, axis=0)  # rowwise
