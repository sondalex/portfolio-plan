from abc import ABC, abstractmethod

import pandas as pd
from plotnine.ggplot import ggplot

from portfolio_plan.utils import _create_str_stack


class FinancialSeries(ABC):
    def __init__(self, data: pd.DataFrame, name: str) -> None:
        self.validate_data(data)
        self._data = data
        self._name = name

    @abstractmethod
    def validate_data(self, data: pd.DataFrame):
        pass

    def __str__(self) -> str:
        dates = self._data.index.get_level_values("date")
        symbols = self._data.index.get_level_values("symbol")
        min_date = dates.min()
        max_date = dates.max()
        name = self.__class__.__name__
        stack: str = _create_str_stack(
            f"{name} for period {min_date} to {max_date}", symbols.unique().tolist()
        )
        return stack

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def plot_open(self, scale_x: bool = True) -> ggplot:
        """
        Plot open price
        """
        pass

    @abstractmethod
    def plot_close(self, scale_x: bool = True) -> ggplot:
        """
        Plot close price
        """
        pass

    @abstractmethod
    def plot_low(self, scale_x: bool = True) -> ggplot:
        """
        Plot low price
        """
        pass

    @abstractmethod
    def plot_high(self, scale_x: bool = True) -> ggplot:
        """
        Plot high price
        """
        pass

    @abstractmethod
    def plot(self, scale_x: bool = True) -> ggplot:
        """
        Plot OHLC price
        """
        pass

    @property
    def data(self) -> pd.DataFrame:
        return self._data
