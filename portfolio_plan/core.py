""" """

from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import List, Literal

import pandas as pd
from plotnine import ggplot

from portfolio_plan import const, resource
from portfolio_plan._properties import Frequency, Weights
from portfolio_plan.errors import PeriodOverlapError
from portfolio_plan.financialseries import CumulativeReturns, Prices, Returns, vstack
from portfolio_plan.prettyprint import ascii_graph
from portfolio_plan.ratios import InformationRatio
from portfolio_plan.utils import (
    Period,
    _create_str_stack,
    plot_allocation,
    plot_allocations,
    plot_allocations2,
    returns_from_prices,
)
from portfolio_plan.visualisation import THEME_ROSE_PINE_BASE_SIZE


class Asset:
    def __init__(self, symbol: str, resource: resource.Resource | None = None):
        """
        Example
        -------

        >>> from portfolio_plan import resource
        >>> asset1 = Asset("A", resource = resource.File("prices.parquet"))
        >>> asset2 = Asset("B", resource = resource.YFinance())
        """
        self._symbol = symbol
        self._resource = resource

    def __str__(self):
        return f"{self.__class__.__name__}(symbol = {self._symbol}, resource={str(self._resource)})"

    def fetch(self, start_date: datetime, end_date: datetime, **kwargs) -> Prices:
        if self._resource is not None:
            prices = self._resource.fetch(
                [self._symbol], start_date, end_date, **kwargs
            )
            return prices
        raise RuntimeError(
            "Can not fetch if resource was not set during class instantiation"
        )

    @property
    def symbol(self):
        return self._symbol

    @property
    def resource(self):
        return self._resource


class Assets:
    def __init__(self, assets: List[Asset], resource: resource.Resource):
        if any(map(lambda asset: not isinstance(asset, Asset), assets)):
            raise ValueError("Expected assets to be a list of Asset")
        self._assets = assets
        self._resource = resource

    def fetch(
        self, start_date: datetime, end_date: datetime, name: str, **kwargs
    ) -> Prices:
        """
        Parameters
        ----------
        name: str
            Passed to :py:meth:`portfolio_plan.financialseries.Prices`
        """
        symbols = []
        prices = []
        for asset in self._assets:
            if asset.resource is None:
                symbols.append(asset.symbol)
            else:
                price = asset.fetch(start_date=start_date, end_date=end_date, **kwargs)
                prices.append(price)
        if symbols:
            price = self._resource.fetch(
                symbols, start_date=start_date, end_date=end_date, name=name, **kwargs
            )
            prices.append(price)
        prices_data = vstack(prices)
        return Prices(prices_data, name)

    @property
    def symbols(self) -> List[str]:
        return [asset._symbol for asset in self._assets]


class Optimizer(Enum):
    MEANVARIANCE = "MeanVariance"


class Portfolio:
    """
    A portfolio consists of assets and its associated weights
    """

    def __init__(
        self,
        assets: Assets,
        weights: Weights | None = None,
        name: str = const.DEFAULT_PORTFOLIO_NAME,
        optimizer: Optimizer = Optimizer.MEANVARIANCE,
        **optimizer_kwargs,
    ):
        """
        Parameters
        ----------
        assets: Assets
        weights: Weights | None
            Defaults to None
        name: str
            Defaults to :py:const:`portfolio_plan.const.DEFAULT_PORTFOLIO_NAME`
        optimizer: Optimizer
            Defaults to MeanVariance optimizer

        Warning
        =======
        For the moment optimizers are ignored (not implemented).

        """
        self._assets = assets
        self._weights: Weights | None = weights
        self._optimizer_kwargs = optimizer_kwargs
        self._name = name

    def fetch(
        self, start_date: datetime, end_date: datetime, name: str, **kwargs
    ) -> Prices:
        """
        Parameters
        ----------
        name: str
            Passed to :py:meth:`portfolio_plan.financialseries.Prices`
        """
        return self._assets.fetch(
            start_date=start_date, end_date=end_date, name=name, **kwargs
        )

    def _compute(self):
        """
        Compute optimization problem for optimal weights
        """
        ...

    @property
    def name(self) -> str:
        return self._name

    def weights(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> Weights:
        if self._weights is not None:
            return self._weights
        else:
            self._compute()
            raise NotImplementedError()

    @property
    def symbols(self) -> List[str]:
        symbs = self._assets.symbols
        return symbs

    def plot_allocation(
        self,
        variant: Literal["main", "moon", "dawn"] = "dawn",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        base_size: int = THEME_ROSE_PINE_BASE_SIZE,
    ) -> ggplot:
        weights: Weights = self.weights(start_date=start_date, end_date=end_date)
        if weights.empty:
            symbols = self._assets.symbols
            n_symbols = len(symbols)
            weight = 1 / n_symbols
            weights = Weights(**{symbol: weight for symbol in symbols})
        return plot_allocation(weights, variant=variant, base_size=base_size)


class Plan:
    def __init__(self, portfolio: Portfolio, start_date: datetime, end_date: datetime):
        """
        Parameters
        ----------
        portfolio: Portfolio
            See :py:class:`Portfolio`
        start_date: datetime
            Start date of the plan
        end_date: datetime
            End date of the plan
        """

        self._portfolio = portfolio
        self._start_date = start_date
        self._end_date = end_date
        self._frequency: Frequency | None = None

    def __str__(self):
        name = self.__class__.__name__
        s = _create_str_stack(
            f"{name} for period {self._start_date}, {self._end_date}",
            self._portfolio.symbols,
        )
        return s

    def __repr__(self):
        return self.__str__()

    @lru_cache
    def prices(self) -> Prices:
        p = self._portfolio.fetch(
            self._start_date, self._end_date, name=self._portfolio.name
        )
        symbols = p.data.index.get_level_values("symbol").unique()
        frequencies = [pd.infer_freq(p.data.loc[symbol].index) for symbol in symbols]
        # assert frequency is the same across all symbols
        assert all(
            frequencies[i] == frequencies[i - 1] for i in range(1, len(frequencies))
        )

        freq = frequencies[0]
        self.frequency = freq  # type: ignore
        return p

    @property
    def frequency(self) -> Frequency | None:
        return self._frequency

    @frequency.setter
    def frequency(self, value: str | Frequency | None):
        if isinstance(value, Frequency):
            self._frequency = value
        elif isinstance(value, str):
            self._frequency = Frequency.from_pandas(value)
        elif value is None:
            self._frequency = None
        else:
            raise TypeError("Unsupported type")

    @lru_cache
    def returns(self, name: str) -> Returns:
        prices = self.prices()
        returns = returns_from_prices(prices.data)

        return Returns(returns, name=self._portfolio.name)

    def portfolio_returns(self) -> Returns:
        """
        Portfolio return across time
        """
        returns = self.returns(name=self._portfolio.name)
        portfolio_name = self._portfolio.name
        return returns.avg(
            self._portfolio.weights(
                start_date=self._start_date, end_date=self._end_date
            ),
            portfolio_name=portfolio_name,
        )

    @property
    def weights(self) -> Weights:
        return self._portfolio.weights(
            start_date=self._start_date, end_date=self._end_date
        )

    @property
    def start_date(self) -> datetime:
        """
        The plan start date, which is not necessarily the effective start date.
        The effective start date is the first date with available data, and it satisfies:
        effective start date >= start_date <= effective end date, where effective end date <= end date.
        """
        return self._start_date

    @property
    def end_date(self) -> datetime:
        """
        The plan end date, which is the final date of the plan period.
        """
        return self._end_date

    @property
    def portfolio(self):
        return self._portfolio

    def __rshift__(self, right):
        if not isinstance(right, self.__class__):
            type1 = type(self.right)
            type2 = self.__class__
            raise TypeError(f"unsupported operand type(s) for >>: {type1} and {type2}")

        return JoinedPlan([self, right])


def _validate_plans(plans: List[Plan]):
    if not plans:
        return
    for i in range(1, len(plans)):
        current: Plan = plans[i]
        previous: Plan = plans[i - 1]
        if current._start_date <= previous._end_date:
            raise PeriodOverlapError(
                "Plans can not overlap in time period and must be ordered by time period"
            )


class JoinedPlan:
    def __init__(self, plans=List[Plan]) -> None:
        if not plans:
            raise ValueError("At least one plan is required")
        self._plans: List[Plan] = plans
        _validate_plans(plans)

    def __str__(self):
        name = self.__class__.__name__
        return ascii_graph(name, self._plans)

    def __repr__(self):
        return self.__str__()

    def portfolio_returns(self) -> List[Returns]:
        """
        Returns each plan portfolio returns

        Returns
        -------
        A list of returns of length equal to the number of plans in the joined plan
        """
        returnss = []
        for plan in self._plans:
            returnss.append(plan.portfolio_returns())
        return returnss

    def _get_portfolio_name(self, i: int) -> str:
        return self._plans[i]._portfolio.name

    def plot_allocations(
        self,
        variant: Literal["main", "moon", "dawn"] = "dawn",
        base_size: int = THEME_ROSE_PINE_BASE_SIZE,
    ) -> ggplot:
        weights: List[Weights] = []
        for plan in self._plans:
            portfolio = plan._portfolio
            w: Weights = portfolio.weights(
                start_date=plan.start_date, end_date=plan.end_date
            )
            if w.empty:
                symbols = portfolio._assets.symbols
                n_symbols = len(symbols)
                weight = 1 / n_symbols
                w = Weights(**{symbol: weight for symbol in symbols})
            weights.append(w)
        return plot_allocations(
            weights, periods=self.periods, variant=variant, base_size=base_size
        )

    def plot_allocations2(
        self,
        variant: Literal["main", "moon", "dawn"] = "dawn",
        base_size: int = THEME_ROSE_PINE_BASE_SIZE,
        text_size: float = 0.8 * THEME_ROSE_PINE_BASE_SIZE,
        text_angle: float = 45,
    ) -> ggplot:
        weights: List[Weights] = []
        for plan in self._plans:
            portfolio = plan._portfolio
            w: Weights = portfolio.weights(
                start_date=plan.start_date, end_date=plan.end_date
            )
            if w.empty:
                symbols = portfolio._assets.symbols
                n_symbols = len(symbols)
                weight = 1 / n_symbols
                w = Weights(**{symbol: weight for symbol in symbols})
            weights.append(w)
        return plot_allocations2(
            weights,
            periods=self.periods,
            variant=variant,
            base_size=base_size,
            text_size=text_size,
            text_angle=text_angle,
        )

    def portfolio_cumulative_returns(
        self, name: str, starting_value: int = 1, keepna: bool = True
    ) -> CumulativeReturns:
        """ """
        plans_portfolio_returns = self.portfolio_returns()
        cumulatives: List[pd.DataFrame] = []
        for i in range(len(plans_portfolio_returns)):
            plan_portfolio_return = plans_portfolio_returns[i]
            portfolio_name = self._get_portfolio_name(i)

            if i == 0:
                cumulative = plan_portfolio_return.cumulative(
                    starting_value=starting_value, name=portfolio_name, keepna=keepna
                ).data
            else:
                previous_cumulative = cumulatives[i - 1]
                new_starting_value: List[float] = previous_cumulative.iloc[
                    -1, :
                ].tolist()
                cumulative = plan_portfolio_return.cumulative(
                    starting_value=new_starting_value,
                    name=plan_portfolio_return.name,
                    keepna=keepna,
                    check_base=False,
                ).data
            index: pd.MultiIndex = cumulative.index  # type: ignore
            newindex = index.set_levels([portfolio_name], level=0)
            cumulative.index = newindex
            cumulatives.append(cumulative)
        cumulatives_data = pd.concat(cumulatives, axis=0)
        return CumulativeReturns(
            cumulatives_data,
            name=name,
            check_base=False,
            periods=self.periods,
            check_frequency=False,
        )

    @property
    def periods(self) -> List[Period]:
        return [
            Period(start=plan.start_date, end=plan.end_date, frequency=plan.frequency)
            for plan in self._plans
        ]

    def information_ratio_returns(self, b_joined_plan: "JoinedPlan") -> Returns:
        """
        Compute information ratio for each of the plans

        Parameters
        ----------
        b_joined_plan: JoinedPlan
            Another joined plan

        Returns
        -------
        A single DataFrame with IR for each Period
        """
        # TODO: How to deal with the Organization
        irs = []
        for plan, bplan in zip(self._plans, b_joined_plan._plans):
            ir = InformationRatio(plan, bplan)
            irs.append(ir.period_returns(keep_index=True).data)
        return Returns(pd.concat(irs, axis=0), "IRS")

    def information_ratio_returns_cumulative(
        self, b_joined_plan: "JoinedPlan", starting_value: float = 1.0
    ) -> CumulativeReturns:
        plans = self._plans
        bplans = b_joined_plan._plans

        cumulatives: List[pd.DataFrame] = []

        for i in range(len(plans)):
            plan = plans[i]
            bplan = bplans[i]
            ir = InformationRatio(plan, bplan)
            irreturns = ir.period_returns(keep_index=True)

            if i == 0:
                cumulative = irreturns.cumulative(
                    "_", starting_value=starting_value
                ).data
            else:
                previous_cumulative = cumulatives[i - 1]
                new_starting_value = previous_cumulative.iloc[-1, :].tolist()
                cumulative = irreturns.cumulative(
                    "", starting_value=new_starting_value, check_base=False
                ).data

            cumulatives.append(cumulative)
        cumulatives_data = pd.concat(cumulatives, axis=0)
        mask = cumulatives_data.isna().all(axis=1)
        # TODO: Transform this in two straight lines instead --> Clearly show it's
        # a ratio. Not a time series.
        return CumulativeReturns(
            cumulatives_data[~mask],
            name="IR Cumulative",
            check_base=False,
            check_frequency=False,
        )
