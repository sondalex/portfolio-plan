from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from portfolio_plan import const
from portfolio_plan._properties import Frequency, Weights
from portfolio_plan.core import Asset, Assets, Plan, Portfolio
from portfolio_plan.errors import PeriodOverlapError
from portfolio_plan.financialseries import ValidationError
from portfolio_plan.resource import Memory as ArrowMemory
from portfolio_plan.utils import (
    add_singleton_level,
    cumulative_from_returns,
    returns_from_prices,
)


class TestWeight:
    @staticmethod
    def test_valid_weights():
        w = Weights(AAPL=0.6, MSFT=0.4)
        assert w["AAPL"] == 0.6
        assert w["MSFT"] == 0.4

    @staticmethod
    def test_valid_weights_complex_key():
        w = Weights(**{"NESN.SW": 1.0})
        assert w["NESN.SW"] == 1.0

    @staticmethod
    def test_sum_not_one():
        with pytest.raises(ValidationError):
            Weights(AAPL=0.5, MSFT=0.3)

    @staticmethod
    def test_empty_weights():
        # Does not raise error
        Weights()

    @staticmethod
    def test_non_numeric_weights():
        with pytest.raises(ValidationError):
            Weights(AAPL="high", MSFT=0.5)

    @staticmethod
    def test_floating_point_tolerance():
        w = Weights(AAPL=0.3333333, MSFT=0.6666667)
        assert abs(sum(w.values()) - 1.0) < 1e-8

    @staticmethod
    def test_repr_output():
        w = Weights(AAPL=0.5, GOOG=0.5)
        rep = repr(w)
        assert "AAPL" in rep
        assert "GOOG" in rep


def setup_data(
    data: Dict[str, list[float | int]], symbols: List[str], start: str, freq: str = "D"
) -> pd.DataFrame:
    nobs = len(data["open"])
    nentity = len(symbols)

    assert nobs % nentity == 0
    nperiod = nobs // nentity
    date = pd.date_range(start=start, periods=nperiod, freq=freq)
    index = pd.MultiIndex.from_product([symbols, date], names=["symbol", "date"])

    df = pd.DataFrame(data, index=index)
    return df


def setup_plan(data: pd.DataFrame) -> Plan:
    symbols: List[str] = data.index.get_level_values(level="symbol").unique().tolist()
    dates = data.index.get_level_values(level="date")
    stream = BytesIO()
    data.to_parquet(stream, engine="pyarrow")
    plan = Plan(
        portfolio=Portfolio(
            assets=Assets(
                [Asset(symbol, resource=None) for symbol in symbols],
                resource=ArrowMemory(
                    data=stream,
                    frequency=Frequency.DAILY,
                ),
            ),
            weights=Weights(),
        ),
        start_date=dates.min(),
        end_date=dates.max(),
    )
    return plan


@pytest.fixture()
def input_mixed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        setup_data(
            {
                "open": [
                    100.0,
                    102.0,
                    101.0,
                    105.0,
                    104.0,
                    106.0,
                    107.0,
                    110.0,
                    108.0,
                    109.0,
                    111.0,
                    112.0,
                    113.0,
                    115.0,
                ],
                "close": [
                    102.0,
                    101.0,
                    105.0,
                    104.0,
                    106.0,
                    107.0,
                    109.0,
                    112.0,
                    110.0,
                    111.0,
                    114.0,
                    115.0,
                    116.0,
                    118.0,
                ],
                "high": [
                    103.0,
                    103.0,
                    106.0,
                    107.0,
                    107.0,
                    108.0,
                    110.0,
                    113.0,
                    111.0,
                    112.0,
                    115.0,
                    116.0,
                    117.0,
                    119.0,
                ],
                "low": [
                    99.0,
                    100.0,
                    100.0,
                    103.0,
                    103.0,
                    105.0,
                    106.0,
                    109.0,
                    107.0,
                    108.0,
                    110.0,
                    111.0,
                    112.0,
                    114.0,
                ],
            },
            symbols=["A", "B"],
            start="2024-01-01",
        ),
        setup_data(
            {
                "open": [200.0, 202.0, 204.0, 206.0, 208.0, 210.0],
                "close": [202.0, 204.0, 206.0, 208.0, 210.0, 212.0],
                "high": [203.0, 205.0, 207.0, 209.0, 211.0, 213.0],
                "low": [199.0, 201.0, 203.0, 205.0, 207.0, 209.0],
            },
            symbols=["B", "C"],
            start="2024-01-08",
        ),
    )


def _expected_returns(input: Tuple[pd.DataFrame, pd.DataFrame]) -> List[pd.DataFrame]:
    price1, price2 = input
    returns1 = returns_from_prices(price1)
    returns2 = returns_from_prices(price2)

    # Equal weight portfolio
    average_returns1 = returns1.groupby(level="date").mean()
    average_returns2 = returns2.groupby(level="date").mean()
    add_singleton_level(average_returns1, const.DEFAULT_PORTFOLIO_NAME)
    add_singleton_level(average_returns2, const.DEFAULT_PORTFOLIO_NAME)
    return [average_returns1, average_returns2]


def _expected_cumulative_returns(average_returns1, average_returns2) -> pd.DataFrame:
    cumulative_returns1 = cumulative_from_returns(average_returns1, 1, True)
    cumulative_returns2 = []
    for column in ("close", "open", "high", "low"):
        last = cumulative_returns1.loc[const.DEFAULT_PORTFOLIO_NAME][column].iloc[-1]
        assert not pd.isna(last)
        _ = cumulative_from_returns(average_returns2[[column]], last, True)
        cumulative_returns2.append(_)
    cumulative_returns2 = pd.concat(cumulative_returns2, axis=1)  # type: ignore
    exp = pd.concat([cumulative_returns1, cumulative_returns2], axis=0)  # type: ignore

    return exp


@pytest.fixture()
def expected_returns_mixed(
    input_mixed: Tuple[pd.DataFrame, pd.DataFrame],
) -> list[pd.DataFrame]:
    return _expected_returns(input_mixed)


@pytest.fixture()
def expected_cumulative_returns_mixed(
    input_mixed: Tuple[pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    avg1, avg2 = _expected_returns(input_mixed)
    return _expected_cumulative_returns(avg1, avg2)


@pytest.fixture
def input_distinct() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        setup_data(
            {
                "open": [100.0, 101.0, 102.0],
                "close": [101.0, 103.0, 104.0],
                "high": [102.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0],
            },
            symbols=["A"],
            start="2024-01-01",
        ),
        setup_data(
            {
                "open": [200.0, 205.0, 208.0],
                "close": [205.0, 210.0, 212.0],
                "high": [206.0, 211.0, 213.0],
                "low": [198.0, 204.0, 206.0],
            },
            symbols=["B"],
            start="2024-01-04",
        ),
    )


@pytest.fixture
def expected_returns_distinct(input_distinct) -> list[pd.DataFrame]:
    return _expected_returns(input_distinct)


@pytest.fixture
def expected_cumulative_returns_distinct(input_distinct) -> pd.DataFrame:
    avg1, avg2 = _expected_returns(input_distinct)
    return _expected_cumulative_returns(avg1, avg2)


@pytest.fixture()
def input_overlapping_period() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        setup_data(
            {
                "open": [100.0, 102.0],
                "close": [102.0, 104.0],
                "high": [103.0, 105.0],
                "low": [99.0, 101.0],
            },
            symbols=["A"],
            start="2024-01-01",
        ),
        setup_data(
            {
                "open": [105.0, 106.0],
                "close": [106.0, 108.0],
                "high": [107.0, 109.0],
                "low": [104.0, 105.0],
            },
            symbols=["A"],
            start="2024-01-02",
        ),
    )


class TestPlan:
    def test_join_returns_mixed(
        self, input_mixed: Tuple[pd.DataFrame, pd.DataFrame], expected_returns_mixed
    ):
        price1, price2 = input_mixed

        plan1 = setup_plan(price1)
        plan2 = setup_plan(price2)

        joined = plan1 >> plan2
        result = joined.portfolio_returns()
        assert len(result) == len(expected_returns_mixed)
        for a, b in zip(result, expected_returns_mixed):
            pd.testing.assert_frame_equal(a.data, b)

    def test_join_cumulative_returns_mixed(
        self,
        input_mixed: Tuple[pd.DataFrame, pd.DataFrame],
        expected_cumulative_returns_mixed,
    ):
        price1, price2 = input_mixed

        plan1 = setup_plan(price1)
        plan2 = setup_plan(price2)

        joined = plan1 >> plan2
        result = joined.portfolio_cumulative_returns("_")
        pd.testing.assert_frame_equal(result.data, expected_cumulative_returns_mixed)

    def test_join_returns_distinct(
        self,
        input_distinct: Tuple[pd.DataFrame, pd.DataFrame],
        expected_returns_distinct: list[pd.DataFrame],
    ):
        price1, price2 = input_distinct

        plan1 = setup_plan(price1)
        plan2 = setup_plan(price2)

        joined = plan1 >> plan2

        result = joined.portfolio_returns()
        assert len(result) == len(expected_returns_distinct)
        for a, b in zip(result, expected_returns_distinct):
            pd.testing.assert_frame_equal(a.data, b)

    def test_join_cumulative_returns_distinct(
        self,
        input_distinct: Tuple[pd.DataFrame, pd.DataFrame],
        expected_cumulative_returns_distinct,
    ):
        price1, price2 = input_distinct

        plan1 = setup_plan(price1)
        plan2 = setup_plan(price2)

        joined = plan1 >> plan2

        result = joined.portfolio_cumulative_returns("_")
        pd.testing.assert_frame_equal(result.data, expected_cumulative_returns_distinct)

    def test_overlapping_period(
        self, input_overlapping_period: Tuple[pd.DataFrame, pd.DataFrame]
    ):
        price1, price2 = input_overlapping_period

        plan1 = setup_plan(price1)
        plan2 = setup_plan(price2)

        with pytest.raises(PeriodOverlapError):
            _ = plan1 >> plan2

    def test_portfolio_returns(self):
        """
        Portfolio return across time
        """
        values = [
            100.0,  # A, 2020-01-01
            110.0,  # A, 2020-01-02
            115.0,  # A, 2020-01-03
            120.0,  # A, 2020-01-04
            200.0,  # B, 2020-01-01
            210.0,  # B, 2020-01-02
            220.0,  # B, 2020-01-03
            225.0,  # B, 2020-01-04
        ]

        index_tuples = [
            ("A", pd.Timestamp("2020-01-01")),
            ("A", pd.Timestamp("2020-01-02")),
            ("A", pd.Timestamp("2020-01-03")),
            ("A", pd.Timestamp("2020-01-04")),
            ("B", pd.Timestamp("2020-01-01")),
            ("B", pd.Timestamp("2020-01-02")),
            ("B", pd.Timestamp("2020-01-03")),
            ("B", pd.Timestamp("2020-01-04")),
        ]

        data = pd.DataFrame(
            {
                "open": values,
                "close": values,
                "high": values,
                "low": values,
            },
            index=pd.MultiIndex.from_tuples(
                index_tuples,
                names=["symbol", "date"],
            ),
        )

        plan = setup_plan(data)
        result = plan.portfolio_returns()

        expected_values = [
            np.nan,  # 2020-01-01
            (((110.0 - 100.0) / 100.0) + ((210.0 - 200.0) / 200.0)) / 2,  # 2020-01-02
            (((115.0 - 110.0) / 110.0) + ((220.0 - 210.0) / 210.0)) / 2,  # 2020-01-03
            (((120.0 - 115.0) / 115.0) + ((225.0 - 220.0) / 220.0)) / 2,  # 2020-01-04
        ]

        expected = pd.DataFrame(
            {
                "open": expected_values,
                "close": expected_values,
                "high": expected_values,
                "low": expected_values,
            },
            index=pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
        )
        add_singleton_level(expected, "Portfolio")
        expected.columns.name = "Returns"

        pd.testing.assert_frame_equal(result.data, expected)
