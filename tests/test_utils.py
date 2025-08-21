from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas import MultiIndex

from portfolio_plan._properties import Frequency, Weights
from portfolio_plan.data import (
    mock_cumulative_returns_path,
    mock_prices_path,
    mock_returns_path,
)
from portfolio_plan.utils import (
    Period,
    _find_gap_lines,
    cumulative_from_returns,
    plot_allocations,
    returns_from_prices,
    validate_dataframe,
)


class TestValidateDataFrame:
    def test_valid_dataframe(self):
        aapl_dates = pd.date_range("2023-01-01", periods=3, freq="D")
        msft_dates = pd.date_range("2023-01-10", periods=4, freq="D")
        index = MultiIndex.from_tuples(
            [("AAPL", d) for d in aapl_dates] + [("MSFT", d) for d in msft_dates],
            names=["symbol", "date"],
        )
        df = pd.DataFrame(
            {
                "open": [150.0, 151.0, 152.0, 210.0, 211.0, 212.0, 213.0],
                "close": [155.0, 156.0, 157.0, 215.0, 216.0, 217.0, 218.0],
                "high": [156.0, 157.0, 158.0, 216.0, 217.0, 218.0, 219.0],
                "low": [149.0, 150.0, 151.0, 209.0, 210.0, 211.0, 212.0],
            },
            index=index,
        )
        validate_dataframe(df)

    def test_not_dataframe(self):
        with pytest.raises(AssertionError, match="Data must be a pandas DataFrame"):
            validate_dataframe("not a dataframe")

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(AssertionError, match="DataFrame should not be empty"):
            validate_dataframe(df)

    def test_wrong_columns(self):
        index = MultiIndex.from_tuples(
            [("AAPL", "2023-01-01")], names=["symbol", "date"]
        )
        df = pd.DataFrame({"price": [150]}, index=index)
        with pytest.raises(AssertionError, match="DataFrame must have exactly columns"):
            validate_dataframe(df)

    def test_non_multiindex(self):
        df = pd.DataFrame(
            {
                "open": [150.0],
                "close": [155.0],
                "high": [156.0],
                "low": [149.0],
            },
            index=[0],
        )
        with pytest.raises(AssertionError, match="Index must be a MultiIndex"):
            validate_dataframe(df)

    def test_wrong_index_names(self):
        index = MultiIndex.from_tuples(
            [("AAPL", "2023-01-01")], names=["ticker", "time"]
        )
        df = pd.DataFrame(
            {
                "open": [150.0],
                "close": [155.0],
                "high": [156.0],
                "low": [149.0],
            },
            index=index,
        )
        with pytest.raises(AssertionError, match="Index names must be"):
            validate_dataframe(df)

    def test_partial_nan_row(self):
        index = MultiIndex.from_tuples(
            [("AAPL", "2023-01-01")], names=["symbol", "date"]
        )
        df = pd.DataFrame(
            {
                "open": [150.0],
                "close": [None],
                "high": [156.0],
                "low": [149.0],
            },
            index=index,
        )
        with pytest.raises(
            AssertionError, match="Invalid NaN pattern found for symbol"
        ):
            validate_dataframe(df)

    def test_unsorted_dates(self):
        index = MultiIndex.from_tuples(
            [("AAPL", "2023-01-02"), ("AAPL", "2023-01-01")], names=["symbol", "date"]
        )
        df = pd.DataFrame(
            {
                "open": [152.0, 150.0],
                "close": [153.0, 155.0],
                "high": [154.0, 156.0],
                "low": [151.0, 149.0],
            },
            index=index,
        )
        with pytest.raises(AssertionError, match="Dates must be sorted for symbol"):
            validate_dataframe(df)

    def test_missing_freq(self):
        index = MultiIndex.from_tuples(
            [
                ("AAPL", pd.Timestamp("2023-01-01")),
                ("AAPL", pd.Timestamp("2023-01-03")),
                ("AAPL", pd.Timestamp("2023-01-06")),
                ("MSFT", pd.Timestamp("2023-01-02")),
                ("MSFT", pd.Timestamp("2023-01-04")),
                ("MSFT", pd.Timestamp("2023-01-08")),
            ],
            names=["symbol", "date"],
        )
        df = pd.DataFrame(
            {
                "open": [150.0, 152.0, 153.0, 210.0, 211.0, 212.0],
                "close": [155.0, 153.0, 154.0, 215.0, 214.0, 216.0],
                "high": [156.0, 154.0, 155.0, 216.0, 215.0, 217.0],
                "low": [149.0, 151.0, 152.0, 209.0, 211.0, 212.0],
            },
            index=index,
        )
        with pytest.raises(
            AssertionError,
            match="Date index must have an inferrable frequency for current symbol. Found freq=None",
        ):
            validate_dataframe(df)


def test__find_gap_lines_nan_blocks():
    """Test gap lines with NaN blocks in different positions."""
    dates = pd.date_range("2025-01-01", "2025-01-10", freq="D")

    values1 = [np.nan, np.nan, 100.0, 110.0, np.nan, 120, 130, np.nan, np.nan, 140]

    values2 = [
        100.0,
        110.0,
        np.nan,
        120.0,
        130.0,
        np.nan,
        140.0,
        np.nan,
        np.nan,
        np.nan,
    ]

    values3 = [100.0, 110.0, np.nan, np.nan, np.nan, 120.0, 130.0, 140.0, 150.0, 160.0]

    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], dates], names=["symbol", "date"]
    )

    test_series = pd.Series(values1 + values2 + values3, index=index, name="close")

    result = _find_gap_lines(test_series)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"symbol", "date", "close", "group"}

    result_tuples = list(
        zip(result["symbol"], result["date"].dt.strftime("%Y-%m-%d"), result["close"])
    )

    # Expected connections
    expected_tuples = [
        ("A", "2025-01-04", 110.0),
        ("A", "2025-01-06", 120.0),
        ("A", "2025-01-07", 130.0),
        ("A", "2025-01-10", 140.0),
        ("B", "2025-01-02", 110.0),
        ("B", "2025-01-04", 120.0),
        ("B", "2025-01-05", 130.0),
        ("B", "2025-01-07", 140.0),
        ("C", "2025-01-02", 110.0),
        ("C", "2025-01-06", 120.0),
    ]

    assert result_tuples == expected_tuples


def test_empty_series():
    """Test behavior with empty series."""
    empty_series = pd.Series([], name="close")
    with pytest.raises(
        ValueError, match="Empty Series, cannot find gaps in empty data."
    ):
        _find_gap_lines(empty_series)


def test_no_gaps():
    """Test behavior with series containing no gaps."""
    # Create series with no gaps
    dates = pd.date_range("2025-01-01", "2025-01-05", freq="D")
    values = [100.0, 110.0, 120.0, 130.0, 140.0]
    index = pd.MultiIndex.from_product([["A"], dates], names=["symbol", "date"])
    series = pd.Series(values, index=index, name="close")

    result = _find_gap_lines(series)
    assert result.empty


def test_middle_nan_block():
    """Test specific case with NaN block only in the middle."""
    dates = pd.date_range("2025-01-01", "2025-01-07", freq="D")
    values = [100.0, 110.0, np.nan, np.nan, np.nan, 120.0, 130.0]

    index = pd.MultiIndex.from_product([["TSLA"], dates], names=["symbol", "date"])

    series = pd.Series(values, index=index, name="close")
    result = _find_gap_lines(series)

    result_tuples = set(
        zip(result["symbol"], result["date"].dt.strftime("%Y-%m-%d"), result["close"])
    )

    expected_tuples = {
        ("TSLA", "2025-01-02", 110.0),
        ("TSLA", "2025-01-06", 120.0),
    }

    assert result_tuples == expected_tuples

    assert len(result["group"].unique()) == 1


def test_multiple_symbols_same_dates():
    """Test handling of multiple symbols with gaps on same dates."""
    dates = pd.date_range("2025-01-01", "2025-01-05", freq="D")
    values = [100.0, np.nan, np.nan, 120.0, 130.0]

    index = pd.MultiIndex.from_product([["A", "B"], dates], names=["symbol", "date"])

    series = pd.Series(values * 2, index=index, name="close")
    result = _find_gap_lines(series)

    assert len(result["group"].unique()) == 2

    # Each symbol should have one connection
    assert len(result[result["symbol"] == "A"]) == 2
    assert len(result[result["symbol"] == "B"]) == 2


def test_plot_allocations():
    with pytest.raises(ValueError, match="Weights can not be empty"):
        plot_allocations(Weights())

    with pytest.raises(ValueError, match="Weights can not be empty"):
        plot_allocations(
            [Weights(), Weights()],
            periods=[
                Period(
                    start=datetime(2020, 1, 1),
                    end=datetime(2021, 1, 1),
                    frequency=Frequency.DAILY,
                ),
                Period(
                    start=datetime(2019, 1, 1),
                    end=datetime(2019, 2, 1),
                    frequency=Frequency.DAILY,
                ),
            ],
        )


@pytest.fixture()
def price_data() -> pd.DataFrame:
    df = pd.read_parquet(mock_prices_path())
    df = df.set_index(["symbol", "date"])
    return df


@pytest.fixture()
def returns_data() -> pd.DataFrame:
    df = pd.read_parquet(mock_returns_path())
    df = df.set_index(["symbol", "date"])
    df.columns.name = "Returns"
    return df.sort_index()


def test_returns_from_prices(price_data, returns_data):
    returns = returns_from_prices(price_data).sort_index()
    pd.testing.assert_frame_equal(
        returns,
        returns_data,
        check_exact=False,
        rtol=1e-3,
        atol=1e-8,
    )


@pytest.fixture()
def cumulative_returns_data() -> pd.DataFrame:
    df = pd.read_parquet(mock_cumulative_returns_path())
    df = df.set_index(["symbol", "date"])
    df.columns.name = "CumulativeReturns"
    return df


class TestCumulativeFromReturns:
    """mock data"""

    data: pd.DataFrame

    @classmethod
    def setup_class(cls):
        data = pd.read_parquet(mock_returns_path())
        data = data.set_index(["symbol", "date"])
        cls.data = data

    def test_full_returns(self, cumulative_returns_data):
        result = cumulative_from_returns(self.data, keepna=True)
        pd.testing.assert_frame_equal(
            result, cumulative_returns_data, check_exact=False, rtol=1e-3, atol=1e-3
        )

    def test_full_returns_nokeepna(self, cumulative_returns_data):
        result = cumulative_from_returns(self.data, keepna=False)
        pd.testing.assert_frame_equal(result, cumulative_returns_data.ffill())

    def test_singleton_returns(self):
        self.data

    def test_cumulative_from_returns_with_nan_handling_and_starting_value(self):
        portfolio_returns = pd.DataFrame(
            {"close": [np.nan, 0.1, 0.05]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )

        result_default = cumulative_from_returns(portfolio_returns)

        expected_default = pd.DataFrame(
            {"close": [1.0, 1.0 * (1.0 + 0.1), 1.0 * (1.0 + 0.1) * (1.0 + 0.05)]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )
        expected_default.columns.name = "CumulativeReturns"

        pd.testing.assert_frame_equal(result_default, expected_default)

        result_zero = cumulative_from_returns(portfolio_returns, starting_value=0.0)

        expected_zero = pd.DataFrame(
            {
                "close": [
                    0.0,
                    ((1.0) * (1.0 + 0.1)) - 1,
                    ((1.0) * (1.0 + 0.1) * (1.0 + 0.05)) - 1.0,
                ]
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )
        expected_zero.columns.name = "CumulativeReturns"

        pd.testing.assert_frame_equal(result_zero, expected_zero)

    def test_cumulative_from_returns_with_multiple_symbols(self):
        """Test that cumulative_from_returns works correctly with multiple symbols."""

        portfolio_returns = pd.DataFrame(
            {"close": [np.nan, 0.1, 0.05, np.nan, 0.2, -0.05]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                    ("B", pd.Timestamp("2020-01-01")),
                    ("B", pd.Timestamp("2020-01-02")),
                    ("B", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )

        result = cumulative_from_returns(portfolio_returns)

        expected = pd.DataFrame(
            {
                "close": [
                    1.0,
                    1.0 * 1.1,
                    1.0 * 1.1 * 1.05,  # A values
                    1.0,
                    1.0 * 1.2,
                    1.0 * 1.2 * 0.95,  # B values
                ]
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                    ("B", pd.Timestamp("2020-01-01")),
                    ("B", pd.Timestamp("2020-01-02")),
                    ("B", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )
        expected.columns.name = "CumulativeReturns"

        pd.testing.assert_frame_equal(result, expected)

    def test_cumulative_from_returns_without_keepna(self):
        """Test that cumulative_from_returns works correctly when keepna=False."""

        portfolio_returns = pd.DataFrame(
            {"close": [np.nan, 0.1, 0.05]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )

        result = cumulative_from_returns(portfolio_returns, keepna=False)
        print(result)

        expected = pd.DataFrame(
            {"close": [1.0, (1.0 + 0.1), (1.0 + 0.1) * (1.0 + 0.05)]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("A", pd.Timestamp("2020-01-01")),
                    ("A", pd.Timestamp("2020-01-02")),
                    ("A", pd.Timestamp("2020-01-03")),
                ],
                names=["symbol", "date"],
            ),
        )
        expected.columns.name = "CumulativeReturns"

        # Check that result matches expected
        pd.testing.assert_frame_equal(result, expected)
