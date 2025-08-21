import numpy as np
import pandas as pd
import pytest

from portfolio_plan._properties import Frequency
from portfolio_plan.financialseries import Prices, Returns, ValidationError


class TestPrices:
    @classmethod
    def setup_class(cls):
        dates = pd.date_range(start="2024-01-01", periods=7, freq="D")
        symbols = ["A"]
        idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])

        cls.data = pd.DataFrame(
            {
                "open": [100, 102, 101, 105, 104, 106, 107],
                "close": [102, 101, 105, 104, 106, 107, 109],
                "high": [103, 103, 106, 107, 107, 108, 110],
                "low": [99, 100, 100, 103, 103, 105, 106],
            },
            index=idx,
        )

        cls.instance = Prices(data=cls.data, name="_")

    def test_resample_daily(self):
        frequency = Frequency.DAILY
        resampled = self.instance.resample_period(frequency)

        pd.testing.assert_frame_equal(resampled._data, self.data)

    def test_resample_weekly(self):
        frequency = Frequency.WEEKLY
        resampled = self.instance.resample_period(frequency)

        expected_idx = pd.MultiIndex.from_tuples(
            [
                ("A", pd.Timestamp("2024-01-07")),
            ],
            names=["symbol", "date"],
        )

        expected = pd.DataFrame(
            {
                "open": [100],
                "close": [109],
                "high": [110],
                "low": [99],
            },
            index=expected_idx,
        )

        pd.testing.assert_frame_equal(resampled._data, expected)

    def test_resample_minute_raises(self):
        frequency = Frequency.MINUTE
        with pytest.raises(ValueError, match="Sampling from D to 1T is not allowed."):
            self.instance.resample_period(frequency)


class TestReturns:
    @classmethod
    def setup_class(cls):
        dates = pd.date_range(start="2024-01-01", periods=4, freq="D")
        symbols = ["A", "B"]
        idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])

        cls.data = pd.DataFrame(
            {
                "open": [np.nan, 0.2, 0.3, 0.2, np.nan, 0.1, 0.2, 0.1],
                "close": [np.nan, 0.2, 0.1, 0.3, np.nan, 0.5, 0.1, 0.2],
                "high": [np.nan, 0.1, 0.1, 0.3, np.nan, 0.5, 0.1, 0.2],
                "low": [np.nan, 0.2, 0.1, 0.3, np.nan, 0.5, 0.1, 0.4],
            },
            index=idx,
        )

        cls.instance = Returns(data=cls.data, name="_")

    @staticmethod
    def test_instantiation():
        dates = pd.date_range(start="2024-01-01", periods=3, freq="D")
        symbols = ["A", "B"]
        idx = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])

        data = pd.DataFrame(
            {
                "open": [0.2, 0.3, 0.2, 0.1, 0.2, 0.1],
                "close": [0.2, 0.1, 0.3, 0.5, 0.1, 0.2],
                "high": [0.1, 0.1, 0.3, 0.5, 0.1, 0.2],
                "low": [0.2, 0.1, 0.3, 0.5, 0.1, 0.4],
            },
            index=idx,
        )
        with pytest.raises(
            ValidationError,
        ):
            Returns(data=data, name="_")

    def test_cumulative(self):
        result = self.instance.cumulative("_").data.sort_index(level=0)

        expected_index = pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range("2024-01-01", periods=4)],
            names=["symbol", "date"],
        )

        expected = pd.DataFrame(
            {
                "open": [
                    1,
                    1.2,
                    1.2 * 1.3,
                    1.2 * 1.3 * 1.2,
                    1,
                    1.1,
                    1.1 * 1.2,
                    1.1 * 1.2 * 1.1,
                ],
                "close": [
                    1,
                    1.2,
                    1.2 * 1.1,
                    1.2 * 1.1 * 1.3,
                    1,
                    1.5,
                    1.5 * 1.1,
                    1.5 * 1.1 * 1.2,
                ],
                "high": [
                    1,
                    1.1,
                    1.1 * 1.1,
                    1.1 * 1.1 * 1.3,
                    1,
                    1.5,
                    1.5 * 1.1,
                    1.5 * 1.1 * 1.2,
                ],
                "low": [
                    1,
                    1.2,
                    1.2 * 1.1,
                    1.2 * 1.1 * 1.3,
                    1,
                    1.5,
                    1.5 * 1.1,
                    1.5 * 1.1 * 1.4,
                ],
            },
            index=expected_index,
        )
        expected.columns.name = "CumulativeReturns"

        pd.testing.assert_frame_equal(result, expected)
