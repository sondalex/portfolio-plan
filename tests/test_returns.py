import numpy as np
import pandas as pd
import pytest

from portfolio_plan import const
from portfolio_plan.errors import ValidationError
from portfolio_plan.financialseries import Returns


class TestReturns:
    @pytest.fixture
    def sample_returns_data(self):
        """
        Create sample returns data for testing
        """
        values = [
            np.nan,  # A, 2020-01-01
            ((110.0 - 100.0) / 100.0),  # A, 2020-01-02
            ((115.0 - 110.0) / 110.0),  # A, 2020-01-03
            ((120.0 - 115.0) / 115.0),  # A, 2020-01-04
            np.nan,  # B, 2020-01-01
            ((210.0 - 200.0) / 200.0),  # B, 2020-01-02
            ((220.0 - 210.0) / 210.0),  # B, 2020-01-03
            ((225.0 - 220.0) / 220.0),  # B, 2020-01-04
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

        return pd.DataFrame(
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

    def test_returns_initialization(self, sample_returns_data):
        """Test that Returns class initializes correctly with valid data"""
        returns = Returns(sample_returns_data, name="Test Returns")
        assert returns.name == "Test Returns"
        pd.testing.assert_frame_equal(returns.data, sample_returns_data)

    def test_returns_initialization_invalid_data(self):
        invalid_values = [
            0.1,  # A, 2020-01-01 (not NaN)
            ((110.0 - 100.0) / 100.0),  # A, 2020-01-02
            ((115.0 - 110.0) / 110.0),  # A, 2020-01-03
            ((120.0 - 115.0) / 115.0),  # A, 2020-01-04
            np.nan,  # B, 2020-01-01 (correctly NaN)
            ((210.0 - 200.0) / 200.0),  # B, 2020-01-02
            ((220.0 - 210.0) / 210.0),  # B, 2020-01-03
            ((225.0 - 220.0) / 220.0),  # B, 2020-01-04
        ]

        invalid_index = pd.MultiIndex.from_tuples(
            [
                ("A", pd.Timestamp("2020-01-01")),
                ("A", pd.Timestamp("2020-01-02")),
                ("A", pd.Timestamp("2020-01-03")),
                ("A", pd.Timestamp("2020-01-04")),
                ("B", pd.Timestamp("2020-01-01")),
                ("B", pd.Timestamp("2020-01-02")),
                ("B", pd.Timestamp("2020-01-03")),
                ("B", pd.Timestamp("2020-01-04")),
            ],
            names=["symbol", "date"],
        )

        invalid_data = pd.DataFrame(
            {
                "open": invalid_values,
                "close": invalid_values,
                "high": invalid_values,
                "low": invalid_values,
            },
            index=invalid_index,
        )

        with pytest.raises(ValidationError):
            Returns(invalid_data, name="Invalid Returns")

        valid_values = [
            np.nan,  # A, 2020-01-01 (correctly NaN)
            ((110.0 - 100.0) / 100.0),  # A, 2020-01-02
            ((115.0 - 110.0) / 110.0),  # A, 2020-01-03
            ((120.0 - 115.0) / 115.0),  # A, 2020-01-04
            np.nan,  # B, 2020-01-01 (correctly NaN)
            ((210.0 - 200.0) / 200.0),  # B, 2020-01-02
            ((220.0 - 210.0) / 210.0),  # B, 2020-01-03
            ((225.0 - 220.0) / 220.0),  # B, 2020-01-04
        ]

        valid_data = pd.DataFrame(
            {
                "open": valid_values,
                "close": valid_values,
                "high": valid_values,
                "low": valid_values,
            },
            index=invalid_index,
        )

        Returns(valid_data, name="Valid Returns")

    def test_avg_without_weights(self, sample_returns_data):
        """Test the avg method without specifying weights"""
        returns = Returns(sample_returns_data, name="Test Returns")
        avg_returns = returns.avg(weights={})

        # Calculate expected results manually: average across symbols for each date
        expected_values = [
            np.nan,  # 2020-01-01 (all NaN)
            (((110.0 - 100.0) / 100.0) + ((210.0 - 200.0) / 200.0)) / 2,  # 2020-01-02
            (((115.0 - 110.0) / 110.0) + ((220.0 - 210.0) / 210.0)) / 2,  # 2020-01-03
            (((120.0 - 115.0) / 115.0) + ((225.0 - 220.0) / 220.0)) / 2,  # 2020-01-04
        ]

        expected_data = pd.DataFrame(
            {
                "open": expected_values,
                "close": expected_values,
                "high": expected_values,
                "low": expected_values,
            },
            index=pd.MultiIndex.from_tuples(
                [
                    (const.DEFAULT_PORTFOLIO_NAME, pd.Timestamp("2020-01-01")),
                    (const.DEFAULT_PORTFOLIO_NAME, pd.Timestamp("2020-01-02")),
                    (const.DEFAULT_PORTFOLIO_NAME, pd.Timestamp("2020-01-03")),
                    (const.DEFAULT_PORTFOLIO_NAME, pd.Timestamp("2020-01-04")),
                ],
                names=["symbol", "date"],
            ),
        )

        pd.testing.assert_frame_equal(avg_returns.data, expected_data)

    def test_avg_with_weights(self, sample_returns_data):
        """Test the avg method with specified weights"""
        returns = Returns(sample_returns_data, name="Test Returns")

        weights = {"A": 0.6, "B": 0.4}

        avg_returns = returns.avg(weights=weights, portfolio_name="Weighted Portfolio")

        # With the fixed implementation, expected values should be a proper weighted average
        expected_values = [
            np.nan,  # 2020-01-01
            ((110.0 - 100.0) / 100.0) * 0.6
            + ((210.0 - 200.0) / 200.0) * 0.4,  # 2020-01-02
            ((115.0 - 110.0) / 110.0) * 0.6
            + ((220.0 - 210.0) / 210.0) * 0.4,  # 2020-01-03
            ((120.0 - 115.0) / 115.0) * 0.6
            + ((225.0 - 220.0) / 220.0) * 0.4,  # 2020-01-04
        ]

        expected_data = pd.DataFrame(
            {
                "open": expected_values,
                "close": expected_values,
                "high": expected_values,
                "low": expected_values,
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("Weighted Portfolio", pd.Timestamp("2020-01-01")),
                    ("Weighted Portfolio", pd.Timestamp("2020-01-02")),
                    ("Weighted Portfolio", pd.Timestamp("2020-01-03")),
                    ("Weighted Portfolio", pd.Timestamp("2020-01-04")),
                ],
                names=["symbol", "date"],
            ),
        )

        pd.testing.assert_frame_equal(avg_returns.data, expected_data)
