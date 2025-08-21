import json
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import IO, Any, List, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from pyarrow.lib import Schema as ArrowSchema

from portfolio_plan._properties import Frequency
from portfolio_plan.financialseries import Prices

EXPECTED_SCHEMA: ArrowSchema = pa.schema(
    cast(
        List[pa.Field],
        [
            pa.field("symbol", pa.string()),
            pa.field("date", pa.timestamp(unit="ns")),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
        ],
    )
)


def read_schema(source: Path | IO[Any]) -> ArrowSchema:
    file = pq.ParquetFile(source)
    return file.schema_arrow


def weakly_equivalent(schema: pa.Schema, expected_schema: pa.Schema) -> bool:
    """
    Checks if two Arrow schemas are weakly equivalent.

    Weak equivalence means:
    - Same set of field names
    - Same data types for corresponding field names
    - Ignores field order and nullability

    Parameters
    ==========
        schema: pa.Schema
        expected_schema: pa.Schema
            The schema to compare against.

    Returns
    =======
        True if schemas are weakly equivalent, False otherwise.
    """

    def normalize_fields(s):
        return {f.name: f.type for f in s}

    return normalize_fields(schema) == normalize_fields(expected_schema)


# TODO: Include weak equality
def validate_schema(schema: ArrowSchema):
    if not weakly_equivalent(schema, EXPECTED_SCHEMA):
        raise ValueError(
            f"""Schema does not match the expected schema.
            Expected:
            {str(EXPECTED_SCHEMA)}

            got {str(schema)}
            """
        )


def has_index(schema: ArrowSchema):
    raw = schema.metadata.get(b"pandas") if schema.metadata else None
    if raw is not None:
        metadata = json.loads(raw)
        index_columns = metadata.get("index_columns")
        if index_columns is not None:
            if not isinstance(index_columns, list):
                return False
            for el in index_columns:
                if not isinstance(el, str):
                    return False
            if set(index_columns) == {"symbol", "date"}:
                return True
    return False


class Resource(ABC):
    """ """

    @abstractmethod
    def fetch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        name: str,
        **kwargs,
    ) -> Prices:
        pass


class File(Resource):
    def __init__(self, frequency: Frequency, path: PathLike):
        """
        Parameters
        ----------
            path: PathLike
                Path to a parquet file
            frequency:
                Target frequency
        """
        self._path = path
        self._frequency = frequency

    def fetch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        name: str,
        **kwargs,
    ) -> Prices:
        """
        Parameters
        ----------
        Raises
        ------
        ValidationError
        IOError
        """
        path = self._path
        schema = read_schema(Path(path))
        validate_schema(schema)

        df = pd.read_parquet(path)
        if not has_index(schema):
            df = df.set_index(["symbol", "date"])
        symbols_idx = df.index.get_level_values("symbol")

        subset = df.loc[symbols_idx.isin(symbols)]
        dates = subset.index.get_level_values("date")

        subset = subset.loc[(dates >= start_date) & (dates <= end_date)]
        prices = Prices(subset, name=name).resample_period(self._frequency)
        return prices


class YFinance(Resource):
    """
    Fetch and process price data from Yahoo! Finance.

    .. warning::
        **DISCLAIMER:**

        * Yahoo!, Y!Finance, and Yahoo! finance are registered trademarks of Yahoo, Inc.
        * This implementation uses the yfinance package to access Yahoo! Finance data.
        * This code is not affiliated, endorsed, or vetted by Yahoo, Inc.
        * Users are responsible for ensuring their usage complies with Yahoo Finance's Terms of Service.
        * `portfolio-plan` features that interface with `yfinance` are intended for **personal, non-commercial use**

        Relevant Yahoo Terms of Service:

        * API Terms: https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm
        * General Terms: https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html
        * Overall Terms: https://policies.yahoo.com/us/en/yahoo/terms/index.htm
    """

    def __init__(self, frequency: Frequency):
        self._frequency = frequency

    def fetch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        name: str,
        **kwargs,
    ) -> Prices:
        """
        Parameters
        ----------
        symbols: List[str]
            List of symbols to download
        start_date: datetime
        end_date: datetime
        name: str

        Raises
        ------
        NetworkError
        IOError
        """
        history = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            interval=self._frequency.to_yf(),
            progress=False,
            auto_adjust=True,
        )
        # NOTE: As freq will add NaN to holidays for example
        history = history.asfreq("B")

        history = history.stack(dropna=False).reorder_levels([1, 0])
        history = history.drop(columns=["Volume"])
        history.columns = history.columns.str.lower()
        history.index.names = ["symbol", "date"]
        return Prices(history, name=name)


class Memory(Resource):
    def __init__(self, data: BytesIO, frequency: Frequency):
        self._data = data
        self._frequency = frequency

    def fetch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        name: str,
        **kwargs,
    ):
        # TODO: validate schema from buffer object
        schema = read_schema(self._data)
        validate_schema(schema)

        df = pd.read_parquet(self._data, engine="pyarrow")
        if not has_index(schema):
            df = df.set_index(["symbol", "date"])
        symbols_idx = df.index.get_level_values("symbol")

        subset = df.loc[symbols_idx.isin(symbols)]
        dates = subset.index.get_level_values("date")

        subset = subset.loc[(dates >= start_date) & (dates <= end_date)]
        prices = Prices(subset, name=name).resample_period(self._frequency)
        return prices
