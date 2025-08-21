from collections import UserDict
from enum import Enum

from portfolio_plan.errors import ValidationError


class Weights(UserDict):
    def __init__(self, **kwargs):
        """
        A user-defined dictionary containing symbols and associated weights.

        Example
        -------
        >>> weights = Weights(A=0.2, B=0.8)
        >>> weights

        >>> a_weight = weights["A"]

        # For more complicated symbols
        >>> weights = Weights(**{"A.S": 1})

        # No parameters passed to a weights object will be considered,
        # by convention as an equal weight
        >>> weights = Weights()
        """
        data = dict(kwargs)
        self._validate(data)
        super().__init__(data)

    @staticmethod
    def _validate(d: dict):
        if not d:
            return

        if not all(isinstance(v, (int, float)) for v in d.values()):
            raise ValidationError("All weights must be numeric (int or float).")

        total = sum(d.values())
        if not abs(total - 1.0) < 1e-8:
            raise ValidationError(f"Weights must sum to 1. Provided sum: {total}")

    def __repr__(self):
        items = ", ".join(f"{k}: {v:.4f}" for k, v in self.items())
        return f"Weights({{{items}}})"

    @property
    def empty(self) -> bool:
        return len(self) == 0


class Frequency(Enum):
    MINUTE = "1m"
    DAILY = "1d"
    BDAILY = "1bd"
    WEEKLY = "1wk"
    MONTHLY = "1mo"

    def to_pandas(self):
        # TODO: support pandas>=2.0.0
        match self:
            case Frequency.MINUTE:
                return "1T"
            case Frequency.DAILY:
                return "1D"  # day
            case Frequency.BDAILY:
                return "1B"
            case Frequency.WEEKLY:
                return "1W"  # week
            case Frequency.MONTHLY:
                return "1M"  # TODO: check this
            case _:
                raise ValueError(f"Unknown frequency: {self}")

    @staticmethod
    def from_pandas(value: str):
        match value:
            case "T":
                return Frequency.MINUTE
            case "D":
                return Frequency.DAILY
            case "B":
                return Frequency.BDAILY
            case "W":
                return Frequency.WEEKLY
            case "M":
                return Frequency.MONTHLY  # TODO: Check this
            case _:
                raise ValueError(f"Unknown pandas frequency: {value}")

    def to_yf(self):
        match self:
            case (
                Frequency.MINUTE
                | Frequency.DAILY
                | Frequency.WEEKLY
                | Frequency.MONTHLY
            ):
                return self.value

            case Frequency.BDAILY:
                return Frequency.DAILY.value
            case _:
                raise ValueError(f"Unknown frequency {self}")
