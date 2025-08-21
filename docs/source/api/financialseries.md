---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
    display_name: Python 3
    language: python
    name: portfolio_plan
---

# Financial Series

Financial Series is an abstract class representing financial time series.

Concrete implementations include Prices, Returns, CumulativeReturns

```{eval-rst}
.. autoclass:: portfolio_plan.financialseries.Prices
   :members:
   :show-inheritance:
```




```{eval-rst}
.. autoclass:: portfolio_plan.financialseries.Returns
   :members:
   :show-inheritance:
```

```{code-cell}
from portfolio_plan import Frequency
from portfolio_plan.resource import File
from portfolio_plan.utils import returns_from_prices
from portfolio_plan.data import example_prices_path
from datetime import datetime


START_DATE = datetime.strptime("2025-01-30", "%Y-%m-%d")
END_DATE = datetime.strptime("2025-03-01", "%Y-%m-%d")


prices = File(path=example_prices_path(), frequency=Frequency.BDAILY).fetch(
    ["A"],
    start_date=START_DATE,
    end_date=END_DATE,
    name="Prices"
)

returns_from_prices(prices.data)
```



```{eval-rst}
.. autoclass:: portfolio_plan.financialseries.CumulativeReturns
   :members:
   :show-inheritance:
```
