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

# Ratios


This module contains different ratios. A ratio is an object that is aimed at comparing a performance of an asset with another one and generally by controlling risk.


```{eval-rst}
.. autoclass:: portfolio_plan.ratios.InformationRatio
   :members:
   :show-inheritance:
```

We will load a mock data and calculate the information ratio between two plans

```{code-cell}
from portfolio_plan import Frequency
from portfolio_plan.resource import File

from portfolio_plan import Assets, Weights, Asset, Portfolio, Frequency, Plan
from portfolio_plan.data import example_prices_path
from datetime import datetime



START_DATE = datetime.strptime("2025-01-30", "%Y-%m-%d")
END_DATE = datetime.strptime("2025-03-01", "%Y-%m-%d")
EXAMPLE = example_prices_path()


def create_benchmark_plan():
    resource = File(path=EXAMPLE, frequency=Frequency.BDAILY)
    assets = Assets(
        [Asset("B")],
        resource=resource,
    )

    portfolio = Portfolio(
        assets=assets,
        weights=Weights(),
        name="B"
    )
    plan = Plan(
        portfolio=portfolio,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    return plan



def create_plan():
    assets = Assets(
        [Asset("A")],  # We use a single asset for simplicity of the example,
        resource=File(path=EXAMPLE, frequency=Frequency.BDAILY)
    )
    portfolio = Portfolio(
        assets=assets,
        name="Portfolio",
        weights=Weights(),
    )
    plan = Plan(
        portfolio = portfolio,
        start_date=START_DATE,
        end_date=END_DATE
    )
    return plan
```

```{code-cell}
from portfolio_plan.ratios import InformationRatio


plan = create_plan()
benchmark_plan = create_benchmark_plan()

ir = InformationRatio(plan, benchmark_plan)
ir_returns = ir.period_returns()
```

```{code-cell}
ir_returns.data
```



```{code-cell}
ir_cumulative_returns = ir_returns.cumulative("IR - Cumulative", check_base=False)
ir_cumulative_returns
```

```{code-cell}
ir_cumulative_returns.data
```
