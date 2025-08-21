----
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
    display_name: Python 3
    language: python
    name: portfolio_plan
----


# Utilities

```{eval-rst}
.. autoclass:: portfolio_plan.utils.plot_lines
   :members:
   :show-inheritance:
```


```{eval-rst}
.. autoclass:: portfolio_plan.utils.plot_prices
   :members:
   :show-inheritance:
```


```{eval-rst}
.. autoclass:: portfolio_plan.utils.plot_returns
   :members:
   :show-inheritance:
```


```{eval-rst}
.. autoclass:: portfolio_plan.utils.plot_cumulative_returns
   :members:
   :show-inheritance:
```


```{eval-rst}
.. autoclass:: portfolio_plan.utils.plot_allocation
   :members:
   :show-inheritance:
```

```{code-cell}
from portfolio_plan.utils import plot_allocation, plot_allocations, Period
from portfolio_plan import Weights


weights = Weights(A=0.2, B=0.8)

p = plot_allocation(weights)
```

```{code-cell}
p
```

```{code-cell}
from datetime import datetime


weights_list = [
    Weights(A=0.4, B=0.3, C=0.3),
    Weights(A=0.5, B= 0.2, C=0.3),
]
periods = [
    Period(
        start=datetime(2025, 1, 1),
        end=datetime(2025, 6, 30),
        frequency=None,
    ),
    Period(
        start=datetime(2025, 7, 1),
        end=datetime(2025, 12, 31),
        frequency=None
    ),
]
plot = plot_allocations(weights_list, periods, variant="dawn")
plot
```

```{code-cell}
from portfolio_plan.utils import plot_allocations2


plot = plot_allocations2(weights_list, periods, variant="dawn")
plot
```
