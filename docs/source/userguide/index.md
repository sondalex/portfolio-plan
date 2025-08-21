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


(userguide)=
# User Guide



An investment plan covers a time period. It has an associated portfolio.
A portfolio includes a set of assets and their associated weights.


Let's define our plan.
We are interested in knowing what we would have earned if we invested equal weights in companies A, B and C.
Starting on January up to today (2025-05-02).

## Defining a Portfolio

```{code-cell}
from portfolio_plan import Assets, Weights, Asset, Portfolio, Frequency
from portfolio_plan.resource import File
from portfolio_plan.data import example_prices_path


portfolio = Portfolio(
    assets=Assets(
        [Asset("A"), Asset("B"), Asset("C")],
        resource=File(path=example_prices_path(), frequency=Frequency.BDAILY),
    ),
    name="Portfolio",
    weights=Weights(),  # No parameter means equal weights across assets
)

```

```{code-cell}
portfolio.plot_allocation()
```



## Defining a Plan


```{code-cell}
from portfolio_plan import Plan
from datetime import datetime


plan1 = Plan(
    portfolio=portfolio,
    start_date=datetime.strptime("2025-01-30", "%Y-%m-%d"),
    end_date=datetime.strptime("2025-03-01", "%Y-%m-%d"),
)
plan1
```

## Returns

### Individual Returns

```{code-cell}
returns = plan1.returns(portfolio.name) # Returns object
returns
```

If we wish to plot the returns:

```{code-cell}
returns.plot_close()
```

Or the cumulative returns:

```{code-cell}
cumulative_returns = returns.cumulative("Cumulative")
cumulative_returns.plot_close()
```

### Portfolio Returns


```{code-cell}
portfolio_returns1 = plan1.portfolio_returns()
portfolio_returns1.plot_close()
```



**Cumulative Portfolio Returns**:

```{code-cell}
cumulative_portfolio_returns1 = portfolio_returns1.cumulative("Cumulative")
cumulative_portfolio_returns1.plot_close()
```

## Combining plans

You might be interested in the joined returns of two investment plan happening at different timeframe.

```{code-cell}
plan2 = Plan(
    portfolio = portfolio,
    start_date = datetime.strptime("2025-03-02", "%Y-%m-%d"),
    end_date = datetime.strptime("2025-05-05", "%Y-%m-%d")
)
print(plan1)
print(plan2)
```
```{code-cell}
portfolio_returns2 = plan2.portfolio_returns()
```

```{code-cell}
cumulative_portfolio_returns2 = portfolio_returns2.cumulative("Cumulative")
```



The `>>` operator enables easy creation of joined plan.


```{code-cell}
joinedplan = plan1 >> plan2
joinedplan
```

```{code-cell}
joinedplan.plot_allocations()
```

**Note**: Plans have to be placed by order of timeframe, otherwise an error will be raised.


```{code-cell}
from portfolio_plan.errors import PeriodOverlapError


try:
    plan2 >> plan1
except PeriodOverlapError as e:
    print(e)
```


Interested in what is the return on investment of a joined plan ?


```{code-cell}
joinedplan_cumulative_returns = joinedplan.portfolio_cumulative_returns("Joined Plan")
joinedplan_cumulative_returns.data
```

```{code-cell}
joinedplan_cumulative_returns.data.loc["Portfolio"].head()
```

```{code-cell}
from plotnine import scale_x_datetime


(
    joinedplan_cumulative_returns.plot_close(scale_x=False) + # Define scale manually
    scale_x_datetime(date_breaks="1 week", date_labels="%Y-%m-%d")
)
```

## Comparing FinancialSeries

You might want to compare each returns, prices

### Prices


```{code-cell}
from portfolio_plan.financialseries import Comparison


comparison = Comparison(
    [plan1.prices(), plan2.prices()]
)
(
    comparison.plot_close(scale_x=False)
    + scale_x_datetime(date_breaks="1 week", date_labels="%Y-%m-%d")
)
```

### Returns

Instead of comparing prices, you want to compare returns.
You may want to look at how your strategy differs with an index fund for example


### Non-Cumulative



```{code-cell}
# NOT YET IMPLEMENTED
```


### Cumulative

We start by defining the benchmark plan (note that this is only necessary for multiple period investments).


```{code-cell}
from portfolio_plan.financialseries import Comparison


assets = Assets(
    [Asset("^D")],
    resource=File(path=example_prices_path(), frequency=Frequency.BDAILY),
)

benchmark = Portfolio(
    assets=assets,
    weights=Weights(),
    name="D Index"
)
benchmark_plan_1 = Plan(
    portfolio=benchmark,
    start_date=datetime.strptime("2025-01-30", "%Y-%m-%d"),
    end_date=datetime.strptime("2025-03-01", "%Y-%m-%d"),
)

benchmark_plan_2 = Plan(
    portfolio = benchmark,
    start_date = datetime.strptime("2025-03-02", "%Y-%m-%d"),
    end_date = datetime.strptime("2025-05-05", "%Y-%m-%d")
)

benchmark_joined = benchmark_plan_1 >> benchmark_plan_2
```
```{code-cell}
comparison = Comparison(
    [
        joinedplan.portfolio_cumulative_returns("Joined Plan"),
        benchmark_joined.portfolio_cumulative_returns("S&P 500")
    ]
)

(
    comparison.plot_close(scale_x=False) +
    scale_x_datetime(date_breaks="1 week", date_labels="%Y-%m")
)
```


## Information Ratio

It is possible to calculate the information ratio between two plan. Since each ratio corresponds to a specific investment plan, no annualization is performed.
Plans must have the same starting period or ending period.


```{code-cell}
from portfolio_plan.ratios import InformationRatio


ir = InformationRatio(plan1, benchmark_plan_1)
ir_returns1 = ir.period_returns()  # Returns a singleton dataframe
assert ir_returns1.is_singleton
ir_returns1
```

```{code-cell}
ir = InformationRatio(plan2, benchmark_plan_2)
ir_returns2 = ir.period_returns()
assert ir_returns2.is_singleton
ir_returns2

```

```{code-cell}
ir_returns1.data
```

```{code-cell}
ir_returns1.data
```

Information Ratio Returns are always singletons.
