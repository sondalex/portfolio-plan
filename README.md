![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsondalex%2Fportfolio-plan%2Fmain%2Fpyproject.toml)


# portfolio-plan

A Python package for creating, analyzing and visualizing investment strategies. Portfolio-plan allows users to define portfolios with custom asset allocations, simulate investment plans over time periods, calculate returns, and compare historical performance against benchmarks

## ⚠️


> **IMPORTANT DISCLAIMER**
>
> `portfolio-plan` supports download of Yahoo! Finance data with the help of yfinance package. **Please note the following important information**:
> * Yahoo!, Y!Finance, and Yahoo! finance are registered trademarks of Yahoo, Inc.
> * This software is not affiliated, endorsed, or vetted by Yahoo, Inc.
> * **Users are responsible** for ensuring their usage complies with Yahoo's Terms of Service.
> * `portfolio-plan` features that interface with `yfinance` are intended for **personal, non-commercial use**
>
> Yahoo's Terms of Service can be found at:
> - [API Terms](https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm)
> - [General Terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
> - [Overall Terms](https://policies.yahoo.com/us/en/yahoo/terms/index.htm)
>
> The maintainers of `portfolio-plan` are not responsible for any violations of Yahoo's Terms of Service by end users.


## Installation

```bash
pip install portfolio-plan
```


## Development

Install uv

```bash
pip install uv
```

Install pre-commit hooks

```bash
uv tool run pre-commit install
```

### Running pre-commit hooks

```bash
uv tool run pre-commit run --all-files
```
### Running the test suite:

```bash
uv run --active -m pytest
```

### Building

```bash
uv build
```


### Building the documentation

Installing the documentation dependencies

```bash
uv sync --active --only-dev
```
Generating the documentation:

```bash
cd docs/ && uv run --active make html
```
