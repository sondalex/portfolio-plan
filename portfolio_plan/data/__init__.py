from importlib import resources
from pathlib import PosixPath


def example_prices_path() -> PosixPath:
    filename = "mock_example.parquet"
    files = resources.files("portfolio_plan.data")
    for file in files.iterdir():
        if file.name == filename:
            if not isinstance(file, PosixPath):
                raise RuntimeError("Uncaught error, can you raise an issue ?")
            return file
    raise FileNotFoundError(f"{filename} file not found in resources")


def mock_prices_path() -> PosixPath:
    filename = "mock.parquet"
    files = resources.files("portfolio_plan.data")
    for file in files.iterdir():
        if file.name == filename:
            if not isinstance(file, PosixPath):
                raise RuntimeError("Uncaught error, can you raise an issue ?")
            return file
    raise FileNotFoundError(f"{filename} file not found in resources")


def mock_returns_path() -> PosixPath:
    filename = "mock_returns.parquet"
    files = resources.files("portfolio_plan.data")
    for file in files.iterdir():
        if file.name == filename:
            if not isinstance(file, PosixPath):
                raise RuntimeError("Uncaught error, can you raise an issue ?")
            return file
    raise FileNotFoundError(f"{filename} file not found in resources")


def mock_cumulative_returns_path() -> PosixPath:
    filename = "mock_cumulative_returns.parquet"
    files = resources.files("portfolio_plan.data")
    for file in files.iterdir():
        if file.name == filename:
            if not isinstance(file, PosixPath):
                raise RuntimeError("Uncaught error, can you raise an issue ?")
            return file
    raise FileNotFoundError(f"{filename} file not found in resources")
