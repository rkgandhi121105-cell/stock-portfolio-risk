"""Pytest configuration and synthetic fixtures for portfolio risk testing.

Provides synthetic price series, deterministic portfolios, and mocked yfinance
responses to verify financial metrics and edge-case behavior without internet access.
"""

from typing import Dict, List, Tuple
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_tickers() -> List[str]:
    """Standard 5-stock portfolio tickers."""
    return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


@pytest.fixture
def sample_investments() -> List[float]:
    """Nominal investment allocations totaling ₹1,00,000."""
    return [30000.0, 25000.0, 15000.0, 20000.0, 10000.0]


@pytest.fixture
def synthetic_price_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Generate 100 days of deterministic synthetic price history.

    Includes 5 equities and 1 benchmark series with reproducible statistical behavior.
    """
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # Generate correlated daily returns
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    n_stocks = len(tickers)

    # Base market factor
    market_returns = np.random.normal(loc=0.0005, scale=0.01, size=n_days)
    stock_returns = np.zeros((n_days, n_stocks))

    betas = [1.1, 0.8, 1.2, 0.9, 1.0]
    for i in range(n_stocks):
        idiosyncratic = np.random.normal(loc=0.0, scale=0.008, size=n_days)
        stock_returns[:, i] = betas[i] * market_returns + idiosyncratic

    # Cumulative price series starting from base prices
    base_stock_prices = [2500.0, 3800.0, 1500.0, 1600.0, 1000.0]
    stock_prices_matrix = np.zeros((n_days, n_stocks))
    stock_prices_matrix[0] = base_stock_prices

    for t in range(1, n_days):
        stock_prices_matrix[t] = stock_prices_matrix[t - 1] * (1.0 + stock_returns[t])

    df_stocks = pd.DataFrame(stock_prices_matrix, index=dates, columns=tickers)

    # Benchmark price series
    base_benchmark = 21000.0
    benchmark_prices = np.zeros(n_days)
    benchmark_prices[0] = base_benchmark
    for t in range(1, n_days):
        benchmark_prices[t] = benchmark_prices[t - 1] * (1.0 + market_returns[t])

    s_benchmark = pd.Series(benchmark_prices, index=dates, name="^NSEI")

    return df_stocks, s_benchmark


@pytest.fixture
def mock_yf_download(synthetic_price_data):
    """Factory fixture returning a mock yfinance.download function."""
    df_stocks, s_benchmark = synthetic_price_data

    def _mock_download(tickers, *args, **kwargs):
        if isinstance(tickers, str):
            t_list = [tickers]
        else:
            t_list = list(tickers)

        if "^NSEI" in t_list:
            # Benchmark call
            df = pd.DataFrame({"Adj Close": s_benchmark, "Close": s_benchmark})
            return df

        # Equity call
        # Return MultiIndex DataFrame matching yfinance standard
        cols = pd.MultiIndex.from_product([["Adj Close", "Close"], t_list])
        data = {}
        for t in t_list:
            if t in df_stocks.columns:
                data[("Adj Close", t)] = df_stocks[t]
                data[("Close", t)] = df_stocks[t]
            else:
                data[("Adj Close", t)] = np.nan
                data[("Close", t)] = np.nan

        res_df = pd.DataFrame(data, index=df_stocks.index)
        return res_df

    return _mock_download
