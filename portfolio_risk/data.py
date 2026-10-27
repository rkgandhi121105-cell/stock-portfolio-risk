"""Historical financial data retrieval and sanitization module.

Fetches adjusted closing price history for National Stock Exchange (NSE) equities
and benchmark indices using Yahoo Finance (yfinance). Robustly handles network
interruptions, missing symbols, MultiIndex structures, and date alignment.
"""

from typing import List, Optional, Sequence, Tuple
import warnings
import pandas as pd
import yfinance as yf

# Suppress upstream yfinance Pandas 4 deprecation warnings
warnings.filterwarnings("ignore", message=".*The inplace keyword.*")
from portfolio_risk.config import BENCHMARK_TICKER, DEFAULT_PERIOD_YEARS, TRADING_DAYS_PER_YEAR


class DataFetchError(RuntimeError):
    """Exception raised when market data retrieval or validation fails."""
    pass


class InsufficientDataError(DataFetchError):
    """Exception raised when retrieved historical data has too few observation points."""
    pass


def _extract_price_series(
    raw_df: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Extract and normalize Adjusted Close prices from a raw yfinance DataFrame.

    Handles different yfinance column layouts, including MultiIndex headers
    and single-ticker responses.

    Args:
        raw_df: Raw DataFrame returned by yfinance.download.
        tickers: Expected list of ticker symbols.

    Returns:
        pd.DataFrame with dates as index and tickers as clean column headers.

    Raises:
        DataFetchError: If neither 'Adj Close' nor 'Close' price fields are present.
    """
    if raw_df.empty:
        raise DataFetchError("Received empty dataset from market data provider.")

    df = raw_df.copy()

    # Handle MultiIndex columns (common when downloading multiple tickers in yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)

        if "Adj Close" in level_0:
            prices = df["Adj Close"].copy()
        elif "Close" in level_0:
            prices = df["Close"].copy()
        elif "Adj Close" in level_1:
            prices = df.xs("Adj Close", axis=1, level=1).copy()
        elif "Close" in level_1:
            prices = df.xs("Close", axis=1, level=1).copy()
        else:
            raise DataFetchError("Neither 'Adj Close' nor 'Close' found in MultiIndex columns.")
    else:
        # Single-level columns
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]].copy()
            # If single ticker requested, rename column to ticker
            if len(tickers) == 1:
                prices.columns = [tickers[0]]
        elif "Close" in df.columns:
            prices = df[["Close"]].copy()
            if len(tickers) == 1:
                prices.columns = [tickers[0]]
        else:
            # Maybe the columns are already tickers directly
            prices = df.copy()

    # Normalize index: ensure sorted DatetimeIndex without timezone offsets
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices = prices.sort_index()
    return prices


def fetch_historical_prices(
    tickers: Sequence[str],
    period: str = "5y",
) -> pd.DataFrame:
    """Download historical adjusted closing prices for a list of equity tickers.

    Args:
        tickers: Sequence of ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS']).
        period: Historical lookback period string accepted by yfinance ('1y', '3y', '5y').

    Returns:
        pd.DataFrame with dates as index and ticker symbols as columns.

    Raises:
        DataFetchError: On download failures, missing tickers, or empty results.
        InsufficientDataError: If valid history is shorter than 30 trading days.
    """
    if not tickers:
        raise DataFetchError("No ticker symbols provided for data retrieval.")

    ticker_list = list(tickers)
    try:
        raw_data = yf.download(
            tickers=ticker_list,
            period=period,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise DataFetchError(
            f"Network or service error while downloading data from Yahoo Finance: {exc}"
        ) from exc

    if raw_data is None or raw_data.empty:
        raise DataFetchError(
            f"No price data returned for tickers: {', '.join(ticker_list)}. "
            "Please verify ticker symbols or internet connection."
        )

    prices = _extract_price_series(raw_data, ticker_list)

    # Verify that all requested tickers are present as columns
    missing_symbols = []
    for t in ticker_list:
        if t not in prices.columns:
            missing_symbols.append(t)
        else:
            # Check if all values are NaN
            if prices[t].dropna().empty:
                missing_symbols.append(t)

    if missing_symbols:
        raise DataFetchError(
            f"Data unavailable for the following ticker(s): {', '.join(missing_symbols)}. "
            "They may be delisted, incorrectly formatted, or have no trading history on Yahoo Finance."
        )

    # Subset strictly to requested tickers in original order
    prices = prices[ticker_list].copy()

    # Drop any leading dates before all stocks started trading to avoid introducing synthetic data
    prices = prices.dropna(how="any")

    if len(prices) < 30:
        raise InsufficientDataError(
            f"Insufficient overlapping historical price points ({len(prices)} days). "
            "At least 30 trading days of synchronized price data are required for statistical analysis."
        )

    return prices


def fetch_benchmark_prices(
    benchmark_ticker: str = BENCHMARK_TICKER,
    period: str = "5y",
) -> pd.Series:
    """Download historical adjusted closing prices for the benchmark index (e.g. ^NSEI).

    Args:
        benchmark_ticker: Benchmark symbol (default: '^NSEI' for Nifty 50).
        period: Historical lookback period.

    Returns:
        pd.Series with dates as index and adjusted close prices.

    Raises:
        DataFetchError: If benchmark data cannot be fetched or is empty.
    """
    try:
        raw_data = yf.download(
            tickers=[benchmark_ticker],
            period=period,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise DataFetchError(
            f"Failed to download benchmark index '{benchmark_ticker}': {exc}"
        ) from exc

    if raw_data is None or raw_data.empty:
        raise DataFetchError(
            f"Benchmark data for '{benchmark_ticker}' was not returned by market provider."
        )

    prices_df = _extract_price_series(raw_data, [benchmark_ticker])
    if benchmark_ticker in prices_df.columns:
        series = prices_df[benchmark_ticker].dropna()
    elif len(prices_df.columns) == 1:
        series = prices_df.iloc[:, 0].dropna()
    else:
        raise DataFetchError(f"Could not extract series for benchmark '{benchmark_ticker}'.")

    series.name = benchmark_ticker
    return series


def fetch_portfolio_and_benchmark(
    tickers: Sequence[str],
    benchmark_ticker: str = BENCHMARK_TICKER,
    period: str = "5y",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Retrieve synchronized historical prices for both portfolio stocks and the benchmark.

    Aligns timestamps on common trading dates where all holdings and the benchmark
    have valid recorded market prices. Missing prices are never silently filled with 0.

    Args:
        tickers: List of stock symbols.
        benchmark_ticker: Symbol for the benchmark index.
        period: Historical lookback window.

    Returns:
        Tuple of (stock_prices_df, benchmark_series), synchronized on matching dates.

    Raises:
        DataFetchError: If either stock or benchmark data fails to download.
        InsufficientDataError: If aligned history has fewer than 30 common trading days.
    """
    stock_prices = fetch_historical_prices(tickers, period=period)
    benchmark_prices = fetch_benchmark_prices(benchmark_ticker, period=period)

    # Synchronize dates (inner join)
    common_dates = stock_prices.index.intersection(benchmark_prices.index)
    if len(common_dates) < 30:
        raise InsufficientDataError(
            f"Only {len(common_dates)} synchronized trading days found between portfolio stocks "
            f"and benchmark '{benchmark_ticker}'. Need at least 30 trading days."
        )

    aligned_stocks = stock_prices.loc[common_dates].copy()
    aligned_benchmark = benchmark_prices.loc[common_dates].copy()

    return aligned_stocks, aligned_benchmark
