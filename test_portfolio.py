"""Validation and end-to-end portfolio analysis tests supporting dynamic holdings."""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from portfolio_risk.config import MAX_HOLDINGS, MIN_HOLDINGS
from portfolio_risk.data import (
    DataFetchError,
    InsufficientDataError,
    _extract_price_series,
    fetch_historical_prices,
    fetch_portfolio_and_benchmark,
)
from portfolio_risk.metrics import analyze_portfolio
from portfolio_risk.validation import (
    PortfolioValidationError,
    normalize_ticker,
    validate_portfolio_inputs,
    validate_ticker_format,
)


def test_normalize_ticker():
    """Verify ticker normalization appends .NS and converts to uppercase."""
    assert normalize_ticker("reliance") == "RELIANCE.NS"
    assert normalize_ticker("tcs.ns") == "TCS.NS"
    assert normalize_ticker("  infy.ns  ") == "INFY.NS"
    assert normalize_ticker("^NSEI") == "^NSEI"


def test_validate_ticker_format():
    """Verify syntactic regex check for NSE tickers."""
    assert validate_ticker_format("RELIANCE.NS") is True
    assert validate_ticker_format("M&M.NS") is True
    assert validate_ticker_format("BAJAJ-AUTO.NS") is True
    assert validate_ticker_format("IOC.NS") is True
    assert validate_ticker_format("RELIANCE") is False  # Missing .NS
    assert validate_ticker_format("INVALID$$$.NS") is False


def test_validate_portfolio_inputs_dynamic_counts():
    """Verify portfolios with 1, 2, 4, and 20 holdings pass validation."""
    # 1 holding (single stock)
    t1, i1 = validate_portfolio_inputs(["RELIANCE.NS"], [50000.0])
    assert len(t1) == 1
    assert len(i1) == 1

    # 2 holdings
    t2, i2 = validate_portfolio_inputs(["RELIANCE.NS", "TCS.NS"], [30000.0, 20000.0])
    assert len(t2) == 2

    # 4 default holdings
    t4, i4 = validate_portfolio_inputs(
        ["RELIANCE.NS", "IOC.NS", "ICICIBANK.NS", "HDFCBANK.NS"],
        [35000.0, 40000.0, 30000.0, 20000.0],
    )
    assert len(t4) == 4

    # 20 holdings (at maximum limit)
    tickers_20 = [f"STOCK{i}.NS" for i in range(MAX_HOLDINGS)]
    amounts_20 = [5000.0] * MAX_HOLDINGS
    t20, i20 = validate_portfolio_inputs(tickers_20, amounts_20)
    assert len(t20) == MAX_HOLDINGS


def test_validate_portfolio_inputs_empty_and_exceeding():
    """Verify error if portfolio is empty or exceeds configured MAX_HOLDINGS."""
    # 0 holdings (empty)
    with pytest.raises(PortfolioValidationError, match="between 1 and 20 stocks"):
        validate_portfolio_inputs([], [])

    # 21 holdings (exceeds max of 20)
    tickers_21 = [f"STOCK{i}.NS" for i in range(MAX_HOLDINGS + 1)]
    amounts_21 = [5000.0] * (MAX_HOLDINGS + 1)
    with pytest.raises(PortfolioValidationError, match="between 1 and 20 stocks"):
        validate_portfolio_inputs(tickers_21, amounts_21)


def test_validate_portfolio_inputs_mismatched_length():
    """Verify error when ticker count does not match investments count."""
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    investments = [30000.0, 25000.0]
    with pytest.raises(PortfolioValidationError, match="Mismatch between number of tickers"):
        validate_portfolio_inputs(tickers, investments)


def test_validate_portfolio_inputs_duplicates():
    """Verify error when duplicate tickers are present."""
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "reliance.ns"]
    investments = [30000.0, 25000.0, 25000.0, 20000.0]
    with pytest.raises(PortfolioValidationError, match="Duplicate tickers are not allowed"):
        validate_portfolio_inputs(tickers, investments)


def test_validate_portfolio_inputs_invalid_investments():
    """Verify error on negative, zero, NaN, or below-minimum investments."""
    tickers = ["RELIANCE.NS", "TCS.NS"]

    # Negative investment
    with pytest.raises(PortfolioValidationError, match="at least ₹100.00"):
        validate_portfolio_inputs(tickers, [30000.0, -500.0])

    # Zero investment
    with pytest.raises(PortfolioValidationError, match="at least ₹100.00"):
        validate_portfolio_inputs(tickers, [30000.0, 0.0])

    # NaN investment
    with pytest.raises(PortfolioValidationError, match="finite number"):
        validate_portfolio_inputs(tickers, [30000.0, float("nan")])

    # Non-numeric string
    with pytest.raises(PortfolioValidationError, match="valid numeric value"):
        validate_portfolio_inputs(tickers, [30000.0, "invalid"])


def test_extract_price_series_multiindex():
    """Verify extracting Adj Close from MultiIndex yfinance response."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    cols = pd.MultiIndex.from_product([["Adj Close", "Close"], ["RELIANCE.NS", "TCS.NS"]])
    df = pd.DataFrame(
        [
            [2500.0, 3800.0, 2490.0, 3790.0],
            [2520.0, 3850.0, 2510.0, 3840.0],
            [2510.0, 3820.0, 2505.0, 3810.0],
        ],
        index=dates,
        columns=cols,
    )
    extracted = _extract_price_series(df, ["RELIANCE.NS", "TCS.NS"])
    assert list(extracted.columns) == ["RELIANCE.NS", "TCS.NS"]
    assert extracted.loc[dates[0], "RELIANCE.NS"] == 2500.0


def test_fetch_portfolio_and_benchmark_mocked(mock_yf_download, sample_tickers, sample_investments):
    """Verify end-to-end data fetching and alignment using mocked yfinance."""
    with patch("yfinance.download", side_effect=mock_yf_download):
        stocks_df, bench_series = fetch_portfolio_and_benchmark(
            sample_tickers,
            benchmark_ticker="^NSEI",
            period="5y",
        )

        assert len(stocks_df.columns) == 5
        assert len(stocks_df) == len(bench_series)
        assert (stocks_df.index == bench_series.index).all()


def test_fetch_historical_prices_missing_ticker(mock_yf_download):
    """Verify DataFetchError is raised when a requested ticker is completely missing/NaN."""
    with patch("yfinance.download", side_effect=mock_yf_download):
        with pytest.raises(DataFetchError, match="Data unavailable for the following ticker"):
            fetch_historical_prices(["RELIANCE.NS", "NONEXISTENT.NS"])


def test_analyze_portfolio_single_holding(synthetic_price_data):
    """Verify analyze_portfolio cleanly calculates metrics for a 1-holding portfolio."""
    stocks_df, bench_series = synthetic_price_data
    single_stock_df = stocks_df[["RELIANCE.NS"]].copy()

    result = analyze_portfolio(
        prices=single_stock_df,
        investments=[50000.0],
        benchmark_prices=bench_series,
    )

    summary = result["summary"]
    assert summary.total_investment == 50000.0
    assert summary.num_holdings == 1
    assert summary.annualized_volatility > 0.0
    assert summary.max_drawdown <= 0.0
    assert not np.isnan(summary.beta)
    assert np.isclose(summary.hhi, 1.0)
    assert np.isclose(summary.diversification_ratio, 1.0)
    assert summary.risk_classification in ["Low", "Moderate", "High"]


def test_analyze_portfolio_multiple_holdings(synthetic_price_data, sample_tickers, sample_investments):
    """Verify analyze_portfolio returns complete metrics for multi-asset portfolios."""
    stocks_df, bench_series = synthetic_price_data

    result = analyze_portfolio(
        prices=stocks_df,
        investments=sample_investments,
        benchmark_prices=bench_series,
    )

    summary = result["summary"]
    assert summary.total_investment == 100000.0
    assert summary.num_holdings == 5
    assert summary.annualized_volatility > 0.0
    assert summary.max_drawdown <= 0.0
    assert not np.isnan(summary.beta)
    assert 0.0 < summary.hhi <= 1.0
    assert summary.diversification_ratio >= 1.0

    hdf = result["holdings_df"]
    assert len(hdf) == 5
    assert np.isclose(hdf["Weight (%)"].sum(), 100.0)

    ew = result["equal_weight"]
    assert np.isclose(ew["weights"].sum(), 1.0)
    assert ew["annualized_volatility"] > 0.0
