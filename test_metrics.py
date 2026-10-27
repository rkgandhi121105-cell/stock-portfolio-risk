"""Unit tests for financial risk metrics calculations in portfolio_risk.metrics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.metrics import (
    allocate_whole_rupees,
    calculate_annualized_volatility,
    calculate_beta,
    calculate_buy_and_hold_portfolio_value,
    calculate_cagr,
    calculate_correlation_matrix,
    calculate_daily_returns,
    calculate_diversification_ratio,
    calculate_drawdown_series,
    calculate_equal_weight_portfolio,
    calculate_hhi,
    calculate_max_drawdown,
    calculate_portfolio_daily_returns,
    calculate_portfolio_weights,
    classify_risk,
    interpret_hhi,
)


def test_calculate_portfolio_weights():
    """Verify weight calculation normalizes to 1.0 and matches nominal ratios."""
    investments = [30000.0, 25000.0, 15000.0, 20000.0, 10000.0]
    weights = calculate_portfolio_weights(investments)

    assert len(weights) == 5
    assert np.isclose(np.sum(weights), 1.0)
    assert np.allclose(weights, [0.30, 0.25, 0.15, 0.20, 0.10])


def test_calculate_portfolio_weights_invalid():
    """Verify weight calculation rejects zero or negative total investment."""
    with pytest.raises(ValueError, match="strictly greater than zero"):
        calculate_portfolio_weights([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="strictly greater than zero"):
        calculate_portfolio_weights([-100.0, -200.0])


def test_calculate_daily_returns():
    """Verify percentage return computation across successive dates."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 99.0],  # +10%, -10%
            "B": [50.0, 50.0, 55.0],    # 0%, +10%
        },
        index=dates,
    )
    returns = calculate_daily_returns(prices)

    assert len(returns) == 2
    assert np.isclose(returns.loc[dates[1], "A"], 0.10)
    assert np.isclose(returns.loc[dates[2], "A"], -0.10)
    assert np.isclose(returns.loc[dates[1], "B"], 0.00)
    assert np.isclose(returns.loc[dates[2], "B"], 0.10)


def test_buy_and_hold_portfolio_value():
    """Verify buy-and-hold capital trajectory matches asset price evolution."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "STOCK1": [100.0, 110.0, 120.0],
            "STOCK2": [200.0, 190.0, 210.0],
        },
        index=dates,
    )
    # Invest ₹50,000 in STOCK1 (500 shares) and ₹50,000 in STOCK2 (250 shares)
    investments = [50000.0, 50000.0]
    pv = calculate_buy_and_hold_portfolio_value(prices, investments)

    # Day 0: 500*100 + 250*200 = 50,000 + 50,000 = 1,00,000
    assert np.isclose(pv.iloc[0], 100000.0)

    # Day 1: 500*110 + 250*190 = 55,000 + 47,500 = 1,02,500
    assert np.isclose(pv.iloc[1], 102500.0)

    # Day 2: 500*120 + 250*210 = 60,000 + 52,500 = 1,12,500
    assert np.isclose(pv.iloc[2], 112500.0)


def test_annualized_volatility():
    """Verify annualized volatility scales daily standard deviation by sqrt(252)."""
    daily_returns = pd.Series([0.01, -0.01, 0.02, -0.015, 0.005])
    daily_std = daily_returns.std(ddof=1)
    expected_annual = daily_std * np.sqrt(252)

    computed = calculate_annualized_volatility(daily_returns, trading_days=252)
    assert np.isclose(computed, expected_annual)


def test_max_drawdown():
    """Verify maximum peak-to-trough calculation with deterministic path."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    # Peak: 120, trough: 90 => drawdown = (90 - 120) / 120 = -30 / 120 = -0.25 (-25%)
    values = pd.Series([100.0, 120.0, 100.0, 90.0, 115.0], index=dates)

    dd_series = calculate_drawdown_series(values)
    assert np.isclose(dd_series.iloc[1], 0.0)  # At peak 120, drawdown is 0
    assert np.isclose(dd_series.iloc[3], -0.25)  # At 90, drawdown is -25%

    mdd = calculate_max_drawdown(values)
    assert np.isclose(mdd, -0.25)


def test_calculate_beta():
    """Verify beta calculation against known market sensitivity."""
    np.random.seed(123)
    market_returns = pd.Series(np.random.normal(0, 0.01, 200))
    # Portfolio is exactly 1.5x market return plus independent noise
    noise = np.random.normal(0, 0.001, 200)
    port_returns = 1.5 * market_returns + noise

    beta = calculate_beta(port_returns, market_returns)
    assert np.isclose(beta, 1.5, atol=0.05)


def test_calculate_beta_zero_variance():
    """Verify zero benchmark variance does not crash and returns 0.0."""
    port_returns = pd.Series([0.01, 0.02, -0.01])
    bench_returns = pd.Series([0.0, 0.0, 0.0])  # zero variance
    beta = calculate_beta(port_returns, bench_returns)
    assert beta == 0.0


def test_calculate_hhi():
    """Verify Herfindahl-Hirschman Index for equal weights and single asset."""
    # 5 assets equal weights: 5 * (0.2)^2 = 0.20
    assert np.isclose(calculate_hhi([0.2, 0.2, 0.2, 0.2, 0.2]), 0.20)

    # 4 assets equal weights: 4 * (0.25)^2 = 0.25
    assert np.isclose(calculate_hhi([0.25, 0.25, 0.25, 0.25]), 0.25)

    # Single stock concentration
    assert np.isclose(calculate_hhi([1.0]), 1.0)

    # Interpretations
    assert interpret_hhi(0.18) == "Highly Diversified"
    assert interpret_hhi(0.28) == "Moderately Concentrated"
    assert interpret_hhi(0.40) == "Highly Concentrated"


def test_calculate_correlation_matrix():
    """Verify Pearson correlation matrix properties."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "A": np.linspace(0.01, 0.10, 10),
            "B": np.linspace(0.01, 0.10, 10),  # Perfectly correlated with A
            "C": -np.linspace(0.01, 0.10, 10),  # Perfectly negatively correlated with A
        },
        index=dates,
    )
    corr = calculate_correlation_matrix(df)

    assert np.isclose(corr.loc["A", "A"], 1.0)
    assert np.isclose(corr.loc["A", "B"], 1.0)
    assert np.isclose(corr.loc["A", "C"], -1.0)


def test_calculate_diversification_ratio():
    """Verify diversification ratio >= 1.0 when assets are not perfectly correlated."""
    weights = [0.5, 0.5]
    individual_vols = [0.20, 0.20]
    # If portfolio vol is 0.15 (due to diversification benefit), DR = (0.5*0.2 + 0.5*0.2) / 0.15 = 0.20 / 0.15 = 1.333
    dr = calculate_diversification_ratio(weights, individual_vols, portfolio_annualized_volatility=0.15)
    assert np.isclose(dr, 0.20 / 0.15)


def test_risk_classification():
    """Verify educational risk categories under defined thresholds."""
    assert classify_risk(0.10) == "Low"
    assert classify_risk(0.1499) == "Low"
    assert classify_risk(0.15) == "Moderate"
    assert classify_risk(0.2499) == "Moderate"
    assert classify_risk(0.25) == "High"
    assert classify_risk(0.35) == "High"


def test_equal_weight_portfolio(synthetic_price_data):
    """Verify equal-weight portfolio constructs matching weights and returns valid metrics."""
    df_stocks, s_benchmark = synthetic_price_data
    benchmark_returns = s_benchmark.pct_change().dropna()

    ew_res = calculate_equal_weight_portfolio(
        prices=df_stocks,
        total_capital=100000.0,
        benchmark_returns=benchmark_returns,
    )

    n_assets = df_stocks.shape[1]
    assert np.allclose(ew_res["weights"], [1.0 / n_assets] * n_assets)
    assert np.isclose(np.sum(ew_res["investments"]), 100000.0)
    assert ew_res["annualized_volatility"] > 0.0
    assert ew_res["max_drawdown"] <= 0.0
    assert ew_res["hhi"] == pytest.approx(1.0 / n_assets)


def test_calculate_cagr_standard():
    """Verify CAGR calculation over exactly 2 years."""
    # From 100 to 144 over 2 years (730 days) => CAGR = (144/100)^(1/2) - 1 = 1.2 - 1 = 0.20 (20%)
    dates = [pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")]
    values = pd.Series([100.0, 144.0], index=dates)
    cagr = calculate_cagr(values)
    assert np.isclose(cagr, 0.20, atol=0.005)


def test_calculate_cagr_insufficient_or_invalid():
    """Verify CAGR gracefully returns NaN for insufficient observations or invalid starting points."""
    # Single date
    assert np.isnan(calculate_cagr(pd.Series([100.0], index=[pd.Timestamp("2024-01-01")])))
    # Negative starting value
    assert np.isnan(calculate_cagr(pd.Series([-100.0, 150.0], index=[pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")])))


def test_allocate_whole_rupees_exact_sum():
    """Verify largest remainder method preserves exact total capital down to the rupee."""
    weights = [0.33333333, 0.33333333, 0.33333334]
    total_capital = 100000.0
    allocations = allocate_whole_rupees(total_capital, weights)

    assert len(allocations) == 3
    assert sum(allocations) == 100000.0
    assert all(isinstance(x, (int, float)) and x.is_integer() for x in allocations)
