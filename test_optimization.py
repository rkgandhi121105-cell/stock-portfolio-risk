"""Unit tests for Minimum-Variance Portfolio Optimization in portfolio_risk.optimization."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.optimization import (
    check_optimization_feasibility,
    optimize_minimum_variance,
)


def test_optimization_feasibility():
    """Verify mathematical feasibility conditions n * w_max >= 1.0."""
    # 4 assets with 40% cap: 4 * 0.4 = 1.6 >= 1.0 -> Feasible
    ok, _ = check_optimization_feasibility(num_assets=4, max_weight=0.40)
    assert ok is True

    # 2 assets with 40% cap: 2 * 0.4 = 0.8 < 1.0 -> Infeasible
    ok, reason = check_optimization_feasibility(num_assets=2, max_weight=0.40)
    assert ok is False
    assert "Infeasible" in reason

    # 1 asset with 100% cap -> Feasible
    ok, _ = check_optimization_feasibility(num_assets=1, max_weight=1.0)
    assert ok is True

    # 1 asset with 50% cap -> Infeasible
    ok, reason = check_optimization_feasibility(num_assets=1, max_weight=0.50)
    assert ok is False
    assert "Infeasible" in reason


def test_optimize_minimum_variance_standard(synthetic_price_data):
    """Verify optimizer calculates constrained minimum-variance weights."""
    stocks_df, bench_series = synthetic_price_data
    bench_returns = bench_series.pct_change().dropna()
    total_capital = 125000.0
    max_weight = 0.40

    res = optimize_minimum_variance(
        prices=stocks_df,
        total_capital=total_capital,
        benchmark_returns=bench_returns,
        max_weight=max_weight,
    )

    assert res.success is True
    assert len(res.weights) == len(stocks_df.columns)
    # Weights sum to 1.0
    assert np.isclose(np.sum(res.weights), 1.0, atol=1e-4)
    # Weights are non-negative
    assert (res.weights >= -1e-6).all()
    # Weights do not exceed max_weight
    assert (res.weights <= max_weight + 1e-4).all()
    # Recommended whole-rupee investments sum exactly to total_capital
    assert np.isclose(np.sum(res.investments), total_capital)
    # Volatility and drawdown are valid
    assert res.annualized_volatility > 0.0
    assert res.max_drawdown <= 0.0
    assert not np.isnan(res.beta)
    assert 0.0 < res.hhi <= 1.0


def test_optimize_minimum_variance_single_asset(synthetic_price_data):
    """Verify single-stock portfolio trivial solution (100% allocation)."""
    stocks_df, bench_series = synthetic_price_data
    bench_returns = bench_series.pct_change().dropna()
    single_df = stocks_df[["RELIANCE.NS"]]

    res = optimize_minimum_variance(
        prices=single_df,
        total_capital=50000.0,
        benchmark_returns=bench_returns,
        max_weight=1.0,
    )

    assert res.success is True
    assert len(res.weights) == 1
    assert np.isclose(res.weights[0], 1.0)
    assert res.investments[0] == 50000.0
    assert res.annualized_volatility > 0.0


def test_optimize_minimum_variance_infeasible_setting(synthetic_price_data):
    """Verify optimizer safely handles and reports infeasible constraints."""
    stocks_df, bench_series = synthetic_price_data
    bench_returns = bench_series.pct_change().dropna()
    # 2 assets with 30% cap cannot sum to 100%
    two_assets_df = stocks_df.iloc[:, :2]

    res = optimize_minimum_variance(
        prices=two_assets_df,
        total_capital=100000.0,
        benchmark_returns=bench_returns,
        max_weight=0.30,
    )

    assert res.success is False
    assert "Infeasible" in res.message
