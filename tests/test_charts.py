"""Tests for portfolio risk visualization charts in portfolio_risk.charts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from portfolio_risk.charts import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_historical_growth,
    plot_portfolio_allocation,
    plot_portfolio_comparison_bars,
    plot_stock_volatilities,
)


def test_plot_portfolio_allocation_donut_and_bars():
    """Verify donut for <=6 holdings and horizontal bar chart for >6 holdings."""
    # 1 holding (donut)
    fig1 = plot_portfolio_allocation(["RELIANCE.NS"], [50000.0])
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # 4 holdings (donut)
    tickers4 = ["RELIANCE.NS", "IOC.NS", "ICICIBANK.NS", "HDFCBANK.NS"]
    inv4 = [35000.0, 40000.0, 30000.0, 20000.0]
    fig4 = plot_portfolio_allocation(tickers4, inv4)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    # 8 holdings (transitions to horizontal bar chart)
    tickers8 = [f"STOCK{i}.NS" for i in range(8)]
    inv8 = [10000.0] * 8
    fig8 = plot_portfolio_allocation(tickers8, inv8)
    assert isinstance(fig8, plt.Figure)
    plt.close(fig8)


def test_plot_historical_growth():
    """Verify historical growth chart with normalized prices."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    port_val = pd.Series(np.linspace(100000, 115000, 10), index=dates)
    bench_val = pd.Series(np.linspace(20000, 22000, 10), index=dates)

    fig = plot_historical_growth(port_val, bench_val)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_drawdown():
    """Verify drawdown underwater chart generation."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    port_val = pd.Series([100, 105, 102, 98, 95, 101, 107, 104, 108, 110], index=dates)

    fig = plot_drawdown(port_val)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_stock_volatilities():
    """Verify individual volatilities horizontal bar chart."""
    vols = pd.Series([0.18, 0.22, 0.15, 0.28], index=["A.NS", "B.NS", "C.NS", "D.NS"])
    fig = plot_stock_volatilities(vols)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmap_single_and_multiple():
    """Verify correlation heatmap handles single asset placeholder and multiple assets."""
    # 1 holding (returns placeholder cleanly without error)
    corr_single = pd.DataFrame([[1.0]], index=["A.NS"], columns=["A.NS"])
    fig_single = plot_correlation_heatmap(corr_single)
    assert isinstance(fig_single, plt.Figure)
    plt.close(fig_single)

    # Multiple holdings
    corr = pd.DataFrame(
        [[1.0, 0.45], [0.45, 1.0]],
        index=["A.NS", "B.NS"],
        columns=["A.NS", "B.NS"],
    )
    fig = plot_correlation_heatmap(corr)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_portfolio_comparison_bars():
    """Verify side-by-side metric comparison chart."""
    u = {"volatility": 0.18, "max_drawdown": -0.15, "beta": 1.05, "div_ratio": 1.25}
    e = {"volatility": 0.17, "max_drawdown": -0.14, "beta": 1.00, "div_ratio": 1.28}
    fig = plot_portfolio_comparison_bars(u, e)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
