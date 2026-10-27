"""Visualization module for portfolio risk charts using Matplotlib and Seaborn.

Provides clean, publication-ready financial charts following Zerodha Kite's restrained,
data-dense design philosophy: muted palettes, clean typography, subtle gridlines,
and precise numerical formatting. Supports dynamic holdings (1 to 20 assets)
and three-way portfolio comparisons.
"""

from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from portfolio_risk.config import KITE_PALETTE

# Configure Matplotlib default aesthetic
plt.rcParams["font.sans-serif"] = ["Inter", "Helvetica Neue", "Arial", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#E0E0E0"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#ECECEC"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6


def _apply_kite_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent Zerodha Kite minimalist formatting to a matplotlib axis."""
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E0E0E0")
    ax.spines["bottom"].set_color("#E0E0E0")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.set_title(title, fontsize=12, fontweight="medium", color="#222222", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color="#666666", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color="#666666", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.7, color="#EFEFEF")


def _empty_chart(message: str) -> plt.Figure:
    """Return a clean placeholder figure when input data is missing or empty."""
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10,
        color="#888888",
        transform=ax.transAxes,
    )
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout()
    return fig


def plot_portfolio_allocation(
    tickers: Sequence[str],
    investments: Sequence[float],
) -> plt.Figure:
    """Generate a clean visual chart illustrating nominal capital allocation across holdings.

    To eliminate label overlaps (Section 3.1):
    - For <= 6 holdings: renders a donut chart using a dedicated legend rather than
      crowded slice callouts, with high-contrast percentage labels inside slices.
    - For > 6 holdings: automatically transitions to a horizontal bar chart with
      direct percentage annotations for maximum clarity.

    Args:
        tickers: Stock ticker symbols.
        investments: Nominal investment amounts.

    Returns:
        matplotlib.figure.Figure.
    """
    if not tickers or not investments:
        return _empty_chart("No allocation data available")

    n_holdings = len(tickers)
    total_val = float(np.sum(investments))
    if total_val <= 0:
        return _empty_chart("Total allocation must be positive")

    weights_pct = [(amt / total_val) * 100.0 for amt in investments]

    # Donut chart with legend for 1 to 6 holdings
    if n_holdings <= 6:
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=100)
        fig.patch.set_facecolor("#FFFFFF")

        colors = [
            "#387ED1", "#5C99E0", "#7FAEE8", "#2E6BB3", "#9BBDF0", "#A8C7F0"
        ]
        slice_colors = colors[:n_holdings]

        # Use autopct only if weight is at least 5% to avoid overlapping tiny text
        def autopct_filter(pct: float) -> str:
            return f"{pct:.1f}%" if pct >= 5.0 else ""

        wedges, texts, autotexts = ax.pie(
            investments,
            autopct=autopct_filter,
            startangle=140,
            colors=slice_colors,
            pctdistance=0.72,
            wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=1.5),
        )

        for autotext in autotexts:
            autotext.set_color("#FFFFFF")
            autotext.set_fontsize(8.5)
            autotext.set_weight("medium")

        # Clean external legend with ticker and percentage to prevent overlapping
        legend_labels = [f"{t} ({w:.1f}%)" for t, w in zip(tickers, weights_pct)]
        ax.legend(
            wedges,
            legend_labels,
            title="Holdings",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=8.5,
            title_fontsize=9.0,
        )

        ax.set_title("Capital Allocation Breakdown", fontsize=11.5, color="#222222", pad=10, loc="left")
        fig.tight_layout()
        return fig

    # Horizontal bar chart for 7 to 20 holdings
    fig_height = max(3.5, n_holdings * 0.32)
    fig, ax = plt.subplots(figsize=(5.5, fig_height), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    df_alloc = pd.DataFrame({"Ticker": tickers, "Weight": weights_pct})
    df_alloc = df_alloc.sort_values("Weight", ascending=True)

    bars = ax.barh(
        df_alloc["Ticker"],
        df_alloc["Weight"],
        color=KITE_PALETTE["accent"],
        height=0.55,
    )

    _apply_kite_style(ax, "Capital Allocation (%)", xlabel="Portfolio Weight (%)")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#444444",
        )

    fig.tight_layout()
    return fig


def plot_historical_growth(
    portfolio_value: pd.Series,
    benchmark_prices: Optional[pd.Series] = None,
) -> plt.Figure:
    """Plot historical cumulative portfolio value trajectory indexed to base 100.

    Allows direct comparison with benchmark growth on a normalized scale.

    Args:
        portfolio_value: Daily portfolio capital series.
        benchmark_prices: Synchronized benchmark index price series.

    Returns:
        matplotlib.figure.Figure.
    """
    if portfolio_value.empty:
        return _empty_chart("No historical growth data available")

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    # Normalize to base 100
    norm_portfolio = (portfolio_value / portfolio_value.iloc[0]) * 100.0
    ax.plot(
        norm_portfolio.index,
        norm_portfolio.values,
        label="User Portfolio",
        color=KITE_PALETTE["accent"],
        linewidth=1.8,
    )

    if benchmark_prices is not None and not benchmark_prices.empty:
        norm_benchmark = (benchmark_prices / benchmark_prices.iloc[0]) * 100.0
        ax.plot(
            norm_benchmark.index,
            norm_benchmark.values,
            label="Nifty 50 Benchmark",
            color=KITE_PALETTE["chart_neutral"],
            linewidth=1.2,
            linestyle="--",
        )

    _apply_kite_style(ax, "Historical Growth (Base = 100)", ylabel="Normalized Value")
    ax.axhline(100.0, color="#CCCCCC", linestyle=":", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


def plot_drawdown(portfolio_value: pd.Series) -> plt.Figure:
    """Plot historical peak-to-trough drawdown (underwater chart) in percentage.

    Args:
        portfolio_value: Daily portfolio capital series.

    Returns:
        matplotlib.figure.Figure.
    """
    if portfolio_value.empty:
        return _empty_chart("No drawdown data available")

    running_max = portfolio_value.cummax()
    drawdown = ((portfolio_value - running_max) / running_max) * 100.0

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    ax.plot(drawdown.index, drawdown.values, color=KITE_PALETTE["negative"], linewidth=1.2)
    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color=KITE_PALETTE["negative"],
        alpha=0.15,
    )

    _apply_kite_style(ax, "Historical Underwater Drawdown (%)", ylabel="Drawdown (%)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_stock_volatilities(individual_volatilities: pd.Series) -> plt.Figure:
    """Generate a horizontal bar chart comparing individual stock annualized volatilities.

    Args:
        individual_volatilities: pd.Series indexed by ticker with annualized volatility decimals.

    Returns:
        matplotlib.figure.Figure.
    """
    if individual_volatilities.empty:
        return _empty_chart("No volatility data available")

    vols_pct = individual_volatilities * 100.0
    sorted_vols = vols_pct.sort_values(ascending=True)

    n_bars = len(sorted_vols)
    fig_height = max(3.0, n_bars * 0.38)

    fig, ax = plt.subplots(figsize=(7, fig_height), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    bars = ax.barh(
        sorted_vols.index,
        sorted_vols.values,
        color=KITE_PALETTE["accent"],
        height=0.45,
    )

    _apply_kite_style(ax, "Individual Annualized Volatility (%)", xlabel="Annualized Volatility (%)")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#444444",
        )

    # Add educational reference line at 25% (High Risk boundary)
    ax.axvline(25.0, color="#FFA000", linestyle=":", linewidth=1.0, alpha=0.8, label="25% High Risk Boundary")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> plt.Figure:
    """Plot correlation heatmap of daily equity returns with clean numerical annotations.

    If the portfolio contains only 1 holding, returns an educational placeholder explaining
    that cross-stock correlation is not applicable.

    Args:
        correlation_matrix: Symmetric Pearson correlation DataFrame.

    Returns:
        matplotlib.figure.Figure.
    """
    if correlation_matrix.empty or correlation_matrix.shape[0] <= 1:
        return _empty_chart("Cross-stock correlation is not applicable for a single holding.")

    n_tickers = correlation_matrix.shape[0]
    fig_dim = max(4.5, n_tickers * 0.65)
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim * 0.8), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    # Restrained diverging colormap (Kite blue to neutral to subtle warm)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        cbar=True,
        square=True,
        ax=ax,
        linewidths=1.0,
        linecolor="#FFFFFF",
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"},
        annot_kws={"size": 9.0 if n_tickers <= 6 else 7.5, "color": "#222222"},
    )

    ax.set_title("Inter-Stock Correlation Matrix", fontsize=11.5, color="#222222", pad=10, loc="left")
    ax.tick_params(axis="both", which="both", length=0, labelsize=8.5)
    fig.tight_layout()
    return fig


def plot_portfolio_comparison_bars(
    user_metrics: Dict[str, float],
    ew_metrics: Dict[str, float],
    opt_metrics: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Compare key risk metrics among User, Equal-Weight, and Minimum-Variance portfolios.

    Args:
        user_metrics: Dict containing 'volatility', 'max_drawdown', 'beta', 'div_ratio'.
        ew_metrics: Matching dict for Equal-Weight portfolio.
        opt_metrics: Optional matching dict for Minimum-Variance portfolio.

    Returns:
        matplotlib.figure.Figure.
    """
    metrics = ["Vol (%)", "Max DD (%)", "Beta", "Div Ratio"]
    user_vals = [
        user_metrics.get("volatility", 0.0) * 100.0,
        abs(user_metrics.get("max_drawdown", 0.0)) * 100.0,
        user_metrics.get("beta", 0.0),
        user_metrics.get("div_ratio", 1.0),
    ]
    ew_vals = [
        ew_metrics.get("volatility", 0.0) * 100.0,
        abs(ew_metrics.get("max_drawdown", 0.0)) * 100.0,
        ew_metrics.get("beta", 0.0),
        ew_metrics.get("div_ratio", 1.0),
    ]

    has_opt = opt_metrics is not None and opt_metrics.get("volatility") is not None
    if has_opt:
        opt_vals = [
            opt_metrics.get("volatility", 0.0) * 100.0,
            abs(opt_metrics.get("max_drawdown", 0.0)) * 100.0,
            opt_metrics.get("beta", 0.0),
            opt_metrics.get("div_ratio", 1.0),
        ]

    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    if has_opt:
        width = 0.25
        bars1 = ax.bar(x - width, user_vals, width, label="User Portfolio", color=KITE_PALETTE["accent"])
        bars2 = ax.bar(x, ew_vals, width, label="Equal-Weight Baseline", color=KITE_PALETTE["chart_secondary"])
        bars3 = ax.bar(x + width, opt_vals, width, label="Min-Variance Allocation", color=KITE_PALETTE["positive"])
        all_bars = list(bars1) + list(bars2) + list(bars3)
    else:
        width = 0.32
        bars1 = ax.bar(x - width / 2, user_vals, width, label="User Portfolio", color=KITE_PALETTE["accent"])
        bars2 = ax.bar(x + width / 2, ew_vals, width, label="Equal-Weight Baseline", color=KITE_PALETTE["chart_secondary"])
        all_bars = list(bars1) + list(bars2)

    _apply_kite_style(ax, "Portfolio Risk Metrics Comparison", ylabel="Metric Magnitude")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Add numeric value labels on bars
    for bar in all_bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#555555",
            )

    fig.tight_layout()
    return fig
