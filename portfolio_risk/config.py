"""Configuration settings and constants for the Portfolio Risk Analysis application.

This module contains default parameters, financial constants, threshold definitions,
and styling constants used throughout the portfolio risk engine and user interface.
"""

from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Default Portfolio Settings
# -----------------------------------------------------------------------------
# Default 4 editable NSE holdings representing diverse sectors
DEFAULT_TICKERS: List[str] = [
    "RELIANCE.NS",
    "IOC.NS",
    "ICICIBANK.NS",
    "HDFCBANK.NS",
]

# Default investment amounts in Indian Rupees (INR) totaling ₹1,25,000
DEFAULT_INVESTMENTS: List[float] = [
    35000.0,
    40000.0,
    30000.0,
    20000.0,
]

# Benchmark index for Indian equity market (Nifty 50 on Yahoo Finance)
BENCHMARK_TICKER: str = "^NSEI"
BENCHMARK_NAME: str = "Nifty 50"

# Historical analysis periods (5, 10, 15 years as specified in Section 4.11)
DEFAULT_PERIOD_YEARS: int = 5
SUPPORTED_PERIODS: Dict[str, str] = {
    "5 Years": "5y",
    "10 Years": "10y",
    "15 Years": "15y",
}

# -----------------------------------------------------------------------------
# Financial Engineering Constants
# -----------------------------------------------------------------------------
# Standard number of trading days in an Indian calendar year for annualizing metrics
TRADING_DAYS_PER_YEAR: int = 252

# Input constraints: fully dynamic holdings from 1 up to configurable MAX_HOLDINGS
MIN_HOLDINGS: int = 1
MAX_HOLDINGS: int = 20
MIN_INVESTMENT_PER_STOCK: float = 100.0

# Central configuration for Minimum-Variance Portfolio Optimisation
# Defines default maximum weight per individual asset (40% cap)
DEFAULT_MAX_OPTIMAL_WEIGHT: float = 0.40

# -----------------------------------------------------------------------------
# Risk Classification Thresholds
# -----------------------------------------------------------------------------
# Educational annualized volatility thresholds
# Low: < 15%
# Moderate: 15% - 25%
# High: >= 25%
VOLATILITY_THRESHOLDS: Dict[str, float] = {
    "LOW_MAX": 0.15,
    "MODERATE_MAX": 0.25,
}

# Herfindahl-Hirschman Index (HHI) concentration interpretation thresholds
# HHI = sum(w_i^2) where w_i is portfolio weight (0 to 1).
# Low Concentration: < 0.20
# Moderate Concentration: 0.20 - 0.35
# High Concentration: > 0.35 (e.g. 1.0 for single asset)
HHI_THRESHOLDS: Dict[str, float] = {
    "LOW_CONCENTRATION": 0.20,
    "MODERATE_CONCENTRATION": 0.35,
}

# -----------------------------------------------------------------------------
# Stress Testing Predefined Market-Wide Scenarios
# -----------------------------------------------------------------------------
# Predefined market-wide drawdowns: 5%, 10%, 15%, 20%
PREDEFINED_MARKET_SHOCKS: List[Dict[str, object]] = [
    {
        "id": "scenario_5pct",
        "name": "Mild Market Pullback (-5%)",
        "description": "Hypothetical uniform 5% decline across all portfolio holdings.",
        "shock_pct": -0.05,
    },
    {
        "id": "scenario_10pct",
        "name": "Broad Market Correction (-10%)",
        "description": "Hypothetical uniform 10% decline across all portfolio holdings.",
        "shock_pct": -0.10,
    },
    {
        "id": "scenario_15pct",
        "name": "Deep Market Correction (-15%)",
        "description": "Hypothetical uniform 15% decline across all portfolio holdings.",
        "shock_pct": -0.15,
    },
    {
        "id": "scenario_20pct",
        "name": "Severe Market Crash (-20%)",
        "description": "Hypothetical uniform 20% decline across all portfolio holdings.",
        "shock_pct": -0.20,
    },
]

# -----------------------------------------------------------------------------
# Zerodha Kite-Inspired Visual Palette
# -----------------------------------------------------------------------------
KITE_PALETTE = {
    "accent": "#387ED1",        # Kite signature blue
    "background": "#FFFFFF",    # Canvas white
    "surface": "#F9F9F9",       # Clean light surface
    "text_primary": "#333333",  # Deep charcoal primary text
    "text_secondary": "#777777",# Muted secondary text
    "border": "#E0E0E0",        # Subtle 1px borders
    "hover": "#F5F5F5",         # Soft hover gray
    "positive": "#00C853",      # Profit / positive green
    "negative": "#FF3D00",      # Loss / negative red
    "muted": "#999999",         # Light divider text
    "chart_bar": "#387ED1",     # Primary chart color
    "chart_secondary": "#E08038", # Comparison / warm secondary
    "chart_neutral": "#6C7A89", # Neutral series
    "chart_grid": "#ECECEC",    # Subtle chart grid lines
}
