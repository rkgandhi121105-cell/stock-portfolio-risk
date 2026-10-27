"""Stock Portfolio Risk Analysis and Stress Testing Application.

A financial analysis terminal inspired by Zerodha Kite's restrained, functional design.
Calculates historical risk metrics, concentration, correlation, diversification,
minimum-variance allocation optimization via SciPy, and instantaneous stress scenarios
for National Stock Exchange (NSE) equity portfolios.
"""

from datetime import datetime
from typing import Dict, List, Tuple
import uuid
import pandas as pd
import streamlit as st

from portfolio_risk.charts import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_historical_growth,
    plot_portfolio_allocation,
    plot_portfolio_comparison_bars,
    plot_stock_volatilities,
)
from portfolio_risk.config import (
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    DEFAULT_INVESTMENTS,
    DEFAULT_MAX_OPTIMAL_WEIGHT,
    DEFAULT_PERIOD_YEARS,
    DEFAULT_TICKERS,
    MAX_HOLDINGS,
    MIN_HOLDINGS,
    SUPPORTED_PERIODS,
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_THRESHOLDS,
)
from portfolio_risk.data import (
    DataFetchError,
    InsufficientDataError,
    fetch_portfolio_and_benchmark,
)
from portfolio_risk.export import (
    generate_csv_report,
    generate_excel_report,
)
from portfolio_risk.metrics import (
    allocate_whole_rupees,
    analyze_portfolio,
    classify_risk,
)
from portfolio_risk.optimization import (
    OptimizationResult,
    optimize_minimum_variance,
)
from portfolio_risk.stress_testing import (
    format_stress_results_table,
    run_stress_testing,
)
from portfolio_risk.validation import (
    PortfolioValidationError,
    normalize_ticker,
    validate_portfolio_inputs,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Risk Terminal | NSE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def format_inr(amount: float) -> str:
    """Format a monetary value into Indian Rupee (INR) representation.

    Follows Indian numbering conventions (lakhs and crores):
    e.g. 100000 -> ₹1,00,000.00, 1234567 -> ₹12,34,567.00.
    """
    if pd.isna(amount):
        return "₹0.00"

    sign = "-" if amount < 0 else ""
    abs_amt = abs(amount)

    parts = f"{abs_amt:.2f}".split(".")
    integer_part = parts[0]
    decimal_part = parts[1]

    if len(integer_part) <= 3:
        formatted_int = integer_part
    else:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_int = ",".join(groups) + "," + last_three

    return f"{sign}₹{formatted_int}.{decimal_part}"


# -----------------------------------------------------------------------------
# Zerodha Kite CSS Design System
# -----------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject custom CSS enforcing Zerodha Kite's restrained terminal aesthetic."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

        /* Selective Font & Spacing Reset (Preserves Material/Streamlit Icon Glyphs) */
        html, body, p, label, .stMarkdown p, .stMarkdown span {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-weight: 400;
            color: #333333;
        }

        /* -----------------------------------------------------------------
           Sidebar Toggle Icon Fix: Replace ligature text with SVG icons
           ----------------------------------------------------------------- */
        /* Open sidebar: close button ('X' icon) */
        [data-testid="stSidebarCollapseButton"] button span,
        [data-testid="stSidebarCollapseButton"] button svg {
            display: none !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 28px !important;
            height: 28px !important;
            border-radius: 4px !important;
            border: 1px solid #E0E0E0 !important;
            background-color: #FFFFFF !important;
        }
        [data-testid="stSidebarCollapseButton"] button::after {
            content: "";
            display: inline-block;
            width: 14px;
            height: 14px;
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23444444' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'%3E%3C/line%3E%3Cline x1='6' y1='6' x2='18' y2='18'%3E%3C/line%3E%3C/svg%3E");
        }

        /* Closed sidebar: collapsed control button (Hamburger Menu icon) */
        [data-testid="stSidebarCollapsedControl"] button span,
        [data-testid="stSidebarCollapsedControl"] button svg,
        [data-testid="collapsedControl"] button span,
        [data-testid="collapsedControl"] button svg {
            display: none !important;
        }
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 32px !important;
            height: 32px !important;
            border-radius: 4px !important;
            border: 1px solid #E0E0E0 !important;
            background-color: #FFFFFF !important;
        }
        [data-testid="stSidebarCollapsedControl"] button::after,
        [data-testid="collapsedControl"] button::after {
            content: "";
            display: inline-block;
            width: 16px;
            height: 16px;
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23333333' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='3' y1='12' x2='21' y2='12'%3E%3C/line%3E%3Cline x1='3' y1='6' x2='21' y2='6'%3E%3C/line%3E%3Cline x1='3' y1='18' x2='21' y2='18'%3E%3C/line%3E%3C/svg%3E");
        }

        /* Top header container */
        .terminal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 18px;
            background-color: #FFFFFF;
            border-bottom: 1px solid #E0E0E0;
            margin-bottom: 18px;
        }
        .terminal-title {
            font-size: 19px;
            font-weight: 500;
            color: #222222;
            letter-spacing: -0.2px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .terminal-badge {
            background-color: #F0F4F8;
            color: #387ED1;
            font-size: 11px;
            font-weight: 500;
            padding: 3px 8px;
            border-radius: 3px;
            border: 1px solid #D6E4F0;
        }
        .terminal-meta {
            font-size: 12px;
            color: #777777;
        }

        /* Metric Cards: Standardized, uniform heights & tabular formatting */
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 4px;
            padding: 12px 14px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            min-height: 98px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .metric-label {
            font-size: 11px;
            color: #777777;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
            font-weight: 500;
            white-space: nowrap;
        }
        .metric-value {
            font-size: 19px;
            font-weight: 500;
            color: #222222;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-sub {
            font-size: 11px;
            color: #888888;
            margin-top: 4px;
            white-space: nowrap;
        }

        /* Risk Badges */
        .badge-low {
            background-color: #E8F5E9;
            color: #2E7D32;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid #C8E6C9;
        }
        .badge-moderate {
            background-color: #FFF8E1;
            color: #F57F17;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid #FFE082;
        }
        .badge-high {
            background-color: #FFEBEE;
            color: #C62828;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
            border: 1px solid #FFCDD2;
        }

        /* Panel Headings and Dividers */
        .panel-heading {
            font-size: 14.5px;
            font-weight: 500;
            color: #222222;
            margin-bottom: 6px;
        }
        .panel-divider {
            border: none;
            border-top: 1px solid #EFEFEF;
            margin-top: 0px;
            margin-bottom: 14px;
        }
        .panel-note {
            font-size: 11.5px;
            color: #777777;
            margin-top: 10px;
            line-height: 1.5;
        }

        /* Button Styling */
        /* Primary / Blue Buttons (Both st.button and st.download_button) */
        button[data-testid="baseButton-primary"],
        button[kind="primary"],
        .stButton button[kind="primary"],
        .stButton button[data-testid="baseButton-primary"],
        .stDownloadButton button[kind="primary"],
        .stDownloadButton button[data-testid="baseButton-primary"],
        div[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"] {
            background-color: #387ED1 !important;
            color: #FFFFFF !important;
            border: 1px solid #387ED1 !important;
            border-radius: 4px !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            transition: background-color 150ms ease !important;
        }

        /* Enforce white text on all inner text nodes (p, span, div) within blue buttons */
        button[data-testid="baseButton-primary"] *,
        button[kind="primary"] *,
        .stButton button[kind="primary"] *,
        .stButton button[data-testid="baseButton-primary"] *,
        .stDownloadButton button[kind="primary"] *,
        .stDownloadButton button[data-testid="baseButton-primary"] *,
        div[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"] *,
        div[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"] p,
        div[data-testid="stDownloadButton"] button[data-testid="baseButton-primary"] span,
        button[data-testid="baseButton-primary"] p,
        button[data-testid="baseButton-primary"] span,
        button[data-testid="baseButton-primary"] div {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        /* Hover and focus states for blue buttons */
        button[data-testid="baseButton-primary"]:hover,
        button[data-testid="baseButton-primary"]:focus,
        button[data-testid="baseButton-primary"]:active,
        button[kind="primary"]:hover,
        button[kind="primary"]:focus,
        button[kind="primary"]:active,
        .stButton button[kind="primary"]:hover,
        .stButton button[data-testid="baseButton-primary"]:hover,
        .stDownloadButton button[kind="primary"]:hover,
        .stDownloadButton button[data-testid="baseButton-primary"]:hover {
            background-color: #2E6BB3 !important;
            border-color: #2E6BB3 !important;
            color: #FFFFFF !important;
        }

        button[data-testid="baseButton-primary"]:hover *,
        button[data-testid="baseButton-primary"]:focus *,
        button[data-testid="baseButton-primary"]:active *,
        button[kind="primary"]:hover *,
        .stDownloadButton button[data-testid="baseButton-primary"]:hover * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        /* Secondary Buttons */
        .stButton button[kind="secondary"],
        .stButton button[data-testid="baseButton-secondary"],
        .stDownloadButton button[kind="secondary"],
        .stDownloadButton button[data-testid="baseButton-secondary"] {
            background-color: #FFFFFF !important;
            color: #333333 !important;
            border: 1px solid #D0D0D0 !important;
            border-radius: 4px !important;
            font-size: 13px !important;
        }
        .stButton button[kind="secondary"]:hover,
        .stButton button[data-testid="baseButton-secondary"]:hover,
        .stDownloadButton button[kind="secondary"]:hover,
        .stDownloadButton button[data-testid="baseButton-secondary"]:hover {
            background-color: #F5F5F5 !important;
            border-color: #B0B0B0 !important;
        }

        /* Streamlit Input Cleanups */
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
            border: 1px solid #D0D0D0 !important;
            border-radius: 4px !important;
            font-size: 13px !important;
            padding: 6px 10px !important;
        }
        div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {
            border-color: #387ED1 !important;
            box-shadow: 0 0 0 1px #387ED1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Cached Market Data Retrieval
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data(
    tickers_tuple: Tuple[str, ...],
    period: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Retrieve and cache market price history for stock basket and benchmark."""
    return fetch_portfolio_and_benchmark(
        tickers=list(tickers_tuple),
        benchmark_ticker=BENCHMARK_TICKER,
        period=period,
    )


# -----------------------------------------------------------------------------
# Dynamic Holdings State Helpers
# -----------------------------------------------------------------------------
def initialize_holdings_state() -> None:
    """Initialize session state holding entries with stable unique IDs."""
    if "holdings" not in st.session_state:
        st.session_state.holdings = [
            {"id": f"h_{i}", "ticker": DEFAULT_TICKERS[i], "amount": DEFAULT_INVESTMENTS[i]}
            for i in range(len(DEFAULT_TICKERS))
        ]
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "last_retrieval_time" not in st.session_state:
        st.session_state.last_retrieval_time = None
    if "selected_period" not in st.session_state:
        st.session_state.selected_period = "5y"
    if "max_optimal_weight" not in st.session_state:
        st.session_state.max_optimal_weight = DEFAULT_MAX_OPTIMAL_WEIGHT
    if "stress_custom_pct" not in st.session_state:
        st.session_state.stress_custom_pct = 12.0
    if "stress_stock_pct" not in st.session_state:
        st.session_state.stress_stock_pct = 30.0


def add_holding_callback() -> None:
    """Add a new empty holding row up to MAX_HOLDINGS."""
    if len(st.session_state.holdings) < MAX_HOLDINGS:
        new_id = f"h_{uuid.uuid4().hex[:6]}"
        st.session_state.holdings.append({
            "id": new_id,
            "ticker": "",
            "amount": 10000.0,
        })


def remove_holding_callback(holding_id: str) -> None:
    """Remove a holding row, ensuring at least MIN_HOLDINGS remains."""
    if len(st.session_state.holdings) > MIN_HOLDINGS:
        st.session_state.holdings = [
            h for h in st.session_state.holdings if h["id"] != holding_id
        ]
        st.session_state.pop(f"ticker_{holding_id}", None)
        st.session_state.pop(f"amt_{holding_id}", None)


def reset_holdings_callback() -> None:
    """Restore default 4 holdings and clear calculation cache."""
    st.session_state.holdings = [
        {"id": f"h_{i}", "ticker": DEFAULT_TICKERS[i], "amount": DEFAULT_INVESTMENTS[i]}
        for i in range(len(DEFAULT_TICKERS))
    ]
    st.session_state.analysis_result = None
    st.session_state.last_retrieval_time = None
    st.session_state.max_optimal_weight = DEFAULT_MAX_OPTIMAL_WEIGHT
    st.cache_data.clear()


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()
    initialize_holdings_state()

    # -------------------------------------------------------------------------
    # Sidebar: Dynamic Holdings Editor & Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding-bottom: 8px; margin-bottom: 12px; border-bottom: 1px solid #E0E0E0;">
                <div style="font-size: 16px; font-weight: 500; color: #222222;">Portfolio Inputs</div>
                <div style="font-size: 11.5px; color: #777777;">
                    Dynamic basket: {MIN_HOLDINGS} to {MAX_HOLDINGS} NSE stocks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Historical Lookback Period Selector (5, 10, 15 Years)
        period_label = st.selectbox(
            "Historical Period",
            options=list(SUPPORTED_PERIODS.keys()),
            index=0,  # Default: 5 Years
            help="Historical period for synchronized adjusted closing price history.",
        )
        period_val = SUPPORTED_PERIODS[period_label]
        st.session_state.selected_period = period_val

        # Maximum Stock Weight Cap for Optimizer
        opt_cap_pct = st.number_input(
            "Optimizer Max Weight Cap (%)",
            min_value=5.0,
            max_value=100.0,
            value=float(st.session_state.max_optimal_weight * 100.0),
            step=5.0,
            help="Configurable maximum allocation ceiling per stock for Minimum-Variance optimization.",
        )
        st.session_state.max_optimal_weight = float(opt_cap_pct / 100.0)

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 6px;">
                <span style="font-size: 12px; font-weight: 500; color: #555555;">Holdings ({len(st.session_state.holdings)}/{MAX_HOLDINGS})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_tickers: List[str] = []
        current_investments: List[float] = []
        holding_to_remove: str = ""

        # Render dynamic holding rows with stable keys
        for idx, holding in enumerate(st.session_state.holdings):
            h_id = holding["id"]
            ticker_key = f"ticker_{h_id}"
            amt_key = f"amt_{h_id}"

            col_t, col_a, col_d = st.columns([1.3, 1.3, 0.45])

            with col_t:
                val_t = st.text_input(
                    f"Ticker {idx+1}",
                    value=holding["ticker"],
                    key=ticker_key,
                    label_visibility="collapsed",
                    placeholder="TICKER.NS",
                )
                holding["ticker"] = val_t
                current_tickers.append(val_t)

            with col_a:
                val_a = st.number_input(
                    f"Amount {idx+1}",
                    value=float(holding["amount"]),
                    min_value=100.0,
                    step=5000.0,
                    key=amt_key,
                    label_visibility="collapsed",
                    format="%.0f",
                )
                holding["amount"] = val_a
                current_investments.append(val_a)

            with col_d:
                can_remove = len(st.session_state.holdings) > MIN_HOLDINGS
                if st.button(
                    "✕",
                    key=f"btn_del_{h_id}",
                    disabled=not can_remove,
                    help="Remove holding" if can_remove else "Minimum 1 holding required",
                ):
                    holding_to_remove = h_id

        if holding_to_remove:
            remove_holding_callback(holding_to_remove)
            st.rerun()

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        at_max = len(st.session_state.holdings) >= MAX_HOLDINGS
        if st.button(
            "+ Add Holding",
            type="secondary",
            use_container_width=True,
            disabled=at_max,
            help="Maximum holdings limit reached" if at_max else "Add another holding",
        ):
            add_holding_callback()
            st.rerun()

        if at_max:
            st.caption(f"Reached configured maximum limit of {MAX_HOLDINGS} holdings.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        analyze_clicked = st.button("Analyze Portfolio", type="primary", use_container_width=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Reset Defaults", type="secondary", use_container_width=True):
                reset_holdings_callback()
                st.rerun()
        with col_r2:
            if st.button("Refresh Data", type="secondary", use_container_width=True):
                st.cache_data.clear()
                analyze_clicked = True

        st.markdown(
            f"""
            <div style="font-size: 11px; color: #888888; line-height: 1.5; border-top: 1px solid #EFEFEF; padding-top: 12px; margin-top: 14px;">
                <strong>Benchmark:</strong> {BENCHMARK_NAME} (<code>{BENCHMARK_TICKER}</code>)<br>
                <strong>Annual Trading Days:</strong> {TRADING_DAYS_PER_YEAR}<br>
                <strong>Pricing Model:</strong> Buy-and-Hold Valuation
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Analysis Trigger & Validation
    # -------------------------------------------------------------------------
    if analyze_clicked or st.session_state.analysis_result is None:
        try:
            clean_tickers, clean_investments = validate_portfolio_inputs(
                current_tickers, current_investments
            )

            with st.spinner("Synchronizing market history, computing risk analytics & optimizer..."):
                stock_prices, benchmark_prices = load_market_data(
                    tuple(clean_tickers),
                    st.session_state.selected_period,
                )

                # Execute quantitative metrics pipeline
                analysis = analyze_portfolio(
                    prices=stock_prices,
                    investments=clean_investments,
                    benchmark_prices=benchmark_prices,
                )

                # Execute SciPy Minimum-Variance portfolio optimization
                total_cap = float(sum(clean_investments))
                opt_result = optimize_minimum_variance(
                    prices=stock_prices,
                    total_capital=total_cap,
                    benchmark_returns=analysis["benchmark_returns"],
                    max_weight=st.session_state.max_optimal_weight,
                )

                # Execute instantaneous stress test scenarios
                stress_results = run_stress_testing(
                    tickers=clean_tickers,
                    investments=clean_investments,
                    selected_single_ticker=clean_tickers[0],
                    single_stock_shock_pct=-abs(st.session_state.stress_stock_pct / 100.0),
                    custom_market_shock_pct=-abs(st.session_state.stress_custom_pct / 100.0),
                )

                st.session_state.analysis_result = {
                    "metrics": analysis,
                    "opt_result": opt_result,
                    "stress": stress_results,
                    "tickers": clean_tickers,
                    "investments": clean_investments,
                }
                st.session_state.last_retrieval_time = datetime.now().strftime("%d %b %Y, %H:%M:%S IST")

        except PortfolioValidationError as pve:
            st.error(f"Portfolio Input Error: {pve}")
            return
        except DataFetchError as dfe:
            st.error(f"Market Data Error: {dfe}")
            return
        except Exception as exc:
            st.error(f"Calculation Error: {exc}")
            return

    # Extract active analysis data
    res = st.session_state.analysis_result
    if res is None:
        return

    analysis_data = res["metrics"]
    opt_result = res["opt_result"]
    stress_results = res["stress"]
    summary = analysis_data["summary"]

    # -------------------------------------------------------------------------
    # Header Bar
    # -------------------------------------------------------------------------
    retrieval_meta = (
        f"Synchronized: {summary.actual_start_date} to {summary.actual_end_date} | "
        f"Period: {period_label} | Refreshed: {st.session_state.last_retrieval_time}"
        if st.session_state.last_retrieval_time
        else f"Period: {period_label}"
    )

    st.markdown(
        f"""
        <div class="terminal-header">
            <div>
                <div class="terminal-title">
                    Stock Portfolio Risk & Stress Terminal
                    <span class="terminal-badge">NSE CASH</span>
                </div>
                <div class="terminal-meta">{retrieval_meta}</div>
            </div>
            <div>
                <span class="terminal-badge">MODEL: BUY-AND-HOLD</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Panel A: Portfolio Summary (5 Core Metric Cards + Beta & DR)
    # -------------------------------------------------------------------------
    risk_class = summary.risk_classification
    badge_class = (
        "badge-low" if risk_class == "Low" else "badge-moderate" if risk_class == "Moderate" else "badge-high"
    )

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label" title="Total initial capital allocated across all holdings">Total Investment</div>
                <div class="metric-value">{format_inr(summary.total_investment)}</div>
                <div class="metric-sub">{summary.num_holdings} {'Holding' if summary.num_holdings == 1 else 'Holdings'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label" title="Historical annualized price fluctuation (Daily StdDev × √252)">Annualized Volatility</div>
                <div class="metric-value">{summary.annualized_volatility * 100:.2f}%</div>
                <div class="metric-sub">Class: <span class="{badge_class}">{risk_class} Risk</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label" title="Maximum historical peak-to-trough capital decline">Max Drawdown</div>
                <div class="metric-value" style="color: #FF3D00;">{summary.max_drawdown * 100:.2f}%</div>
                <div class="metric-sub">Peak-to-Trough Loss</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label" title="Stock with the highest monetary capital weight">Largest Holding</div>
                <div class="metric-value">{summary.largest_holding_ticker}</div>
                <div class="metric-sub">{summary.largest_holding_pct:.1f}% ({format_inr(summary.largest_holding_amount)})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m5:
        hhi_str = f"{summary.hhi:.3f}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label" title="Herfindahl-Hirschman Index: sum of squared asset weights">Concentration (HHI)</div>
                <div class="metric-value">{hhi_str}</div>
                <div class="metric-sub">{analysis_data['hhi_label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Panel B: Holdings Allocation & Stock-Level Metrics (Native DataFrame + Chart)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="panel-heading">Holdings Allocation & Stock Metrics</div>', unsafe_allow_html=True)
        st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

        col_alloc1, col_alloc2 = st.columns([1.15, 0.85])

        with col_alloc1:
            raw_holdings = analysis_data["holdings_df"]
            display_holdings = pd.DataFrame({
                "Ticker": raw_holdings["Ticker"],
                "Investment (₹)": raw_holdings["Investment (₹)"].apply(format_inr),
                "Weight (%)": raw_holdings["Weight (%)"].apply(lambda v: f"{v:.2f}%"),
                "Annualized Volatility (%)": raw_holdings["Annualized Volatility (%)"].apply(
                    lambda v: f"{v:.2f}%" if not pd.isna(v) else "N/A"
                ),
                "Beta (^NSEI)": raw_holdings["Beta (^NSEI)"].apply(
                    lambda v: f"{v:.2f}" if not pd.isna(v) else "N/A"
                ),
                "CAGR (%)": raw_holdings["CAGR (%)"].apply(
                    lambda v: f"{v:+.2f}%" if not pd.isna(v) else "N/A"
                ),
            })
            st.dataframe(
                display_holdings,
                use_container_width=True,
                hide_index=True,
            )

        with col_alloc2:
            fig_alloc = plot_portfolio_allocation(res["tickers"], res["investments"])
            st.pyplot(fig_alloc)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Panel C: Historical Risk Analytics & Charts (2x2 Grid)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="panel-heading">Historical Risk & Performance Trajectory</div>', unsafe_allow_html=True)
        st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_growth = plot_historical_growth(
                analysis_data["portfolio_value"],
                analysis_data["benchmark_prices"],
            )
            st.pyplot(fig_growth)
        with col_c2:
            fig_dd = plot_drawdown(analysis_data["portfolio_value"])
            st.pyplot(fig_dd)

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig_vols = plot_stock_volatilities(analysis_data["individual_volatilities"])
            st.pyplot(fig_vols)
        with col_c4:
            fig_corr = plot_correlation_heatmap(analysis_data["correlation_matrix"])
            st.pyplot(fig_corr)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Panel D: Portfolio Comparison & Optimization (User vs Equal vs Min-Variance)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading">Portfolio Allocation Comparison & Minimum-Variance Optimization</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size: 12px; color: #666666; margin-bottom: 12px; line-height: 1.5;">
                <em>Methodology Note:</em> The minimum-variance allocation uses historical return relationships to find weights that minimise estimated portfolio variance, subject to the configured constraints. It does not predict future prices or guarantee higher returns or lower future losses.
            </div>
            """,
            unsafe_allow_html=True,
        )

        ew = analysis_data["equal_weight"]

        # 1. Stock-Level Allocation Table
        if opt_result.success:
            mv_investments = opt_result.investments
            mv_weights_str = [f"{w*100:.2f}%" for w in opt_result.weights]
            mv_amounts_str = [format_inr(amt) for amt in opt_result.investments]
        else:
            mv_weights_str = ["N/A"] * len(res["tickers"])
            mv_amounts_str = ["Infeasible"] * len(res["tickers"])

        stock_comp_df = pd.DataFrame({
            "Stock": res["tickers"],
            "Original Amount": [format_inr(amt) for amt in res["investments"]],
            "Original Weight": [f"{w*100:.2f}%" for w in analysis_data["weights"]],
            "Equal-Weight Amount": [format_inr(amt) for amt in ew["investments"]],
            "Equal Weight": [f"{w*100:.2f}%" for w in ew["weights"]],
            "Min-Variance Amount": mv_amounts_str,
            "Min-Variance Weight": mv_weights_str,
        })
        st.dataframe(stock_comp_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # 2. Portfolio-Level Risk Metrics Comparison Table & Chart
        comp_col1, comp_col2 = st.columns([1.1, 0.9])
        with comp_col1:
            risk_metrics_rows = [
                {
                    "Metric": "Total Investment",
                    "User Portfolio": format_inr(summary.total_investment),
                    "Equal-Weight Baseline": format_inr(summary.total_investment),
                    "Min-Variance Portfolio": format_inr(summary.total_investment),
                },
                {
                    "Metric": "Annualized Volatility",
                    "User Portfolio": f"{summary.annualized_volatility * 100:.2f}%",
                    "Equal-Weight Baseline": f"{ew['annualized_volatility'] * 100:.2f}%",
                    "Min-Variance Portfolio": f"{opt_result.annualized_volatility * 100:.2f}%" if opt_result.success else "N/A",
                },
                {
                    "Metric": "Maximum Drawdown",
                    "User Portfolio": f"{summary.max_drawdown * 100:.2f}%",
                    "Equal-Weight Baseline": f"{ew['max_drawdown'] * 100:.2f}%",
                    "Min-Variance Portfolio": f"{opt_result.max_drawdown * 100:.2f}%" if opt_result.success else "N/A",
                },
                {
                    "Metric": "CAGR",
                    "User Portfolio": f"{summary.cagr * 100:+.2f}%" if not pd.isna(summary.cagr) else "N/A",
                    "Equal-Weight Baseline": f"{ew['cagr'] * 100:+.2f}%" if not pd.isna(ew['cagr']) else "N/A",
                    "Min-Variance Portfolio": f"{opt_result.cagr * 100:+.2f}%" if (opt_result.success and not pd.isna(opt_result.cagr)) else "N/A",
                },
                {
                    "Metric": "Portfolio Beta (^NSEI)",
                    "User Portfolio": f"{summary.beta:.2f}" if not pd.isna(summary.beta) else "N/A",
                    "Equal-Weight Baseline": f"{ew['beta']:.2f}" if not pd.isna(ew['beta']) else "N/A",
                    "Min-Variance Portfolio": f"{opt_result.beta:.2f}" if (opt_result.success and not pd.isna(opt_result.beta)) else "N/A",
                },
                {
                    "Metric": "Concentration (HHI)",
                    "User Portfolio": f"{summary.hhi:.3f}",
                    "Equal-Weight Baseline": f"{ew['hhi']:.3f}",
                    "Min-Variance Portfolio": f"{opt_result.hhi:.3f}" if opt_result.success else "N/A",
                },
                {
                    "Metric": "Diversification Ratio",
                    "User Portfolio": f"{summary.diversification_ratio:.2f}x",
                    "Equal-Weight Baseline": f"{ew['diversification_ratio']:.2f}x",
                    "Min-Variance Portfolio": f"{opt_result.diversification_ratio:.2f}x" if opt_result.success else "N/A",
                },
                {
                    "Metric": "Risk Classification",
                    "User Portfolio": summary.risk_classification,
                    "Equal-Weight Baseline": ew["risk_classification"],
                    "Min-Variance Portfolio": opt_result.risk_classification if opt_result.success else "N/A",
                },
            ]
            st.dataframe(pd.DataFrame(risk_metrics_rows), use_container_width=True, hide_index=True)

        with comp_col2:
            user_metrics_dict = {
                "volatility": summary.annualized_volatility,
                "max_drawdown": summary.max_drawdown,
                "beta": summary.beta,
                "div_ratio": summary.diversification_ratio,
            }
            ew_metrics_dict = {
                "volatility": ew["annualized_volatility"],
                "max_drawdown": ew["max_drawdown"],
                "beta": ew["beta"],
                "div_ratio": ew["diversification_ratio"],
            }
            opt_metrics_dict = (
                {
                    "volatility": opt_result.annualized_volatility,
                    "max_drawdown": opt_result.max_drawdown,
                    "beta": opt_result.beta,
                    "div_ratio": opt_result.diversification_ratio,
                }
                if opt_result.success
                else None
            )
            fig_comp = plot_portfolio_comparison_bars(user_metrics_dict, ew_metrics_dict, opt_metrics_dict)
            st.pyplot(fig_comp)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Panel E: Extended Stress Testing (Predefined + Single Stock + Custom)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="panel-heading">Hypothetical Stress Testing Scenarios</div>', unsafe_allow_html=True)
        st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

        # Interactive controls for single-stock and custom shocks
        col_st1, col_st2, col_st3 = st.columns([1.2, 1.0, 1.0])
        with col_st1:
            selected_st_stock = st.selectbox(
                "Individual Stock to Stress",
                options=res["tickers"],
                index=0,
                help="Select an equity holding to model an isolated idiosyncratic decline.",
            )
        with col_st2:
            selected_st_shock = st.number_input(
                "Individual Stock Drop (%)",
                min_value=1.0,
                max_value=99.0,
                value=float(st.session_state.stress_stock_pct),
                step=5.0,
                help="Percentage drop applied exclusively to the selected holding.",
            )
            st.session_state.stress_stock_pct = selected_st_shock
        with col_st3:
            custom_mkt_shock = st.number_input(
                "Custom Market Shock (%)",
                min_value=1.0,
                max_value=99.0,
                value=float(st.session_state.stress_custom_pct),
                step=5.0,
                help="Custom uniform market-wide drawdown applied across all holdings.",
            )
            st.session_state.stress_custom_pct = custom_mkt_shock

        # Re-run stress test scenarios dynamically with active parameters
        active_stress = run_stress_testing(
            tickers=res["tickers"],
            investments=res["investments"],
            selected_single_ticker=selected_st_stock,
            single_stock_shock_pct=-abs(selected_st_shock / 100.0),
            custom_market_shock_pct=-abs(custom_mkt_shock / 100.0),
        )

        stress_display_rows = []
        for sr in active_stress:
            stress_display_rows.append({
                "Scenario": sr.scenario_name,
                "Affected holdings": sr.affected_holdings,
                "Estimated loss (₹)": f"-{format_inr(sr.loss_amount)}",
                "Portfolio loss (%)": f"{sr.loss_pct:.1f}%",
                "Remaining value (₹)": format_inr(sr.remaining_value),
            })
        st.dataframe(pd.DataFrame(stress_display_rows), use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="panel-note">
                <strong>Assumptions:</strong> Scenarios represent instantaneous market drawdowns applied directly to nominal capital allocations under liquidity-freeze assumptions.
                These are educational stress sensitivities and do not account for taxes, transaction costs, slippage, bid-ask spreads, or other real-world market frictions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Panel F: Downloadable Results (Multi-Sheet Excel & Consolidated CSV)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="panel-heading">Downloadable Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            excel_bytes = generate_excel_report(
                analysis_data=analysis_data,
                opt_result=opt_result,
                stress_results=active_stress,
                period_label=period_label,
            )
            st.download_button(
                label="Download Multi-Sheet Excel Workbook (Format .xlsx)",
                data=excel_bytes,
                file_name=f"Portfolio_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                help="Includes Stock Metrics, Original Allocation, Equal-Weight, Min-Variance, Comparison, Stress Tests, and Assumptions.",
            )

        with col_dl2:
            csv_text = generate_csv_report(
                analysis_data=analysis_data,
                opt_result=opt_result,
                stress_results=active_stress,
                period_label=period_label,
            )
            st.download_button(
                label="Download Consolidated CSV Report (Format .csv)",
                data=csv_text,
                file_name=f"Portfolio_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True,
                help="Plaintext CSV export containing complete tabular analytics.",
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Panel G: Methodology & Limitations (Comprehensive Academic Sections A to F)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<div class="panel-heading">Methodology, Mathematical Formulas, and Academic Disclaimer</div>', unsafe_allow_html=True)
        st.markdown('<hr class="panel-divider">', unsafe_allow_html=True)

        st.markdown(
            f"""
            #### A. Return Methodology
            - **Buy-and-Hold Portfolio Valuation:** At inception ($t_0$), initial capital $I_i$ purchases $n_i = I_i / P_{{i,0}}$ shares.
              At any subsequent date $t$, holding valuation is $V_{{i,t}} = n_i P_{{i,t}}$, and total portfolio value is:
              $$V_t = \\sum_{{i=1}}^N V_{{i,t}} = \\sum_{{i=1}}^N I_i \\left(\\frac{{P_{{i,t}}}}{{P_{{i,0}}}}\\right)$$
              Daily portfolio returns are derived from the valuation trajectory:
              $$r_{{p,t}} = \\frac{{V_t - V_{{t-1}}}}{{V_{{t-1}}}}$$
            - **Fractional Shares Assumption:** Initial quantities are calculated as continuous fractional shares ($I_i / P_{{i,0}}$) to evaluate exact capital deployment without cash residue.

            #### B. Risk Metrics & Formulations
            - **Annualized Volatility:** $\\sigma_{{annual}} = \\sigma_{{daily}} \\times \\sqrt{{{TRADING_DAYS_PER_YEAR}}}$, where $\\sigma_{{daily}}$ is sample standard deviation (ddof=1).
            - **Compounded Annual Growth Rate (CAGR):** $\\text{{CAGR}} = \\left(\\frac{{V_{{end}}}}{{V_{{start}}}}\\right)^{{1 / T}} - 1$, where $T$ is elapsed calendar time in years.
            - **Maximum Drawdown (MDD):** Peak-to-trough decline: $\\text{{MDD}} = \\min_t \\left(\\frac{{V_t - \\max_{{s \\le t}} V_s}}{{\\max_{{s \\le t}} V_s}}\\right)$.
            - **Portfolio Beta ($\\beta$):** Systematic market sensitivity relative to {BENCHMARK_NAME} (`{BENCHMARK_TICKER}`):
              $$\\beta = \\frac{{\\text{{Cov}}(r_p, r_m)}}{{\\text{{Var}}(r_m)}}$$
            - **Herfindahl-Hirschman Concentration Index (HHI):** $\\text{{HHI}} = \\sum_{{i=1}}^N w_i^2$, where $w_i = I_i / \\sum I$.
            - **Correlation Matrix:** Pearson correlation of aligned daily equity returns ($N \\times N$).
            - **Diversification Ratio (DR):** $\\text{{DR}} = \\frac{{\\sum_{{i=1}}^N w_i \\sigma_i}}{{\\sigma_p}}$. A ratio $> 1.0$ indicates risk reduction benefit from non-perfect inter-asset correlation.

            #### C. Minimum-Variance Allocation Optimizer
            - Constrained quadratic numerical optimization using SciPy (`SLSQP`):
              $$\\min_w \\quad w^T \\Sigma w \\quad \\text{{subject to}} \\quad \\sum_{{i=1}}^n w_i = 1, \\quad 0 \\le w_i \\le w_{{max}}$$
            - Enforces full investment and a configurable single-asset ceiling ($w_{{max}} = {st.session_state.max_optimal_weight*100:.0f}\\%$).
            - Feasibility condition: $n \\times w_{{max}} \\ge 1.0$.

            #### D. Stress Testing Scenarios
            - **Predefined Shocks:** Uniform instantaneous market corrections of $-5\\%$, $-10\\%$, $-15\\%$, and $-20\\%$.
            - **Individual-Stock Shock:** User-selected asset decline with all other portfolio holdings remaining flat.
            - **Custom Market Shock:** User-specified uniform percentage decline across all assets.

            #### E. Educational Risk Classification
            - **Low Risk:** Annualized Volatility $< 15\\%$
            - **Moderate Risk:** $15\\% \\le$ Annualized Volatility $< 25\\%$
            - **High Risk:** Annualized Volatility $\\ge 25\\%$
            *Note:* These thresholds are pedagogical guidelines and do not constitute statutory or advisory definitions.

            #### F. Data Limitations & Academic Disclaimer
            - Historical adjusted closing prices are retrieved from Yahoo Finance via the `yfinance` library.
            - Benchmark: {BENCHMARK_NAME} (`{BENCHMARK_TICKER}`).
            - Excludes transaction fees, brokerage, taxes, slippage, and liquidity impact.
            - **This application is built solely for educational and academic research. It does NOT forecast future market prices or provide buy/sell investment advice.**
            """
        )


if __name__ == "__main__":
    main()
