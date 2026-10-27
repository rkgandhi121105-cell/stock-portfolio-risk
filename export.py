"""Export module for generating downloadable analysis reports in Excel and CSV formats.

Produces multi-sheet Excel workbooks and CSV files containing stock metrics,
original allocation, equal-weight allocation, minimum-variance allocation,
comprehensive risk comparisons, stress test scenarios, and methodology metadata.
"""

import io
from typing import Dict, List, Optional
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
import pandas as pd

from portfolio_risk.config import BENCHMARK_NAME, BENCHMARK_TICKER, TRADING_DAYS_PER_YEAR
from portfolio_risk.optimization import OptimizationResult
from portfolio_risk.stress_testing import StressScenarioResult


def generate_excel_report(
    analysis_data: Dict[str, object],
    opt_result: OptimizationResult,
    stress_results: List[StressScenarioResult],
    period_label: str,
) -> bytes:
    """Generate a multi-sheet formatted Excel workbook containing complete portfolio results.

    Sheets created:
    1. Stock Metrics
    2. Original Allocation
    3. Equal-Weight Allocation
    4. Minimum-Variance Allocation
    5. Portfolio Comparison
    6. Stress Tests
    7. Assumptions & Metadata

    Args:
        analysis_data: Output dictionary from analyze_portfolio.
        opt_result: Output from optimize_minimum_variance.
        stress_results: List of StressScenarioResult objects.
        period_label: Selected historical lookback label (e.g. '5 Years').

    Returns:
        Raw bytes of the generated .xlsx workbook.
    """
    output = io.BytesIO()
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="387ED1", end_color="387ED1", fill_type="solid")
    right_align = Alignment(horizontal="right")
    left_align = Alignment(horizontal="left")

    summary = analysis_data["summary"]
    ew = analysis_data["equal_weight"]
    tickers = analysis_data["tickers"]
    raw_holdings = analysis_data["holdings_df"]

    # -------------------------------------------------------------------------
    # Sheet 1: Stock Metrics
    # -------------------------------------------------------------------------
    ws_stocks = wb.create_sheet(title="Stock Metrics")
    headers_s = ["Ticker", "Investment (INR)", "Weight (%)", "Annualized Volatility (%)", "Beta (^NSEI)", "CAGR (%)"]
    ws_stocks.append(headers_s)
    for _, row in raw_holdings.iterrows():
        ws_stocks.append([
            row["Ticker"],
            float(row["Investment (₹)"]),
            float(row["Weight (%)"]),
            float(row["Annualized Volatility (%)"]),
            float(row["Beta (^NSEI)"]) if not pd.isna(row["Beta (^NSEI)"]) else "N/A",
            float(row["CAGR (%)"]) if not pd.isna(row["CAGR (%)"]) else "N/A",
        ])

    # -------------------------------------------------------------------------
    # Sheet 2: Original Allocation
    # -------------------------------------------------------------------------
    ws_orig = wb.create_sheet(title="Original Allocation")
    ws_orig.append(["Ticker", "Investment (INR)", "Weight (%)"])
    for t, inv, w in zip(tickers, raw_holdings["Investment (₹)"], raw_holdings["Weight (%)"]):
        ws_orig.append([t, float(inv), float(w)])
    ws_orig.append(["Total", summary.total_investment, 100.0])

    # -------------------------------------------------------------------------
    # Sheet 3: Equal-Weight Allocation
    # -------------------------------------------------------------------------
    ws_ew = wb.create_sheet(title="Equal-Weight Allocation")
    ws_ew.append(["Ticker", "Recommended Investment (INR)", "Weight (%)"])
    for t, inv, w in zip(tickers, ew["investments"], ew["weights"]):
        ws_ew.append([t, float(inv), float(w * 100.0)])
    ws_ew.append(["Total", summary.total_investment, 100.0])

    # -------------------------------------------------------------------------
    # Sheet 4: Minimum-Variance Allocation
    # -------------------------------------------------------------------------
    ws_mv = wb.create_sheet(title="Minimum-Variance Allocation")
    ws_mv.append(["Ticker", "Recommended Investment (INR)", "Weight (%)", "Cap Limit (%)"])
    if opt_result.success:
        for t, inv, w in zip(tickers, opt_result.investments, opt_result.weights):
            ws_mv.append([t, float(inv), float(w * 100.0), float(opt_result.max_weight_constraint * 100.0)])
        ws_mv.append(["Total", summary.total_investment, 100.0, "—"])
    else:
        ws_mv.append(["Status", opt_result.message, "", ""])

    # -------------------------------------------------------------------------
    # Sheet 5: Portfolio Comparison
    # -------------------------------------------------------------------------
    ws_comp = wb.create_sheet(title="Portfolio Comparison")
    ws_comp.append(["Metric", "Original User Portfolio", "Equal-Weight Baseline", "Minimum-Variance Portfolio"])
    ws_comp.append(["Total Capital (INR)", summary.total_investment, summary.total_investment, summary.total_investment])
    ws_comp.append([
        "Annualized Volatility (%)",
        summary.annualized_volatility * 100.0,
        ew["annualized_volatility"] * 100.0,
        opt_result.annualized_volatility * 100.0 if opt_result.success else "N/A",
    ])
    ws_comp.append([
        "Maximum Drawdown (%)",
        summary.max_drawdown * 100.0,
        ew["max_drawdown"] * 100.0,
        opt_result.max_drawdown * 100.0 if opt_result.success else "N/A",
    ])
    ws_comp.append([
        "CAGR (%)",
        summary.cagr * 100.0 if not pd.isna(summary.cagr) else "N/A",
        ew["cagr"] * 100.0 if not pd.isna(ew["cagr"]) else "N/A",
        opt_result.cagr * 100.0 if (opt_result.success and not pd.isna(opt_result.cagr)) else "N/A",
    ])
    ws_comp.append([
        "Portfolio Beta (^NSEI)",
        summary.beta if not pd.isna(summary.beta) else "N/A",
        ew["beta"] if not pd.isna(ew["beta"]) else "N/A",
        opt_result.beta if (opt_result.success and not pd.isna(opt_result.beta)) else "N/A",
    ])
    ws_comp.append([
        "Concentration (HHI)",
        summary.hhi,
        ew["hhi"],
        opt_result.hhi if opt_result.success else "N/A",
    ])
    ws_comp.append([
        "Diversification Ratio",
        summary.diversification_ratio,
        ew["diversification_ratio"],
        opt_result.diversification_ratio if opt_result.success else "N/A",
    ])
    ws_comp.append([
        "Risk Classification",
        summary.risk_classification,
        ew["risk_classification"],
        opt_result.risk_classification if opt_result.success else "N/A",
    ])

    # -------------------------------------------------------------------------
    # Sheet 6: Stress Tests
    # -------------------------------------------------------------------------
    ws_stress = wb.create_sheet(title="Stress Tests")
    ws_stress.append(["Scenario", "Affected Holdings", "Estimated Loss (INR)", "Portfolio Loss (%)", "Remaining Value (INR)"])
    for sr in stress_results:
        ws_stress.append([
            sr.scenario_name,
            sr.affected_holdings,
            float(sr.loss_amount),
            float(sr.loss_pct),
            float(sr.remaining_value),
        ])

    # -------------------------------------------------------------------------
    # Sheet 7: Assumptions
    # -------------------------------------------------------------------------
    ws_meta = wb.create_sheet(title="Assumptions")
    ws_meta.append(["Parameter", "Value", "Description"])
    ws_meta.append(["Benchmark Index", f"{BENCHMARK_NAME} ({BENCHMARK_TICKER})", "Standard market proxy for Indian equities"])
    ws_meta.append(["Requested Period", period_label, "Selected historical lookback"])
    ws_meta.append(["Actual Start Date", summary.actual_start_date, "First synchronized observation trading day"])
    ws_meta.append(["Actual End Date", summary.actual_end_date, "Latest synchronized observation trading day"])
    ws_meta.append(["Annual Trading Days", TRADING_DAYS_PER_YEAR, "Calendar sessions scaling factor"])
    ws_meta.append(["Portfolio Valuation Model", "Buy-and-Hold", "Cumulative capital evolution from initial share count"])
    ws_meta.append(["Optimizer Method", "SciPy SLSQP", "Constrained quadratic variance minimization w^T * Sigma * w"])
    ws_meta.append(["Max Stock Weight Cap", f"{opt_result.max_weight_constraint*100:.1f}%", "Configurable single-stock allocation ceiling"])
    ws_meta.append(["Optimizer Status", opt_result.message, "Mathematical feasibility and solver convergence"])
    ws_meta.append(["Academic Disclaimer", "Educational use only", "Not investment advice; no future price predictions"])

    # Style all sheets with header formatting and column auto-width
    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = left_align
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = max(max_len + 3, 14)

    wb.save(output)
    output.seek(0)
    return output.getvalue()


def generate_csv_report(
    analysis_data: Dict[str, object],
    opt_result: OptimizationResult,
    stress_results: List[StressScenarioResult],
    period_label: str,
) -> str:
    """Generate consolidated CSV text summarizing complete portfolio risk analytics."""
    summary = analysis_data["summary"]
    ew = analysis_data["equal_weight"]
    raw_holdings = analysis_data["holdings_df"]

    lines = []
    lines.append("# STOCK PORTFOLIO RISK & STRESS TESTING REPORT")
    lines.append(f"# Benchmark: {BENCHMARK_NAME} ({BENCHMARK_TICKER})")
    lines.append(f"# Period: {period_label} ({summary.actual_start_date} to {summary.actual_end_date})")
    lines.append(f"# Model: Buy-and-Hold Valuation | Annual Trading Days: {TRADING_DAYS_PER_YEAR}")
    lines.append("")

    lines.append("## STOCK LEVEL METRICS")
    lines.append(raw_holdings.to_csv(index=False))

    lines.append("## PORTFOLIO ALLOCATION COMPARISON")
    comp_alloc = pd.DataFrame({
        "Ticker": analysis_data["tickers"],
        "Original_Investment_INR": raw_holdings["Investment (₹)"],
        "Original_Weight_Pct": raw_holdings["Weight (%)"],
        "EqualWeight_Investment_INR": ew["investments"],
        "EqualWeight_Weight_Pct": [w * 100 for w in ew["weights"]],
        "MinVariance_Investment_INR": opt_result.investments if opt_result.success else ["N/A"] * len(analysis_data["tickers"]),
        "MinVariance_Weight_Pct": [w * 100 for w in opt_result.weights] if opt_result.success else ["N/A"] * len(analysis_data["tickers"]),
    })
    lines.append(comp_alloc.to_csv(index=False))

    lines.append("## PORTFOLIO RISK COMPARISON")
    comp_risk = pd.DataFrame({
        "Metric": [
            "Total Capital (INR)", "Annualized Volatility (%)", "Maximum Drawdown (%)",
            "CAGR (%)", "Portfolio Beta (^NSEI)", "Concentration (HHI)", "Diversification Ratio", "Risk Classification"
        ],
        "Original_Portfolio": [
            summary.total_investment, summary.annualized_volatility * 100.0, summary.max_drawdown * 100.0,
            summary.cagr * 100.0 if not pd.isna(summary.cagr) else "N/A", summary.beta, summary.hhi, summary.diversification_ratio, summary.risk_classification
        ],
        "Equal_Weight_Baseline": [
            summary.total_investment, ew["annualized_volatility"] * 100.0, ew["max_drawdown"] * 100.0,
            ew["cagr"] * 100.0 if not pd.isna(ew["cagr"]) else "N/A", ew["beta"], ew["hhi"], ew["diversification_ratio"], ew["risk_classification"]
        ],
        "Minimum_Variance_Portfolio": [
            summary.total_investment,
            opt_result.annualized_volatility * 100.0 if opt_result.success else "N/A",
            opt_result.max_drawdown * 100.0 if opt_result.success else "N/A",
            opt_result.cagr * 100.0 if (opt_result.success and not pd.isna(opt_result.cagr)) else "N/A",
            opt_result.beta if opt_result.success else "N/A",
            opt_result.hhi if opt_result.success else "N/A",
            opt_result.diversification_ratio if opt_result.success else "N/A",
            opt_result.risk_classification if opt_result.success else "N/A",
        ],
    })
    lines.append(comp_risk.to_csv(index=False))

    lines.append("## STRESS TESTING SCENARIOS")
    stress_rows = []
    for sr in stress_results:
        stress_rows.append({
            "Scenario": sr.scenario_name,
            "Affected_Holdings": sr.affected_holdings,
            "Estimated_Loss_INR": sr.loss_amount,
            "Portfolio_Loss_Pct": sr.loss_pct,
            "Remaining_Value_INR": sr.remaining_value,
        })
    lines.append(pd.DataFrame(stress_rows).to_csv(index=False))

    return "\n".join(lines)
