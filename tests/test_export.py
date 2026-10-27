"""Unit tests for Excel and CSV report exports in portfolio_risk.export."""

import io
import openpyxl
import pandas as pd
import pytest

from portfolio_risk.export import generate_csv_report, generate_excel_report
from portfolio_risk.metrics import analyze_portfolio
from portfolio_risk.optimization import optimize_minimum_variance
from portfolio_risk.stress_testing import run_stress_testing


def test_generate_excel_and_csv_reports(synthetic_price_data):
    """Verify Excel workbook contains all 7 sheets and CSV export contains all sections."""
    df_stocks, s_benchmark = synthetic_price_data
    tickers = list(df_stocks.columns)
    investments = [25000.0] * len(tickers)

    # Run analysis
    analysis_data = analyze_portfolio(
        prices=df_stocks,
        investments=investments,
        benchmark_prices=s_benchmark,
    )

    # Run optimization
    opt_result = optimize_minimum_variance(
        prices=df_stocks,
        total_capital=100000.0,
        benchmark_returns=analysis_data["benchmark_returns"],
        max_weight=0.50,
    )

    # Run stress testing
    stress_results = run_stress_testing(tickers, investments)

    # 1. Test Excel generation
    excel_bytes = generate_excel_report(
        analysis_data=analysis_data,
        opt_result=opt_result,
        stress_results=stress_results,
        period_label="5 Years",
    )
    assert len(excel_bytes) > 0

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Stock Metrics" in sheet_names
    assert "Original Allocation" in sheet_names
    assert "Equal-Weight Allocation" in sheet_names
    assert "Minimum-Variance Allocation" in sheet_names
    assert "Portfolio Comparison" in sheet_names
    assert "Stress Tests" in sheet_names
    assert "Assumptions" in sheet_names

    # 2. Test CSV generation
    csv_text = generate_csv_report(
        analysis_data=analysis_data,
        opt_result=opt_result,
        stress_results=stress_results,
        period_label="5 Years",
    )
    assert len(csv_text) > 0
    assert "# STOCK PORTFOLIO RISK & STRESS TESTING REPORT" in csv_text
    assert "## STOCK LEVEL METRICS" in csv_text
    assert "## PORTFOLIO ALLOCATION COMPARISON" in csv_text
    assert "## PORTFOLIO RISK COMPARISON" in csv_text
    assert "## STRESS TESTING SCENARIOS" in csv_text
