"""Unit tests for portfolio stress testing scenarios in portfolio_risk.stress_testing."""

import numpy as np
import pytest

from portfolio_risk.stress_testing import (
    StressScenarioResult,
    format_stress_results_table,
    run_stress_testing,
)


def test_stress_testing_predefined_four_scenarios():
    """Verify predefined scenarios (-5%, -10%, -15%, -20%) and individual stock shock."""
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    # Total: ₹1,00,000 with largest holding RELIANCE.NS at ₹40,000
    investments = [40000.0, 30000.0, 20000.0, 10000.0]

    results = run_stress_testing(tickers, investments)
    # 4 predefined + 1 default individual stock shock = 5 scenarios
    assert len(results) == 5

    # 1. Mild Market Pullback (-5%)
    s_5 = results[0]
    assert s_5.scenario_id == "scenario_5pct"
    assert np.isclose(s_5.loss_amount, 5000.0)
    assert np.isclose(s_5.loss_pct, -5.0)
    assert np.isclose(s_5.remaining_value, 95000.0)
    assert s_5.affected_holdings == "All Holdings (Uniform)"

    # 2. Broad Market Correction (-10%)
    s_10 = results[1]
    assert s_10.scenario_id == "scenario_10pct"
    assert np.isclose(s_10.loss_amount, 10000.0)
    assert np.isclose(s_10.loss_pct, -10.0)
    assert np.isclose(s_10.remaining_value, 90000.0)

    # 3. Deep Market Correction (-15%)
    s_15 = results[2]
    assert s_15.scenario_id == "scenario_15pct"
    assert np.isclose(s_15.loss_amount, 15000.0)
    assert np.isclose(s_15.loss_pct, -15.0)
    assert np.isclose(s_15.remaining_value, 85000.0)

    # 4. Severe Market Crash (-20%)
    s_20 = results[3]
    assert s_20.scenario_id == "scenario_20pct"
    assert np.isclose(s_20.loss_amount, 20000.0)
    assert np.isclose(s_20.loss_pct, -20.0)
    assert np.isclose(s_20.remaining_value, 80000.0)

    # 5. Individual shock to largest holding (default 30% decline on RELIANCE.NS: 40k * 0.3 = 12k)
    s_ind = results[4]
    assert s_ind.scenario_id == "scenario_single_stock"
    assert "RELIANCE.NS" in s_ind.affected_holdings
    assert np.isclose(s_ind.loss_amount, 12000.0)
    assert np.isclose(s_ind.loss_pct, -12.0)
    assert np.isclose(s_ind.remaining_value, 88000.0)


def test_stress_testing_selectable_individual_stock():
    """Verify user can select a specific holding and shock percentage for individual testing."""
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    investments = [50000.0, 30000.0, 20000.0]

    # Select TCS.NS with 25% decline (30,000 * 0.25 = 7,500)
    results = run_stress_testing(
        tickers=tickers,
        investments=investments,
        selected_single_ticker="TCS.NS",
        single_stock_shock_pct=-0.25,
    )
    s_ind = results[4]
    assert "TCS.NS" in s_ind.affected_holdings
    assert np.isclose(s_ind.loss_amount, 7500.0)
    assert np.isclose(s_ind.loss_pct, -7.5)  # 7500 / 100000
    assert np.isclose(s_ind.remaining_value, 92500.0)


def test_stress_testing_custom_market_scenario():
    """Verify custom market-wide uniform percentage decline."""
    tickers = ["RELIANCE.NS", "TCS.NS"]
    investments = [60000.0, 40000.0]  # Total 1,00,000

    # Custom shock of -12.5%
    results = run_stress_testing(
        tickers=tickers,
        investments=investments,
        custom_market_shock_pct=-0.125,
    )
    # 4 predefined + 1 individual + 1 custom = 6 scenarios
    assert len(results) == 6
    s_custom = results[5]
    assert s_custom.scenario_id == "scenario_custom_market"
    assert np.isclose(s_custom.loss_amount, 12500.0)
    assert np.isclose(s_custom.loss_pct, -12.5)
    assert np.isclose(s_custom.remaining_value, 87500.0)


def test_stress_testing_single_holding():
    """Verify stress test calculations for a 1-holding portfolio."""
    tickers = ["RELIANCE.NS"]
    investments = [50000.0]

    results = run_stress_testing(tickers, investments)
    assert len(results) == 5

    # Uniform -10%
    assert np.isclose(results[1].loss_amount, 5000.0)
    assert np.isclose(results[1].remaining_value, 45000.0)

    # Uniform -20%
    assert np.isclose(results[3].loss_amount, 10000.0)
    assert np.isclose(results[3].remaining_value, 40000.0)

    # Single holding shock (-30% default)
    # Since single holding is 100% of capital, loss is 30% of total capital
    assert np.isclose(results[4].loss_amount, 15000.0)
    assert np.isclose(results[4].loss_pct, -30.0)
    assert np.isclose(results[4].remaining_value, 35000.0)
    assert "RELIANCE.NS" in results[4].affected_holdings


def test_stress_testing_dynamic_largest_identification():
    """Verify individual shock defaults to the largest holding dynamically."""
    tickers = ["STOCK1.NS", "STOCK2.NS", "STOCK3.NS"]
    # Middle holding is the largest: ₹60,000 out of ₹1,00,000
    investments = [20000.0, 60000.0, 20000.0]

    results = run_stress_testing(tickers, investments)
    s_ind = results[4]
    assert "STOCK2.NS" in s_ind.affected_holdings
    assert np.isclose(s_ind.loss_amount, 18000.0)  # 30% of 60,000 = 18,000
    assert np.isclose(s_ind.loss_pct, -18.0)
    assert np.isclose(s_ind.remaining_value, 82000.0)


def test_stress_testing_formatting_table():
    """Verify formatting function constructs a valid pandas DataFrame with required columns."""
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    investments = [25000.0, 25000.0, 25000.0, 25000.0]
    results = run_stress_testing(tickers, investments)
    table_df = format_stress_results_table(results)

    assert len(table_df) == 5
    expected_cols = [
        "Scenario",
        "Affected holdings",
        "Estimated loss (₹)",
        "Portfolio loss (%)",
        "Remaining value (₹)",
    ]
    for col in expected_cols:
        assert col in table_df.columns


def test_stress_testing_invalid_inputs():
    """Verify error raised on mismatched input lengths or non-positive capital."""
    with pytest.raises(ValueError, match="same length"):
        run_stress_testing(["RELIANCE.NS", "TCS.NS"], [50000.0])

    with pytest.raises(ValueError, match="greater than zero"):
        run_stress_testing(["RELIANCE.NS", "TCS.NS"], [0.0, 0.0])
