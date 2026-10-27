"""Stress testing module for hypothetical instantaneous portfolio shocks.

Models static, instantaneous equity market drawdowns to evaluate portfolio resilience
and idiosyncratic risk exposure without requiring predictive forecasting.
Supports predefined market-wide shocks (5%, 10%, 15%, 20%), single-stock stress,
and custom market-wide stress tests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


@dataclass
class StressScenarioResult:
    """Container holding output metrics for an individual stress test scenario."""
    scenario_id: str
    scenario_name: str
    description: str
    affected_holdings: str
    shock_pct: float
    initial_portfolio_value: float
    loss_amount: float
    loss_pct: float
    remaining_value: float


def run_stress_testing(
    tickers: Sequence[str],
    investments: Sequence[float],
    selected_single_ticker: Optional[str] = None,
    single_stock_shock_pct: float = -0.30,
    custom_market_shock_pct: Optional[float] = None,
) -> List[StressScenarioResult]:
    """Execute comprehensive instantaneous stress test scenarios.

    Scenarios evaluated:
    1. Predefined Market Pullback: -5% uniform
    2. Predefined Market Correction: -10% uniform
    3. Predefined Deep Correction: -15% uniform
    4. Predefined Severe Market Crash: -20% uniform
    5. Individual-Stock Idiosyncratic Shock (e.g. -30% or user specified on selected holding)
    6. Custom Market-Wide Scenario (if provided)

    Args:
        tickers: Sequence of ticker symbols.
        investments: Nominal investment amount per ticker in INR.
        selected_single_ticker: Specific ticker for individual stress test (default: largest holding).
        single_stock_shock_pct: Percentage decline for individual stock (default: -0.30, i.e., -30%).
        custom_market_shock_pct: Optional custom percentage decline for all holdings (e.g. -0.12 for -12%).

    Returns:
        List of StressScenarioResult objects.

    Raises:
        ValueError: If inputs are invalid or length mismatch occurs.
    """
    if len(tickers) != len(investments):
        raise ValueError("Tickers and investments must have the same length.")

    inv_array = np.asarray(investments, dtype=float)
    total_val = float(np.sum(inv_array))
    if total_val <= 0:
        raise ValueError("Total portfolio value must be greater than zero.")

    results: List[StressScenarioResult] = []

    # 1. Predefined Market Scenarios: 5%, 10%, 15%, 20%
    predefined_shocks = [
        ("scenario_5pct", "Mild Market Pullback (-5%)", -0.05),
        ("scenario_10pct", "Broad Market Correction (-10%)", -0.10),
        ("scenario_15pct", "Deep Market Correction (-15%)", -0.15),
        ("scenario_20pct", "Severe Market Crash (-20%)", -0.20),
    ]

    for s_id, s_name, s_pct in predefined_shocks:
        loss = total_val * abs(s_pct)
        rem = total_val - loss
        results.append(
            StressScenarioResult(
                scenario_id=s_id,
                scenario_name=s_name,
                description=f"Hypothetical uniform {abs(s_pct)*100:.0f}% decline across all portfolio holdings.",
                affected_holdings="All Holdings (Uniform)",
                shock_pct=s_pct,
                initial_portfolio_value=total_val,
                loss_amount=loss,
                loss_pct=s_pct * 100.0,
                remaining_value=rem,
            )
        )

    # 2. Individual Stock Stress Scenario
    # Determine which stock to shock
    if selected_single_ticker and selected_single_ticker in tickers:
        stock_idx = list(tickers).index(selected_single_ticker)
        stock_ticker = selected_single_ticker
    else:
        # Default to largest holding
        stock_idx = int(np.argmax(inv_array))
        stock_ticker = tickers[stock_idx]

    stock_amt = inv_array[stock_idx]
    single_shock = -abs(single_stock_shock_pct)
    stock_loss = stock_amt * abs(single_shock)
    stock_rem = total_val - stock_loss
    stock_loss_pct = -(stock_loss / total_val) * 100.0

    results.append(
        StressScenarioResult(
            scenario_id="scenario_single_stock",
            scenario_name=f"Individual Shock: {stock_ticker} ({single_shock*100:.1f}%)",
            description=(
                f"Hypothetical {abs(single_shock)*100:.1f}% decline in {stock_ticker} (₹{stock_amt:,.2f}), "
                "with all other assets remaining unchanged."
            ),
            affected_holdings=f"{stock_ticker} ({single_shock*100:.1f}%)",
            shock_pct=single_shock,
            initial_portfolio_value=total_val,
            loss_amount=stock_loss,
            loss_pct=stock_loss_pct,
            remaining_value=stock_rem,
        )
    )

    # 3. Custom Market-Wide Scenario (if supplied)
    if custom_market_shock_pct is not None:
        c_pct = -abs(custom_market_shock_pct)
        if -1.0 <= c_pct <= 0.0 and abs(c_pct) > 1e-6:
            c_loss = total_val * abs(c_pct)
            c_rem = total_val - c_loss
            results.append(
                StressScenarioResult(
                    scenario_id="scenario_custom_market",
                    scenario_name=f"Custom Market Shock ({c_pct*100:.1f}%)",
                    description=f"User-specified uniform {abs(c_pct)*100:.1f}% decline across all portfolio holdings.",
                    affected_holdings="All Holdings (Uniform)",
                    shock_pct=c_pct,
                    initial_portfolio_value=total_val,
                    loss_amount=c_loss,
                    loss_pct=c_pct * 100.0,
                    remaining_value=c_rem,
                )
            )

    return results


def format_stress_results_table(results: List[StressScenarioResult]) -> pd.DataFrame:
    """Format stress test results into a clean tabular DataFrame for UI presentation.

    Args:
        results: List of StressScenarioResult dataclasses.

    Returns:
        pd.DataFrame formatted with scenario names, estimated losses, and remaining values.
    """
    rows = []
    for r in results:
        rows.append({
            "Scenario": r.scenario_name,
            "Affected holdings": r.affected_holdings,
            "Estimated loss (₹)": r.loss_amount,
            "Portfolio loss (%)": r.loss_pct,
            "Remaining value (₹)": r.remaining_value,
        })
    return pd.DataFrame(rows)
