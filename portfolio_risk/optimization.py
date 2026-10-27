"""Minimum-Variance Portfolio Allocation Optimizer.

Implements constrained numerical optimization using SciPy to find the portfolio weights
that minimize estimated historical return variance subject to full investment and
maximum asset weight constraints:

    minimize     w^T * Sigma * w
    subject to   sum(w_i) = 1
                 0 <= w_i <= w_max
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_risk.config import DEFAULT_MAX_OPTIMAL_WEIGHT, TRADING_DAYS_PER_YEAR
from portfolio_risk.metrics import (
    allocate_whole_rupees,
    calculate_annualized_volatility,
    calculate_beta,
    calculate_buy_and_hold_portfolio_value,
    calculate_cagr,
    calculate_diversification_ratio,
    calculate_hhi,
    calculate_max_drawdown,
    calculate_portfolio_daily_returns,
    classify_risk,
)


class OptimizationError(ValueError):
    """Exception raised when portfolio optimization fails or constraints are infeasible."""
    pass


@dataclass
class OptimizationResult:
    """Container holding output metrics and status for the Minimum-Variance allocation."""
    success: bool
    message: str
    weights: np.ndarray
    investments: List[float]
    annualized_volatility: float
    max_drawdown: float
    cagr: float
    beta: float
    hhi: float
    diversification_ratio: float
    risk_classification: str
    max_weight_constraint: float
    portfolio_value: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None


def check_optimization_feasibility(
    num_assets: int,
    max_weight: float,
) -> Tuple[bool, str]:
    """Check mathematical feasibility of the constrained optimization problem.

    To achieve sum(w_i) = 1.0 with w_i <= w_max for all i in {1..n},
    the maximum possible total weight n * w_max must be >= 1.0.

    Args:
        num_assets: Number of equities in the candidate universe (n).
        max_weight: Upper bound constraint per asset (w_max).

    Returns:
        Tuple of (is_feasible: bool, reason: str).
    """
    if num_assets < 1:
        return False, "Portfolio must contain at least 1 asset for optimization."

    if max_weight <= 0.0 or max_weight > 1.0:
        return False, f"Maximum weight constraint must be between 0 and 1. Received {max_weight:.2f}."

    # For a single asset, 100% in that asset is the only fully invested allocation
    if num_assets == 1:
        if max_weight < 1.0:
            return (
                False,
                f"Infeasible constraint for single-stock portfolio: Cap of {max_weight*100:.0f}% "
                "prevents 100% full investment in the single holding.",
            )
        return True, "Feasible single-stock allocation."

    if num_assets * max_weight < 1.0 - 1e-9:
        max_possible = num_assets * max_weight * 100.0
        return (
            False,
            f"Infeasible constraints: With {num_assets} holdings and a maximum weight cap of "
            f"{max_weight*100:.1f}%, the maximum achievable portfolio allocation is {max_possible:.1f}%, "
            "which cannot satisfy the 100% full investment requirement.",
        )

    return True, "Constraints are mathematically feasible."


def optimize_minimum_variance(
    prices: pd.DataFrame,
    total_capital: float,
    benchmark_returns: pd.Series,
    max_weight: float = DEFAULT_MAX_OPTIMAL_WEIGHT,
) -> OptimizationResult:
    """Solve for the Minimum-Variance portfolio allocation using SciPy.

    Minimizes historical portfolio return variance w^T * Sigma * w subject to:
        sum(w_i) = 1.0
        0.0 <= w_i <= max_weight

    Args:
        prices: Synchronized historical price DataFrame for all portfolio assets.
        total_capital: Nominal portfolio investment amount in INR.
        benchmark_returns: Synchronized daily return Series for the benchmark.
        max_weight: Maximum allowable weight per asset (e.g. 0.40 for 40%).

    Returns:
        OptimizationResult object detailing status, weights, investments, and metrics.
    """
    n_assets = prices.shape[1]
    tickers = list(prices.columns)

    if n_assets == 0:
        return OptimizationResult(
            success=False,
            message="No assets provided for optimization.",
            weights=np.array([]),
            investments=[],
            annualized_volatility=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            beta=0.0,
            hhi=0.0,
            diversification_ratio=1.0,
            risk_classification="N/A",
            max_weight_constraint=max_weight,
        )

    # 1. Feasibility validation
    feasible, reason = check_optimization_feasibility(n_assets, max_weight)
    if not feasible:
        return OptimizationResult(
            success=False,
            message=reason,
            weights=np.array([1.0 / n_assets] * n_assets),
            investments=allocate_whole_rupees(total_capital, [1.0 / n_assets] * n_assets),
            annualized_volatility=float(np.nan),
            max_drawdown=float(np.nan),
            cagr=float(np.nan),
            beta=float(np.nan),
            hhi=float(np.nan),
            diversification_ratio=float(np.nan),
            risk_classification="Infeasible",
            max_weight_constraint=max_weight,
        )

    # 2. Single asset trivial solution
    if n_assets == 1:
        weights = np.array([1.0])
        investments = [float(total_capital)]
        val_series = calculate_buy_and_hold_portfolio_value(prices, investments)
        d_returns = calculate_portfolio_daily_returns(val_series)
        ann_vol = float(calculate_annualized_volatility(d_returns))
        mdd = calculate_max_drawdown(val_series)
        cagr = calculate_cagr(val_series)
        beta = calculate_beta(d_returns, benchmark_returns)
        return OptimizationResult(
            success=True,
            message="Single asset portfolio (100% allocation).",
            weights=weights,
            investments=investments,
            annualized_volatility=ann_vol,
            max_drawdown=mdd,
            cagr=cagr,
            beta=beta,
            hhi=1.0,
            diversification_ratio=1.0,
            risk_classification=classify_risk(ann_vol),
            max_weight_constraint=max_weight,
            portfolio_value=val_series,
            daily_returns=d_returns,
        )

    # 3. Calculate sample covariance matrix (annualized)
    daily_returns = prices.pct_change().dropna(how="all")
    cov_matrix = daily_returns.cov(ddof=1).values * TRADING_DAYS_PER_YEAR

    # Objective function: portfolio annualized variance
    def portfolio_variance(w: np.ndarray) -> float:
        return float(np.dot(w.T, np.dot(cov_matrix, w)))

    # Constraints and bounds
    # Equality constraint: sum(w_i) - 1.0 = 0
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, float(max_weight)) for _ in range(n_assets))

    # Initial guess: equal weighting
    w0 = np.array([1.0 / n_assets] * n_assets)

    # 4. Numerical Optimization via SLSQP
    try:
        opt_res = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
    except Exception as exc:
        return OptimizationResult(
            success=False,
            message=f"Solver exception during optimization: {exc}",
            weights=w0,
            investments=allocate_whole_rupees(total_capital, w0),
            annualized_volatility=float(np.nan),
            max_drawdown=float(np.nan),
            cagr=float(np.nan),
            beta=float(np.nan),
            hhi=float(np.nan),
            diversification_ratio=float(np.nan),
            risk_classification="Optimization Error",
            max_weight_constraint=max_weight,
        )

    if not opt_res.success:
        return OptimizationResult(
            success=False,
            message=f"Optimization failed: {opt_res.message}",
            weights=w0,
            investments=allocate_whole_rupees(total_capital, w0),
            annualized_volatility=float(np.nan),
            max_drawdown=float(np.nan),
            cagr=float(np.nan),
            beta=float(np.nan),
            hhi=float(np.nan),
            diversification_ratio=float(np.nan),
            risk_classification="Solver Failed",
            max_weight_constraint=max_weight,
        )

    # 5. Numerical sanitization and validation
    raw_weights = opt_res.x
    # Clip slight floating-point deviations
    clean_weights = np.clip(raw_weights, 0.0, max_weight)
    sum_w = np.sum(clean_weights)
    if sum_w <= 0:
        return OptimizationResult(
            success=False,
            message="Optimization resulted in degenerate non-positive weights.",
            weights=w0,
            investments=allocate_whole_rupees(total_capital, w0),
            annualized_volatility=float(np.nan),
            max_drawdown=float(np.nan),
            cagr=float(np.nan),
            beta=float(np.nan),
            hhi=float(np.nan),
            diversification_ratio=float(np.nan),
            risk_classification="Degenerate Solution",
            max_weight_constraint=max_weight,
        )

    # Normalize minor floating-point tolerance
    final_weights = clean_weights / sum_w

    # 6. Discrete whole-rupee allocation matching total_capital exactly
    recommended_investments = allocate_whole_rupees(total_capital, final_weights)

    # 7. Evaluate historical Buy-and-Hold trajectory for optimal portfolio
    opt_val_series = calculate_buy_and_hold_portfolio_value(prices, recommended_investments)
    opt_daily_returns = calculate_portfolio_daily_returns(opt_val_series)
    opt_vol = float(calculate_annualized_volatility(opt_daily_returns))
    opt_mdd = calculate_max_drawdown(opt_val_series)
    opt_cagr = calculate_cagr(opt_val_series)
    opt_beta = calculate_beta(opt_daily_returns, benchmark_returns)
    opt_hhi = calculate_hhi(final_weights)

    indiv_vols = calculate_annualized_volatility(daily_returns).values
    opt_dr = calculate_diversification_ratio(final_weights, indiv_vols, opt_vol)

    return OptimizationResult(
        success=True,
        message="Optimal minimum-variance portfolio solved successfully.",
        weights=final_weights,
        investments=recommended_investments,
        annualized_volatility=opt_vol,
        max_drawdown=opt_mdd,
        cagr=opt_cagr,
        beta=opt_beta,
        hhi=opt_hhi,
        diversification_ratio=opt_dr,
        risk_classification=classify_risk(opt_vol),
        max_weight_constraint=max_weight,
        portfolio_value=opt_val_series,
        daily_returns=opt_daily_returns,
    )
