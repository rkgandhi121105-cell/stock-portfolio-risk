"""Financial risk metrics and portfolio mathematics module.

Implements rigorous calculations for returns, annualized volatility, maximum drawdown,
benchmark beta, concentration index (HHI), correlation matrix, diversification ratio,
and Compounded Annual Growth Rate (CAGR).

Return Methodology:
A buy-and-hold portfolio model is applied. At inception (t_0), initial capital is
allocated into individual equities to establish share quantities. Portfolio value
evolves as the sum of individual equity valuations through time. Daily portfolio
returns are derived directly from this cumulative valuation series.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from portfolio_risk.config import (
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_THRESHOLDS,
    HHI_THRESHOLDS,
)


@dataclass
class MetricSummary:
    """Structured container holding all primary portfolio risk statistics."""
    total_investment: float
    num_holdings: int
    annualized_volatility: float
    max_drawdown: float
    cagr: float
    beta: float
    hhi: float
    diversification_ratio: float
    risk_classification: str
    largest_holding_ticker: str
    largest_holding_pct: float
    largest_holding_amount: float
    actual_start_date: str
    actual_end_date: str


def calculate_portfolio_weights(investments: Sequence[float]) -> np.ndarray:
    """Calculate normalized portfolio weights from nominal investment allocations.

    Args:
        investments: List or array of monetary allocations.

    Returns:
        1D numpy array of weights summing to 1.0.

    Raises:
        ValueError: If total investment is non-positive or empty.
    """
    inv = np.asarray(investments, dtype=float)
    total = np.sum(inv)
    if total <= 0:
        raise ValueError("Total investment amount must be strictly greater than zero.")
    return inv / total


def allocate_whole_rupees(
    total_capital: float,
    weights: Sequence[float],
) -> List[float]:
    """Deterministically convert continuous weights into discrete whole-rupee amounts.

    Uses the Largest Remainder Method (Hare-Niemeyer / Hamilton quota method)
    to guarantee that the sum of recommended integer investments matches
    total_capital to the exact whole rupee.

    Args:
        total_capital: Total capital in Indian Rupees (INR).
        weights: Normalized asset allocation weights summing to 1.0.

    Returns:
        List of whole-rupee investment amounts whose sum equals round(total_capital).
    """
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return []

    total_int = int(round(total_capital))
    unrounded = w * total_int
    floored = np.floor(unrounded).astype(int)
    remainder = unrounded - floored

    shortfall = total_int - int(np.sum(floored))
    if shortfall > 0:
        # Give +1 to the assets with the largest fractional remainders
        indices_by_remainder = np.argsort(-remainder)
        for i in range(shortfall):
            idx = indices_by_remainder[i % len(w)]
            floored[idx] += 1

    return [float(x) for x in floored]


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage daily returns from a price DataFrame.

    Formula: r_t = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices: Historical price DataFrame with dates as index.

    Returns:
        pd.DataFrame of percentage daily returns (first row dropped).
    """
    returns = prices.pct_change().dropna(how="all")
    return returns


def calculate_buy_and_hold_portfolio_value(
    prices: pd.DataFrame,
    investments: Sequence[float],
) -> pd.Series:
    """Calculate cumulative daily portfolio value under a buy-and-hold methodology.

    At index t=0 (the first observation date), capital I_i is deployed into each
    asset i. The number of shares held is n_i = I_i / P_{i,0}.
    For each subsequent day t, the portfolio value is V_t = sum(n_i * P_{i,t}).

    Args:
        prices: Historical price DataFrame for all portfolio equities.
        investments: Sequence of initial monetary amounts per stock.

    Returns:
        pd.Series of portfolio valuation through time.
    """
    if prices.empty:
        raise ValueError("Price DataFrame cannot be empty.")

    p0 = prices.iloc[0].values
    if np.any(p0 <= 0):
        raise ValueError("Initial prices must be strictly positive.")

    inv = np.asarray(investments, dtype=float)
    if len(inv) != prices.shape[1]:
        raise ValueError("Length of investments must match number of price columns.")

    # Number of hypothetical shares allocated
    shares = inv / p0

    # Valuation matrix: daily prices * shares
    daily_stock_values = prices.values * shares
    portfolio_value = pd.Series(
        np.sum(daily_stock_values, axis=1),
        index=prices.index,
        name="Portfolio Value",
    )
    return portfolio_value


def calculate_portfolio_daily_returns(portfolio_value: pd.Series) -> pd.Series:
    """Derive daily returns from a buy-and-hold portfolio value series.

    Args:
        portfolio_value: pd.Series of daily portfolio capital values.

    Returns:
        pd.Series of percentage returns.
    """
    returns = portfolio_value.pct_change().dropna()
    returns.name = "Portfolio Return"
    return returns


def calculate_cagr(value_series: pd.Series) -> float:
    """Calculate Compounded Annual Growth Rate (CAGR) from a historical valuation series.

    Formula: CAGR = (V_end / V_start) ** (1 / T) - 1
    Where T is elapsed calendar time in years (days / 365.25).

    Args:
        value_series: Chronologically indexed price or capital valuation Series.

    Returns:
        CAGR as a decimal (e.g. 0.145 for 14.5%), or NaN if period or values are invalid.
    """
    if value_series.empty or len(value_series) < 2:
        return float(np.nan)

    v_start = float(value_series.iloc[0])
    v_end = float(value_series.iloc[-1])

    if v_start <= 0 or v_end <= 0:
        return float(np.nan)

    start_date = value_series.index[0]
    end_date = value_series.index[-1]
    days = (end_date - start_date).days
    if days <= 0:
        return float(np.nan)

    t_years = days / 365.25
    cagr = (v_end / v_start) ** (1.0 / t_years) - 1.0
    return float(cagr)


def calculate_annualized_volatility(
    returns: pd.Series | pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float | pd.Series:
    """Compute annualized volatility (standard deviation of daily returns scaled by sqrt(252)).

    Formula: sigma_annual = sigma_daily * sqrt(N_days)

    Args:
        returns: Daily returns as a Series (single asset/portfolio) or DataFrame (multiple stocks).
        trading_days: Number of trading sessions per calendar year (default: 252).

    Returns:
        Annualized volatility as a float (for Series) or Series (for DataFrame).
    """
    if len(returns) < 2:
        if isinstance(returns, pd.DataFrame):
            return pd.Series(np.nan, index=returns.columns)
        return float(np.nan)

    # ddof=1 applies Bessel's correction for unbiased sample standard deviation
    daily_vol = returns.std(ddof=1)
    annualized_vol = daily_vol * np.sqrt(trading_days)
    return annualized_vol


def calculate_drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    """Calculate the historical drawdown series for a portfolio value trajectory.

    Formula:
        Peak_t = max_{s <= t} V_s
        Drawdown_t = (V_t - Peak_t) / Peak_t

    Args:
        portfolio_value: Series of cumulative portfolio valuation.

    Returns:
        pd.Series of percentage drawdowns (values <= 0.0).
    """
    if portfolio_value.empty:
        return pd.Series(dtype=float)

    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    drawdown.name = "Drawdown"
    return drawdown


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """Compute the maximum historical peak-to-trough decline (Maximum Drawdown).

    Args:
        portfolio_value: Series of cumulative portfolio values.

    Returns:
        Maximum drawdown as a negative float (e.g. -0.185 for an 18.5% drop).
    """
    drawdown_series = calculate_drawdown_series(portfolio_value)
    if drawdown_series.empty:
        return 0.0
    return float(drawdown_series.min())


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate the portfolio Beta (systematic market risk sensitivity) relative to benchmark.

    Formula: Beta = Cov(r_p, r_m) / Var(r_m)

    Args:
        portfolio_returns: Daily returns of the portfolio.
        benchmark_returns: Daily returns of the market index (Nifty 50).

    Returns:
        Beta coefficient as a float. Returns 0.0 if benchmark variance is zero.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float(np.nan)

    rp = aligned.iloc[:, 0]
    rm = aligned.iloc[:, 1]

    cov_matrix = np.cov(rp, rm, ddof=1)
    cov_pm = cov_matrix[0, 1]
    var_m = cov_matrix[1, 1]

    if var_m <= 1e-12 or np.isnan(var_m):
        return 0.0

    beta = cov_pm / var_m
    return float(beta)


def calculate_hhi(weights: Sequence[float]) -> float:
    """Calculate the Herfindahl-Hirschman Index (HHI) for portfolio concentration.

    Formula: HHI = sum(w_i^2)
    For decimal weights (summing to 1.0), HHI ranges from 1/N (equal diversification)
    to 1.0 (complete single-stock concentration).

    Args:
        weights: Sequence of normalized portfolio weights summing to 1.0.

    Returns:
        HHI value between 0.0 and 1.0.
    """
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return 0.0
    return float(np.sum(w ** 2))


def interpret_hhi(hhi: float) -> str:
    """Provide a human-readable interpretation of the HHI concentration score.

    Args:
        hhi: Concentration index between 0.0 and 1.0.

    Returns:
        Textual classification: 'Highly Diversified', 'Moderately Concentrated',
        or 'Highly Concentrated'.
    """
    if hhi <= HHI_THRESHOLDS["LOW_CONCENTRATION"]:
        return "Highly Diversified"
    elif hhi <= HHI_THRESHOLDS["MODERATE_CONCENTRATION"]:
        return "Moderately Concentrated"
    else:
        return "Highly Concentrated"


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Pearson correlation matrix of daily equity returns.

    Args:
        returns: DataFrame of daily percentage returns per stock.

    Returns:
        Symmetric NxN correlation DataFrame with values in [-1.0, 1.0].
    """
    return returns.corr(method="pearson")


def calculate_diversification_ratio(
    weights: Sequence[float],
    individual_annualized_volatilities: Sequence[float],
    portfolio_annualized_volatility: float,
) -> float:
    """Calculate the Diversification Ratio (DR) of the portfolio.

    Formula: DR = (sum(w_i * sigma_i)) / sigma_p
    A higher diversification ratio (> 1.0) reflects risk reduction achieved
    through imperfectly correlated assets. For a 1-holding basket, DR = 1.0.

    Args:
        weights: Asset weights (sum = 1.0).
        individual_annualized_volatilities: Volatility for each holding.
        portfolio_annualized_volatility: Volatility of the consolidated portfolio.

    Returns:
        Diversification ratio as a float.
    """
    if portfolio_annualized_volatility <= 1e-12:
        return 1.0

    w = np.asarray(weights, dtype=float)
    vols = np.asarray(individual_annualized_volatilities, dtype=float)

    weighted_vols = np.sum(w * vols)
    dr = weighted_vols / portfolio_annualized_volatility
    return float(dr)


def classify_risk(annualized_volatility: float) -> str:
    """Classify portfolio risk based on annualized volatility thresholds.

    Thresholds:
        - Low Risk: Volatility < 15%
        - Moderate Risk: 15% <= Volatility < 25%
        - High Risk: Volatility >= 25%

    Args:
        annualized_volatility: Annualized standard deviation of returns.

    Returns:
        'Low', 'Moderate', or 'High'.
    """
    if annualized_volatility < VOLATILITY_THRESHOLDS["LOW_MAX"]:
        return "Low"
    elif annualized_volatility < VOLATILITY_THRESHOLDS["MODERATE_MAX"]:
        return "Moderate"
    else:
        return "High"


def calculate_equal_weight_portfolio(
    prices: pd.DataFrame,
    total_capital: float,
    benchmark_returns: pd.Series,
) -> Dict[str, object]:
    """Compute risk metrics for an equivalent benchmark Equal-Weight portfolio.

    Allocates total_capital equally across all N assets (w_i = 1/N) and evaluates
    historical buy-and-hold performance over the identical time horizon.

    Args:
        prices: Synchronized historical equity prices DataFrame.
        total_capital: User's total investment capital.
        benchmark_returns: Synchronized benchmark daily returns.

    Returns:
        Dictionary containing equal-weight portfolio metrics and series.
    """
    n_assets = prices.shape[1]
    if n_assets == 0:
        raise ValueError("Cannot calculate equal-weight portfolio with zero assets.")

    ew_weights = np.array([1.0 / n_assets] * n_assets)
    ew_investments = allocate_whole_rupees(total_capital, ew_weights)

    ew_value = calculate_buy_and_hold_portfolio_value(prices, ew_investments)
    ew_daily_returns = calculate_portfolio_daily_returns(ew_value)
    ew_annual_vol = float(calculate_annualized_volatility(ew_daily_returns))
    ew_max_dd = calculate_max_drawdown(ew_value)
    ew_cagr = calculate_cagr(ew_value)
    ew_beta = calculate_beta(ew_daily_returns, benchmark_returns)
    ew_hhi = calculate_hhi(ew_weights)

    stock_returns = calculate_daily_returns(prices)
    indiv_vols = calculate_annualized_volatility(stock_returns).values
    ew_dr = calculate_diversification_ratio(ew_weights, indiv_vols, ew_annual_vol)

    return {
        "investments": ew_investments,
        "weights": ew_weights,
        "portfolio_value": ew_value,
        "daily_returns": ew_daily_returns,
        "annualized_volatility": ew_annual_vol,
        "max_drawdown": ew_max_dd,
        "cagr": ew_cagr,
        "beta": ew_beta,
        "hhi": ew_hhi,
        "diversification_ratio": ew_dr,
        "risk_classification": classify_risk(ew_annual_vol),
    }


def analyze_portfolio(
    prices: pd.DataFrame,
    investments: Sequence[float],
    benchmark_prices: pd.Series,
) -> Dict[str, object]:
    """Execute complete end-to-end historical portfolio risk analysis.

    Args:
        prices: Historical price DataFrame for portfolio assets.
        investments: List of nominal capital allocations.
        benchmark_prices: Historical price Series for the benchmark index.

    Returns:
        Comprehensive dictionary containing values, return series, individual statistics,
        aggregate summary metrics, and equal-weight comparison data.
    """
    tickers = list(prices.columns)
    weights = calculate_portfolio_weights(investments)
    total_capital = float(np.sum(investments))

    # Calculate returns and portfolio value trajectories
    stock_returns = calculate_daily_returns(prices)
    benchmark_returns = calculate_daily_returns(benchmark_prices.to_frame()).iloc[:, 0]
    portfolio_value = calculate_buy_and_hold_portfolio_value(prices, investments)
    portfolio_daily_returns = calculate_portfolio_daily_returns(portfolio_value)

    # Core statistics
    individual_vols = calculate_annualized_volatility(stock_returns)
    portfolio_vol = float(calculate_annualized_volatility(portfolio_daily_returns))
    max_dd = calculate_max_drawdown(portfolio_value)
    cagr = calculate_cagr(portfolio_value)
    beta = calculate_beta(portfolio_daily_returns, benchmark_returns)
    hhi = calculate_hhi(weights)
    hhi_label = interpret_hhi(hhi)
    correlation_mat = calculate_correlation_matrix(stock_returns)
    div_ratio = calculate_diversification_ratio(weights, individual_vols.values, portfolio_vol)
    risk_class = classify_risk(portfolio_vol)

    # Stock-level metrics: individual CAGRs and Betas
    stock_cagrs = {t: calculate_cagr(prices[t]) for t in tickers}
    stock_betas = {t: calculate_beta(stock_returns[t], benchmark_returns) for t in tickers}

    # Largest holding identification
    largest_idx = int(np.argmax(investments))
    largest_ticker = tickers[largest_idx]
    largest_amt = float(investments[largest_idx])
    largest_pct = float(weights[largest_idx] * 100.0)

    # Dates
    start_str = prices.index[0].strftime("%d %b %Y")
    end_str = prices.index[-1].strftime("%d %b %Y")

    # Individual holdings summary table data
    holdings_df = pd.DataFrame({
        "Ticker": tickers,
        "Investment (₹)": list(investments),
        "Weight (%)": [w * 100 for w in weights],
        "Annualized Volatility (%)": [individual_vols[t] * 100 for t in tickers],
        "Beta (^NSEI)": [stock_betas[t] for t in tickers],
        "CAGR (%)": [stock_cagrs[t] * 100 for t in tickers],
    })

    # Summary container
    summary = MetricSummary(
        total_investment=total_capital,
        num_holdings=len(tickers),
        annualized_volatility=portfolio_vol,
        max_drawdown=max_dd,
        cagr=cagr,
        beta=beta,
        hhi=hhi,
        diversification_ratio=div_ratio,
        risk_classification=risk_class,
        largest_holding_ticker=largest_ticker,
        largest_holding_pct=largest_pct,
        largest_holding_amount=largest_amt,
        actual_start_date=start_str,
        actual_end_date=end_str,
    )

    # Equal-weight comparison
    equal_weight = calculate_equal_weight_portfolio(prices, total_capital, benchmark_returns)

    return {
        "tickers": tickers,
        "weights": weights,
        "total_capital": total_capital,
        "holdings_df": holdings_df,
        "stock_prices": prices,
        "stock_returns": stock_returns,
        "benchmark_prices": benchmark_prices,
        "benchmark_returns": benchmark_returns,
        "portfolio_value": portfolio_value,
        "portfolio_daily_returns": portfolio_daily_returns,
        "individual_volatilities": individual_vols,
        "individual_cagrs": stock_cagrs,
        "individual_betas": stock_betas,
        "summary": summary,
        "correlation_matrix": correlation_mat,
        "equal_weight": equal_weight,
        "hhi_label": hhi_label,
        "actual_start_date": start_str,
        "actual_end_date": end_str,
    }
