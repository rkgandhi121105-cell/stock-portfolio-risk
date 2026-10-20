"""
Stock Portfolio Risk Analysis and Stress Testing Using Python
Educational project based on the uploaded proposal.

Run:
    pip install -r requirements.txt
    streamlit run app.py

This tool analyses historical data. It does not predict prices or recommend trades.
"""

from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Stock Portfolio Risk Analyzer", page_icon="📊", layout="wide")

DEFAULT_TICKERS = ["HDFCBANK.NS", "TCS.NS", "ITC.NS", "RELIANCE.NS"]
BENCHMARK = "^NSEI"  # Nifty 50 Yahoo Finance symbol
TRADING_DAYS = 252


def download_adjusted_prices(tickers, start, end):
    """Download each ticker separately to avoid Yahoo Finance MultiIndex/partial-download issues."""
    symbols = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not 4 <= len(symbols) <= 5:
        raise ValueError("Enter between 4 and 5 unique stock ticker symbols.")

    requested = symbols + [BENCHMARK]
    series_by_symbol = {}
    failed = []

    for symbol in requested:
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                failed.append(symbol)
                continue

            # Single-ticker downloads may still use MultiIndex columns in newer yfinance versions.
            if isinstance(raw.columns, pd.MultiIndex):
                close_candidates = [
                    col for col in raw.columns
                    if (("Close" in col) if isinstance(col, tuple) else col == "Close")
                ]
                if not close_candidates:
                    failed.append(symbol)
                    continue
                close = raw[close_candidates[0]]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                if "Close" not in raw.columns:
                    failed.append(symbol)
                    continue
                close = raw["Close"]

            close = pd.to_numeric(close, errors="coerce")
            close.index = pd.to_datetime(close.index)
            close = close[~close.index.duplicated(keep="last")].sort_index()
            close.name = symbol
            if close.dropna().empty:
                failed.append(symbol)
                continue
            series_by_symbol[symbol] = close
        except Exception:
            failed.append(symbol)

    if failed:
        raise ValueError(
            "Yahoo Finance returned no usable prices for: "
            + ", ".join(failed)
            + ". Check each ticker spelling, internet access, and whether Yahoo Finance currently serves that symbol."
        )

    # Outer-join first so we can inspect each symbol's coverage before choosing common dates.
    prices = pd.concat(series_by_symbol.values(), axis=1, join="outer").sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.loc[~prices.index.duplicated(keep="last")]

    # For portfolio returns, use only dates where all chosen stocks and the benchmark have prices.
    aligned = prices[requested].dropna(how="any")
    if len(aligned) < 30:
        coverage = {
            symbol: int(prices[symbol].notna().sum())
            for symbol in requested
        }
        coverage_text = ", ".join(f"{symbol}: {count} rows" for symbol, count in coverage.items())
        raise ValueError(
            f"Only {len(aligned)} common trading dates were found, fewer than the required 30. "
            f"Available rows by ticker: {coverage_text}. Check the ticker symbols and date range."
        )
    return aligned, symbols


def calculate_metrics(prices, tickers, amounts):
    """Calculate weights, returns, volatility, drawdown, beta, HHI and diversification."""
    stock_prices = prices[tickers]
    stock_returns = stock_prices.pct_change().dropna()
    benchmark_returns = prices[BENCHMARK].pct_change().dropna()
    common = stock_returns.index.intersection(benchmark_returns.index)
    stock_returns = stock_returns.loc[common]
    benchmark_returns = benchmark_returns.loc[common]

    amounts = pd.Series(amounts, index=tickers, dtype=float)
    total_value = float(amounts.sum())
    if not np.isfinite(total_value) or total_value <= 0:
        raise ValueError("The total investment amount must be greater than zero.")
    weights = amounts / total_value

    portfolio_returns = stock_returns.mul(weights, axis=1).sum(axis=1)
    portfolio_growth = (1 + portfolio_returns).cumprod()
    portfolio_value = total_value * portfolio_growth
    running_peak = portfolio_value.cummax()
    drawdown = portfolio_value / running_peak - 1

    stock_vol = stock_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    portfolio_vol = portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    stock_drawdowns = {}
    for ticker in tickers:
        growth = (1 + stock_returns[ticker]).cumprod()
        stock_drawdowns[ticker] = float((growth / growth.cummax() - 1).min())

    benchmark_var = benchmark_returns.var(ddof=1)
    betas = {}
    for ticker in tickers:
        cov = stock_returns[ticker].cov(benchmark_returns)
        betas[ticker] = float(cov / benchmark_var) if benchmark_var and not np.isnan(benchmark_var) else np.nan

    hhi = float((weights ** 2).sum())
    weighted_stock_vol = float((weights * stock_vol.reindex(tickers)).sum())
    diversification_ratio = weighted_stock_vol / portfolio_vol if portfolio_vol > 0 else np.nan

    metrics = pd.DataFrame({
        "Ticker": tickers + ["PORTFOLIO"],
        "Investment (₹)": [amounts[t] for t in tickers] + [total_value],
        "Weight (%)": [weights[t] * 100 for t in tickers] + [100.0],
        "Annualised volatility (%)": [stock_vol[t] * 100 for t in tickers] + [portfolio_vol * 100],
        "Maximum drawdown (%)": [stock_drawdowns[t] * 100 for t in tickers] + [float(drawdown.min()) * 100],
        "Beta vs Nifty 50": [betas[t] for t in tickers] + [float(portfolio_returns.cov(benchmark_returns) / benchmark_var) if benchmark_var else np.nan],
    })
    corr = stock_returns.corr()
    # Equal-weighted portfolio, evaluated over the same aligned return dates.
    equal_weights = pd.Series(1 / len(tickers), index=tickers)
    equal_returns = stock_returns.mul(equal_weights, axis=1).sum(axis=1)
    equal_vol = float(equal_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    equal_growth = (1 + equal_returns).cumprod()
    equal_dd = float((equal_growth / equal_growth.cummax() - 1).min())
    equal_hhi = float((equal_weights ** 2).sum())

    return {
        "amounts": amounts, "weights": weights, "total_value": total_value,
        "stock_returns": stock_returns, "benchmark_returns": benchmark_returns,
        "portfolio_returns": portfolio_returns, "portfolio_value": portfolio_value,
        "drawdown": drawdown, "stock_vol": stock_vol, "portfolio_vol": float(portfolio_vol),
        "stock_drawdowns": stock_drawdowns, "betas": betas, "hhi": hhi,
        "diversification_ratio": float(diversification_ratio) if np.isfinite(diversification_ratio) else np.nan,
        "metrics": metrics, "corr": corr, "equal_vol": equal_vol,
        "equal_drawdown": equal_dd, "equal_hhi": equal_hhi, "equal_returns": equal_returns,
    }


def classify_risk(volatility, hhi):
    """Transparent educational bands, not universal investment standards."""
    vol_pct = volatility * 100
    if vol_pct < 15 and hhi < 0.25:
        return "Low", "Project-defined rule: annualised volatility < 15% and HHI < 0.25."
    if vol_pct >= 25 or hhi >= 0.40:
        return "High", "Project-defined rule: annualised volatility ≥ 25% or HHI ≥ 0.40."
    return "Moderate", "All other combinations under the project's illustrative rules."


def stress_test(amounts, weights):
    """Apply proposal's simplified shocks to current hypothetical portfolio value."""
    total = float(amounts.sum())
    rows = []
    for shock in (-0.10, -0.20):
        loss = total * abs(shock)
        rows.append({
            "Scenario": f"All holdings decline {shock:.0%}",
            "Assumption": f"Every holding changes by {shock:.0%}",
            "Estimated loss (₹)": loss,
            "Loss (%)": abs(shock) * 100,
            "Estimated remaining value (₹)": total - loss,
        })
    largest = weights.idxmax()
    loss = float(amounts[largest]) * 0.30
    rows.append({
        "Scenario": f"Largest holding ({largest}) declines 30%",
        "Assumption": f"{largest} falls 30%; all other holdings unchanged",
        "Estimated loss (₹)": loss,
        "Loss (%)": loss / total * 100,
        "Estimated remaining value (₹)": total - loss,
    })
    return pd.DataFrame(rows)



def simulate_future_values(initial_amount, daily_returns, years, n_sims=3000, seed=42):
    """
    Historical-return bootstrap simulation. This is a scenario model, not an AI forecast.
    Samples observed daily portfolio returns with replacement.
    """
    clean = pd.Series(daily_returns).dropna().to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 30:
        raise ValueError("Not enough historical portfolio returns for a projection.")
    steps = int(TRADING_DAYS * years)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean, size=(n_sims, steps), replace=True)
    # Avoid mathematically invalid values if extreme historical data exists.
    sampled = np.clip(sampled, -0.99, None)
    outcomes = initial_amount * np.prod(1.0 + sampled, axis=1)
    outcomes = outcomes[np.isfinite(outcomes)]
    if len(outcomes) == 0:
        raise ValueError("Projection simulation did not produce valid outcomes.")
    return {
        "pessimistic": float(np.percentile(outcomes, 10)),
        "median": float(np.percentile(outcomes, 50)),
        "optimistic": float(np.percentile(outcomes, 90)),
        "mean": float(np.mean(outcomes)),
        "outcomes": outcomes,
    }

def fmt_inr(value):
    return f"₹{value:,.2f}"


st.title("📊 Stock Portfolio Risk Analysis & Stress Testing")
st.caption("Historical analysis of NSE-listed stocks. Educational decision-support tool, not financial advice.")

with st.sidebar:
    st.header("Portfolio inputs")
    st.caption("Add or remove holdings below. Each ticker has its own hypothetical investment amount.")
    if "holding_rows" not in st.session_state:
        st.session_state.holding_rows = [
            {"ticker": "HDFCBANK.NS", "amount": 25000.0},
            {"ticker": "TCS.NS", "amount": 25000.0},
            {"ticker": "ITC.NS", "amount": 25000.0},
            {"ticker": "RELIANCE.NS", "amount": 25000.0},
        ]

    add_col, remove_col = st.columns(2)
    if add_col.button("＋ Add stock", use_container_width=True):
        if len(st.session_state.holding_rows) < 5:
            st.session_state.holding_rows.append({"ticker": "", "amount": 1000.0})
        else:
            st.warning("The project supports up to 5 stocks.")
    if remove_col.button("− Remove last", use_container_width=True):
        if len(st.session_state.holding_rows) > 4:
            st.session_state.holding_rows.pop()
        else:
            st.warning("Keep at least 4 stocks for this project.")

    tickers = []
    amounts = {}
    for i, row in enumerate(st.session_state.holding_rows):
        st.markdown(f"**Stock {i + 1}**")
        ticker_value = st.text_input(
            "Ticker symbol",
            value=row["ticker"],
            key=f"ticker_input_{i}",
            placeholder="e.g. INFY.NS",
        ).strip().upper()
        amount_value = st.number_input(
            "Investment amount (₹)",
            min_value=0.0,
            value=float(row["amount"]),
            step=100.0,
            key=f"amount_input_{i}",
        )
        row["ticker"] = ticker_value
        row["amount"] = amount_value
        if ticker_value:
            tickers.append(ticker_value)
            amounts[ticker_value] = amount_value

    st.divider()
    st.subheader("Projection simulator")
    projection_amount = st.number_input(
        "Amount to simulate (₹)", min_value=1.0, value=1000.0, step=100.0,
        key="projection_amount"
    )
    projection_years = st.selectbox(
        "Projection horizon", options=[1, 2, 3, 5, 10], index=2,
        format_func=lambda x: f"{x} year{'s' if x != 1 else ''}",
        key="projection_years"
    )
    simulations = st.select_slider(
        "Simulation runs", options=[1000, 3000, 5000], value=3000,
        key="simulation_runs"
    )
    end_date = st.date_input("End date (exclusive)", value=date.today())
    start_date = st.date_input("Start date", value=end_date - timedelta(days=365 * 5))
    run = st.button("Run portfolio analysis", type="primary", use_container_width=True)

st.info("Default sample tickers are illustrative. Check ticker validity and data availability before interpreting results.")

if run:
    if len(set(tickers)) != len(tickers) or not 4 <= len(tickers) <= 5 or any(not t for t in tickers):
        st.error("Please enter 4–5 unique, non-empty ticker symbols. Use Add stock or Remove last to change the number of holdings.")
        st.stop()
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()
    if any(v < 0 for v in amounts.values()) or sum(amounts.values()) <= 0:
        st.error("Amounts must be non-negative and total investment must be greater than zero.")
        st.stop()

    with st.spinner("Downloading historical data and calculating risk measures..."):
        try:
            prices, valid_tickers = download_adjusted_prices(tickers, start_date, end_date)
            # Amounts mapped to the actual unique symbols in the order used for analysis.
            chosen_amounts = [amounts[t] for t in valid_tickers]
            result = calculate_metrics(prices, valid_tickers, chosen_amounts)
        except Exception as exc:
            st.error(f"Could not complete analysis: {exc}")
            st.stop()

    st.caption(
        f"Data window: {prices.index.min().date()} to {prices.index.max().date()} | "
        f"Valid aligned price observations: {len(prices)} | "
        f"Retrieved: {date.today().isoformat()} | Benchmark: {BENCHMARK}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total hypothetical investment", fmt_inr(result["total_value"]))
    c2.metric("Portfolio annualised volatility", f'{result["portfolio_vol"]:.2%}')
    c3.metric("Maximum drawdown", f'{result["drawdown"].min():.2%}')
    c4.metric("HHI concentration", f'{result["hhi"]:.3f}')

    risk_label, risk_rule = classify_risk(result["portfolio_vol"], result["hhi"])
    st.subheader(f"Rule-based risk summary: {risk_label}")
    st.write(risk_rule)
    st.warning("This Low / Moderate / High label uses illustrative project-defined thresholds, not universal risk standards.")

    st.subheader("1. Portfolio and stock metrics")
    st.dataframe(result["metrics"].round(3), use_container_width=True, hide_index=True)

    st.subheader("2. Portfolio allocation")
    fig, ax = plt.subplots()
    ax.pie(result["weights"].values, labels=result["weights"].index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Hypothetical portfolio weights")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Historical portfolio value and drawdown")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        result["portfolio_value"].plot(ax=ax)
        ax.set_title("Illustrative growth of ₹1 invested at the portfolio weights")
        ax.set_ylabel("Value (₹)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        (result["drawdown"] * 100).plot(ax=ax)
        ax.set_title("Portfolio drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("4. Correlation between stock returns")
    fig, ax = plt.subplots()
    sns.heatmap(result["corr"], annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Daily return correlation matrix")
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"**Diversification Ratio:** {result['diversification_ratio']:.3f}" if np.isfinite(result["diversification_ratio"]) else "**Diversification Ratio:** unavailable")
    st.caption("Correlation describes historical co-movement. Neither correlation nor the Diversification Ratio guarantees future diversification.")

    st.subheader("5. Stress testing")
    stress = stress_test(result["amounts"], result["weights"])
    st.dataframe(stress.style.format({
        "Estimated loss (₹)": "₹{:,.2f}",
        "Loss (%)": "{:.2f}%",
        "Estimated remaining value (₹)": "₹{:,.2f}",
    }), use_container_width=True, hide_index=True)
    st.caption("Stress scenarios are hypothetical. They are not forecasts and do not model changing correlations, taxes, transaction costs, or market liquidity.")

    st.subheader("6. User allocation vs equal-weight allocation")
    comparison = pd.DataFrame([
        {"Allocation": "User-entered", "Annualised volatility (%)": result["portfolio_vol"] * 100,
         "Maximum drawdown (%)": result["drawdown"].min() * 100, "HHI": result["hhi"]},
        {"Allocation": "Equal-weighted", "Annualised volatility (%)": result["equal_vol"] * 100,
         "Maximum drawdown (%)": result["equal_drawdown"] * 100, "HHI": result["equal_hhi"]},
    ])
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
    st.caption("Equal weighting is a comparison baseline, not an optimised or recommended portfolio.")

    st.subheader("7. Investment amount projection simulator")
    st.write(
        "Choose any starting amount, such as ₹10, ₹1,000, ₹10,000, or ₹20,000. "
        "The simulator resamples historical daily portfolio returns to show a range of hypothetical future values."
    )
    try:
        projection = simulate_future_values(
            projection_amount, result["portfolio_returns"], projection_years, simulations
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Lower scenario (10th percentile)", fmt_inr(projection["pessimistic"]),
                  delta=f"{projection['pessimistic'] - projection_amount:,.2f} vs starting amount")
        p2.metric("Middle scenario (median)", fmt_inr(projection["median"]),
                  delta=f"{projection['median'] - projection_amount:,.2f} vs starting amount")
        p3.metric("Upper scenario (90th percentile)", fmt_inr(projection["optimistic"]),
                  delta=f"{projection['optimistic'] - projection_amount:,.2f} vs starting amount")
        projection_table = pd.DataFrame([
            {"Scenario": "Lower historical simulation (10th percentile)", "Starting amount (₹)": projection_amount,
             "Illustrative future value (₹)": projection["pessimistic"],
             "Profit / loss (₹)": projection["pessimistic"] - projection_amount,
             "Return (%)": (projection["pessimistic"] / projection_amount - 1) * 100},
            {"Scenario": "Median historical simulation", "Starting amount (₹)": projection_amount,
             "Illustrative future value (₹)": projection["median"],
             "Profit / loss (₹)": projection["median"] - projection_amount,
             "Return (%)": (projection["median"] / projection_amount - 1) * 100},
            {"Scenario": "Upper historical simulation (90th percentile)", "Starting amount (₹)": projection_amount,
             "Illustrative future value (₹)": projection["optimistic"],
             "Profit / loss (₹)": projection["optimistic"] - projection_amount,
             "Return (%)": (projection["optimistic"] / projection_amount - 1) * 100},
        ])
        st.dataframe(projection_table.style.format({
            "Starting amount (₹)": "₹{:,.2f}",
            "Illustrative future value (₹)": "₹{:,.2f}",
            "Profit / loss (₹)": "₹{:,.2f}",
            "Return (%)": "{:.2f}%",
        }), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots()
        ax.hist(projection["outcomes"], bins=40)
        ax.axvline(projection_amount, linestyle="--", label="Starting amount")
        ax.set_title(f"Simulated future values after {projection_years} year(s)")
        ax.set_xlabel("Illustrative future portfolio value (₹)")
        ax.set_ylabel("Simulation count")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Important: this is a historical bootstrap simulation, not an AI prediction or a guarantee. "
            "It assumes historical daily returns are a useful sample of possible future returns; real markets can behave differently. "
            "The model does not know your financial situation and does not recommend an investment amount."
        )
    except Exception as exc:
        st.warning(f"Projection unavailable: {exc}")

    st.subheader("8. Download results")
    csv_bytes = result["metrics"].to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics as CSV", data=csv_bytes, file_name="portfolio_risk_metrics.csv", mime="text/csv")
    stress_csv = stress.to_csv(index=False).encode("utf-8")
    st.download_button("Download stress tests as CSV", data=stress_csv, file_name="portfolio_stress_tests.csv", mime="text/csv")

    report = f"""STOCK PORTFOLIO RISK ANALYSIS AND STRESS TESTING
Data window: {prices.index.min().date()} to {prices.index.max().date()}
Retrieved on: {date.today().isoformat()}
Benchmark: Nifty 50 ({BENCHMARK})
Tickers: {', '.join(valid_tickers)}
Valid aligned price observations: {len(prices)}

Total hypothetical investment: {fmt_inr(result['total_value'])}
Annualised portfolio volatility: {result['portfolio_vol']:.2%}
Maximum drawdown: {result['drawdown'].min():.2%}
HHI concentration: {result['hhi']:.4f}
Diversification Ratio: {result['diversification_ratio']:.4f}
Project-defined risk label: {risk_label}
Classification rule: {risk_rule}

STOCK METRICS
{result['metrics'].to_string(index=False)}

STRESS TESTS
{stress.to_string(index=False)}

LIMITATIONS
Historical metrics may not represent future conditions. Stress scenarios are hypothetical.
This educational tool does not predict prices or provide buy/sell recommendations.
"""
    st.download_button("Download text report", data=report, file_name="portfolio_risk_report.txt", mime="text/plain")
else:
    st.markdown("""
    ### What this project calculates
    - Daily returns and portfolio weights
    - Annualised volatility and maximum drawdown
    - Beta relative to the Nifty 50
    - HHI concentration, return correlations and Diversification Ratio
    - Hypothetical market-wide shocks of −10% and −20%, plus a −30% shock to the largest holding
    - Comparison against an equal-weighted allocation
    - Tables, charts, and downloadable CSV/text reports

    Enter 4–5 ticker symbols and hypothetical amounts in the sidebar, then select **Run portfolio analysis**.
    """)
