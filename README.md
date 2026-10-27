# Stock Portfolio Risk Analysis & Stress Testing Terminal

An academic-grade financial risk engineering and portfolio optimization dashboard for National Stock Exchange (NSE) equity portfolios, built with Python, Streamlit, SciPy, and a visual interface inspired by Zerodha Kite.

---

## 1. Clone and Run Locally

Follow these step-by-step instructions for macOS (zsh terminal) or Linux:

### Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd stock-portfolio-risk
```
> **Note:** Replace `<YOUR_REPOSITORY_URL>` with your actual repository URL.

### Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### Run the Application
```bash
streamlit run app.py
```
After launching, open your browser at:
```text
http://localhost:8501
```

### Run Automated Tests
```bash
python -m pytest -v
```

---

## 2. Docker Usage (Optional)

To run the application in an isolated Docker container:

```bash
docker build -t stock-portfolio-risk .
docker run --rm -p 8501:8501 stock-portfolio-risk
```

- Access the dashboard at: `http://localhost:8501`
- Stop the container at any time by pressing `Ctrl+C` or running `docker stop <container_id>`.

---

## 3. Project Overview

This application is an educational, quantitative risk terminal designed for equity portfolios listed on the National Stock Exchange of India (NSE). It evaluates downside tail risk, co-movement relative to the Indian market benchmark (Nifty 50, `^NSEI`), asset concentration, and capital solvency under instantaneous market shocks.

In addition to traditional historical risk analysis, the platform provides constrained numerical portfolio optimization using **SciPy SLSQP** to compute an analytical **Minimum-Variance Portfolio**, benchmarks it against a naive **Equal-Weight Portfolio**, and presents a side-by-side **Three-Way Before-and-After Comparison**.

> **Academic Disclaimer:** This software is developed strictly for educational and academic risk analysis. It does **not** predict future stock prices, nor does it provide financial advice, buy/sell recommendations, or guaranteed investment strategies.

---

## 4. Key Features

1. **Dynamic Holdings Management (1 to 20 Stocks):**
   - Interactive UI to add or remove holdings dynamically.
   - Real-time input validation for NSE ticker formats (e.g. `RELIANCE.NS`, `INFY.NS`).
   - Rejection of invalid, empty, or duplicate tickers.
   - Non-zero positive capital enforcement per holding.

2. **Historical Market Data Retrieval:**
   - Synchronized historical adjusted closing prices fetched via `yfinance` for selected stocks and the Nifty 50 benchmark (`^NSEI`).
   - Configurable lookback horizon: **5 Years**, **10 Years**, or **15 Years**.
   - Explicit reporting of requested horizon alongside actual synchronized start and end dates.
   - Graceful handling of MultiIndex DataFrame variations from Yahoo Finance.

3. **Stock-Level Financial Metrics:**
   - Individual Nominal Investment (₹) and Portfolio Weight (%).
   - Individual Annualized Volatility ($\sigma_{\text{daily}} \times \sqrt{252}$).
   - Individual Systematic Beta ($\beta$) relative to Nifty 50.
   - Individual Compounded Annual Growth Rate (CAGR).
   - Displayed using native Streamlit DataFrames with right-aligned currency and percentage formatting.

4. **Portfolio-Level Summary Metric Cards:**
   - Total Investment Capital.
   - Annualized Portfolio Volatility.
   - Historical Maximum Drawdown (MDD).
   - Largest Holding (ticker and weight).
   - Herfindahl-Hirschman Index (HHI) Concentration.

5. **Inter-Asset Correlation & Allocation Visualizations:**
   - Responsive allocation donut chart with legend for $\le 6$ holdings; clean horizontal bar chart for $> 6$ holdings to eliminate label overlap.
   - Inter-asset Pearson correlation heatmap with single-asset fallback.
   - Cumulative historical portfolio growth vs. Nifty 50 benchmark chart.
   - Underwater drawdown trajectory chart.

6. **Hypothetical Instantaneous Stress Testing:**
   - 4 Predefined Market-Wide Scenarios:
     - Mild Market Pullback (-5%)
     - Broad Market Correction (-10%)
     - Deep Market Correction (-15%)
     - Severe Market Crash (-20%)
   - Selectable Individual-Stock Shock: choose any portfolio holding and enter a custom percentage decline while all other holdings remain unchanged.
   - Custom Uniform Market Shock: enter any user-specified market-wide drawdown percentage.
   - Single bordered results panel showing scenario name, affected holdings, estimated loss (₹), portfolio loss (%), and remaining capital (₹).

7. **SciPy-Based Minimum-Variance Optimization:**
   - Numerically minimizes historical portfolio variance ($w^T \Sigma w$) subject to full investment ($\sum w_i = 1$) and configurable maximum stock weight constraint ($0 \le w_i \le w_{\max}$).
   - Mathematical feasibility validation before solving ($n \times w_{\max} \ge 1.0$).
   - Deterministic discrete whole-rupee capital allocation using the Largest Remainder Method (Hamilton quota) ensuring total capital is preserved to the exact rupee.
   - Objective, honest reporting of results without claiming guaranteed outperformance.

8. **Three-Way Before-and-After Portfolio Comparison:**
   - Side-by-side comparison across:
     - **Original User Allocation**
     - **Equal-Weight Baseline**
     - **Minimum-Variance Optimized Allocation**
   - Stock-level allocation comparison table with nominal amounts and percentage weights.
   - Portfolio risk comparison table covering Capital, Volatility, CAGR, Max Drawdown, Beta, HHI, Diversification Ratio, and Risk Classification.
   - Comparative grouped bar charts for visual evaluation.

9. **Multi-Format Report Downloads:**
   - **Multi-Sheet Excel Workbook (`.xlsx`)** generated with `openpyxl`:
     1. *Stock Metrics*
     2. *Original Allocation*
     3. *Equal-Weight Allocation*
     4. *Minimum-Variance Allocation*
     5. *Portfolio Comparison*
     6. *Stress Tests*
     7. *Assumptions*
   - **Consolidated CSV Report (`.csv`)** with formatted sections, metadata, and calculation parameters.

10. **Zerodha Kite-Inspired Design System:**
    - Clean typography using Google Fonts Inter (weights 400 and 500 only; no bold or italic tags).
    - Curated color palette: Accent blue (`#387ED1`), light background (`#ffffff`), text (`#333333`), borders (`#e0e0e0`), positive values (`#00c853`), negative values (`#ff3d00`).
    - Standardized 1px bordered containers enclosing entire panels (no partial heading borders).
    - Native Streamlit components with zero raw HTML leaks.

---

## 5. Technology Stack

- **Language:** Python 3.11+ / 3.13
- **Web Framework:** Streamlit (>= 1.35)
- **Data Engineering:** Pandas (>= 2.2), NumPy (>= 1.26)
- **Quantitative Optimization:** SciPy (>= 1.13)
- **Market Data Provider:** yfinance (>= 0.2.40)
- **Visualization:** Matplotlib (>= 3.8), Seaborn (>= 0.13)
- **Export Engine:** openpyxl (>= 3.1)
- **Testing Framework:** pytest (>= 8.0)

---

## 6. Project Architecture

```text
stock-portfolio-risk/
├── .streamlit/
│   └── config.toml          # Streamlit theme (Kite palette) & server settings
├── portfolio_risk/
│   ├── __init__.py          # Public package API exports
│   ├── charts.py            # Matplotlib & Seaborn chart rendering functions
│   ├── config.py            # Central parameters, thresholds, lookback periods, palette
│   ├── data.py              # yfinance data retrieval, validation, MultiIndex extraction
│   ├── export.py            # Multi-sheet Excel workbook and CSV report generation
│   ├── metrics.py           # Buy-and-hold valuation, CAGR, volatility, beta, HHI, DR
│   ├── optimization.py      # SciPy SLSQP constrained minimum-variance optimizer
│   ├── stress_testing.py    # Predefined, single-stock, and custom stress scenario engine
│   └── validation.py        # Input sanitation, ticker normalization, and regex checks
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures and synthetic deterministic price data
│   ├── test_charts.py       # Visualization rendering and layout tests
│   ├── test_export.py       # Excel and CSV export generation tests
│   ├── test_metrics.py      # Financial formulas, CAGR, whole-rupee allocation tests
│   ├── test_optimization.py # SLSQP optimizer, feasibility, and constraint tests
│   ├── test_portfolio.py    # End-to-end integration and dynamic inputs tests
│   └── test_stress_testing.py # Predefined and custom stress scenario tests
├── .dockerignore
├── .gitignore
├── app.py                   # Main Streamlit dashboard application
├── Dockerfile               # Containerized production build
├── README.md                # Project documentation
├── requirements.txt         # Application dependencies
└── requirements-dev.txt     # Development, testing, and linting dependencies
```

---

## 7. Financial Methodology & Formulas

### 7.1 Buy-and-Hold Portfolio Valuation Model
At inception $t_0$, the user allocates nominal capital $I_i$ to holding $i \in \{1, \dots, N\}$. The number of shares purchased (assuming fractional shares) is:
$$n_i = \frac{I_i}{P_{i,0}}$$

For any subsequent trading day $t$, the holding value is $V_{i,t} = n_i P_{i,t}$, and total portfolio value is:
$$V_t = \sum_{i=1}^N V_{i,t} = \sum_{i=1}^N I_i \left(\frac{P_{i,t}}{P_{i,0}}\right)$$

Daily portfolio simple returns are derived directly from the valuation trajectory:
$$r_{p,t} = \frac{V_t - V_{t-1}}{V_{t-1}}$$

### 7.2 Annualized Volatility
Given daily returns over $T$ trading days, sample standard deviation $\sigma_{\text{daily}}$ (with Bessel's correction, $ddof=1$) is annualized over 252 sessions:
$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

### 7.3 Compounded Annual Growth Rate (CAGR)
Computed from beginning capital $V_{\text{start}}$ and ending capital $V_{\text{end}}$ over elapsed calendar time $T$ in years:
$$\text{CAGR} = \left(\frac{V_{\text{end}}}{V_{\text{start}}}\right)^{1/T} - 1, \quad T = \frac{\text{Days}}{365.25}$$

### 7.4 Maximum Drawdown (MDD)
Measures the maximum observed decline from a historical peak:
$$\text{Peak}_t = \max_{0 \le s \le t} V_s, \quad \text{Drawdown}_t = \frac{V_t - \text{Peak}_t}{\text{Peak}_t}, \quad \text{MDD} = \min_{t} \text{Drawdown}_t$$

### 7.5 Systematic Beta ($\beta$)
Measures sensitivity relative to Nifty 50 daily returns ($r_m$):
$$\beta = \frac{\text{Cov}(r_p, r_m)}{\text{Var}(r_m)}$$

### 7.6 Herfindahl-Hirschman Concentration Index (HHI)
Measures capital concentration from normalized portfolio weights $w_i \in [0, 1]$ where $\sum w_i = 1$:
$$\text{HHI} = \sum_{i=1}^N w_i^2$$
- $\text{HHI} \le 0.20$: Highly Diversified
- $0.20 < \text{HHI} \le 0.35$: Moderately Concentrated
- $\text{HHI} > 0.35$: Highly Concentrated

### 7.7 Diversification Ratio (DR)
Measures risk reduction efficiency from non-perfectly correlated assets:
$$\text{DR} = \frac{\sum_{i=1}^N w_i \sigma_i}{\sigma_p}$$
Where $\sigma_i$ is stock $i$'s annualized volatility and $\sigma_p$ is portfolio annualized volatility.

---

## 8. Portfolio Allocation & Optimization Methods

The dashboard compares three deterministic portfolio strategies:

1. **Original User Allocation:**
   - Uses the nominal investment amounts entered directly by the user:
     $$w_i = \frac{I_i}{\sum_{j=1}^N I_j}$$

2. **Equal-Weight Allocation ($1/N$):**
   - Allocates capital uniformly across all $N$ holdings:
     $$w_i = \frac{1}{N}$$

3. **Minimum-Variance Optimization (SciPy SLSQP):**
   - Solves the constrained quadratic minimization problem using the sample covariance matrix $\Sigma$:
     $$\min_w \quad w^T \Sigma w$$
     $$\text{subject to} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le w_{\max}$$
   - **Feasibility Condition:** Optimization is mathematically feasible if and only if:
     $$N \times w_{\max} \ge 1.0$$
   - **Capital Preservation:** Continuous weights are converted into discrete whole-rupee investments using the **Largest Remainder Method** (Hare-Niemeyer quota):
     $$\sum_{i=1}^N I_i^{\text{recommended}} \equiv \text{Total Capital}$$

---

## 9. Hypothetical Stress Testing

All stress scenarios represent instantaneous capital shocks applied to nominal allocations:

- **Predefined Market Shocks:**
  - Mild Market Pullback: $-5\%$ uniform
  - Broad Market Correction: $-10\%$ uniform
  - Deep Market Correction: $-15\%$ uniform
  - Severe Market Crash: $-20\%$ uniform
- **Selectable Individual-Stock Shock:**
  - Specific stock $k$ drops by user-specified percentage $s_k$:
    $$\text{Loss} = I_k \times |s_k|, \quad \text{Remaining Capital} = V_0 - \text{Loss}$$
- **Custom Uniform Market Shock:**
  - All holdings drop by user-specified percentage $s_{\text{custom}}$:
    $$\text{Loss} = V_0 \times |s_{\text{custom}}|$$

---

## 10. Educational Risk Classification

Portfolios are categorized into educational risk bands based on historical annualized volatility:

| Annualized Volatility | Risk Category | Description |
|---|---|---|
| $\sigma_{\text{annual}} < 15\%$ | **Low Risk** | Lower historical fluctuation; typical of defensive large-cap allocations. |
| $15\% \le \sigma_{\text{annual}} < 25\%$ | **Moderate Risk** | Typical broad-market equity index volatility. |
| $\sigma_{\text{annual}} \ge 25\%$ | **High Risk** | Elevated price fluctuation; high cyclical or thematic exposure. |

*Note: Thresholds are defined in `portfolio_risk/config.py` for educational purposes and do not represent statutory or advisory standards.*

---

## 11. Data Assumptions and Limitations

- **Historical Prices:** Retrieved from Yahoo Finance via `yfinance` using split- and dividend-adjusted closing prices.
- **Benchmark Index:** Nifty 50 (`^NSEI`) serves as the standard proxy for the Indian equity market.
- **Listing Date Discrepancies:** For stocks with listing histories shorter than the requested horizon (e.g., 10 or 15 years), calculations synchronize on the common overlapping trading sessions.
- **Transaction Costs & Friction:** Analysis assumes frictionless execution without brokerage commissions, Securities Transaction Tax (STT), exchange charges, stamp duty, or bid-ask slippage.
- **Optimization Sensitivity:** Historical minimum-variance weights minimize *past* covariance and do not forecast future volatility or prevent future losses.

---

## 12. Automated Test Suite

Run the full automated test suite:
```bash
python -m pytest -v
```

All 46 unit and integration tests execute deterministically against synthetic price fixtures without requiring a live internet connection:

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/prateekjadhav69/Documents/Personal Repos/stock-portfolio-risk

tests/test_charts.py .................................... PASSED
tests/test_export.py .................................... PASSED
tests/test_metrics.py ................................... PASSED
tests/test_optimization.py .............................. PASSED
tests/test_portfolio.py ................................. PASSED
tests/test_stress_testing.py ............................ PASSED

============================== 46 passed in 1.79s ==============================
```
