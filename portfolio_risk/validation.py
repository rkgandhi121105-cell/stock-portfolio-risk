"""Input validation module for the portfolio risk analysis application.

Ensures all user-provided tickers, investment amounts, and date parameters conform
to mathematical, structural, and financial integrity constraints before querying
external market data services.
"""

import math
import re
from typing import List, Sequence, Tuple
from portfolio_risk.config import MAX_HOLDINGS, MIN_HOLDINGS, MIN_INVESTMENT_PER_STOCK


class PortfolioValidationError(ValueError):
    """Custom exception raised when portfolio input validation fails."""
    pass


# Regular expression pattern for valid NSE ticker symbols.
# Typically consists of uppercase alphanumeric characters, optional hyphens or ampersands
# (e.g., 'M&M.NS', 'BAJAJ-AUTO.NS', 'IOC.NS'), followed by the '.NS' suffix for National Stock Exchange of India.
NSE_TICKER_PATTERN = re.compile(r"^[A-Z0-9&_\-]+\.NS$")


def normalize_ticker(ticker: str) -> str:
    """Normalize a raw ticker string by stripping whitespace and converting to uppercase.

    If the user forgot the '.NS' suffix, append it automatically for better usability.

    Args:
        ticker: Raw stock ticker string (e.g. ' reliance.ns ' or 'TCS').

    Returns:
        Cleaned, uppercase ticker string ending in '.NS'.
    """
    if not isinstance(ticker, str):
        return ""
    cleaned = ticker.strip().upper()
    if cleaned and not cleaned.endswith(".NS") and not cleaned.startswith("^"):
        cleaned = f"{cleaned}.NS"
    return cleaned


def validate_ticker_format(ticker: str) -> bool:
    """Validate whether a ticker matches standard NSE format with .NS suffix.

    Note: Format validity does NOT guarantee that the ticker actively exists on
    the exchange or has trading history. It only validates syntactic correctness.

    Args:
        ticker: Normalized ticker string.

    Returns:
        True if the ticker format is valid.
    """
    if not isinstance(ticker, str):
        return False
    return bool(NSE_TICKER_PATTERN.match(ticker))


def validate_portfolio_inputs(
    tickers: Sequence[str],
    investments: Sequence[float],
) -> Tuple[List[str], List[float]]:
    """Validate ticker symbols and investment amounts for the user's portfolio.

    Rules enforced:
    1. Between MIN_HOLDINGS (1) and MAX_HOLDINGS (configurable, e.g. 20) tickers must be supplied.
    2. Ticker count must exactly match investment amount count.
    3. No empty, None, or whitespace-only ticker symbols.
    4. Each ticker must adhere to NSE naming standards (ending in .NS).
    5. No duplicate tickers are permitted (case-insensitive).
    6. Investment amounts must be positive, finite numbers above MIN_INVESTMENT_PER_STOCK.
    7. Total portfolio investment must be positive and non-zero.

    Args:
        tickers: List of ticker strings provided by user.
        investments: List of monetary allocation amounts.

    Returns:
        Tuple of (clean_tickers, clean_investments).

    Raises:
        PortfolioValidationError: If any validation rule is violated.
    """
    if tickers is None or investments is None:
        raise PortfolioValidationError("Tickers and investment amounts cannot be None.")

    if len(tickers) != len(investments):
        raise PortfolioValidationError(
            f"Mismatch between number of tickers ({len(tickers)}) and "
            f"investment amounts ({len(investments)})."
        )

    clean_tickers: List[str] = [normalize_ticker(t) for t in tickers if isinstance(t, str)]

    # Check holding count bounds (supports 1 to MAX_HOLDINGS)
    if len(clean_tickers) < MIN_HOLDINGS or len(clean_tickers) > MAX_HOLDINGS:
        raise PortfolioValidationError(
            f"Portfolio must contain between {MIN_HOLDINGS} and {MAX_HOLDINGS} stocks. "
            f"Received {len(clean_tickers)}."
        )

    # Check ticker formats and emptiness
    for original, normalized in zip(tickers, clean_tickers):
        if not normalized or normalized == ".NS":
            raise PortfolioValidationError("Ticker symbol cannot be empty.")
        if not validate_ticker_format(normalized):
            raise PortfolioValidationError(
                f"Invalid NSE ticker format: '{original}'. "
                "NSE tickers must contain alphanumeric symbols and end with '.NS' (e.g., RELIANCE.NS, IOC.NS)."
            )

    # Check duplicates
    seen = set()
    duplicates = set()
    for t in clean_tickers:
        if t in seen:
            duplicates.add(t)
        seen.add(t)

    if duplicates:
        dup_str = ", ".join(sorted(duplicates))
        raise PortfolioValidationError(
            f"Duplicate tickers are not allowed in the portfolio: {dup_str}. "
            "Combine investments into a single entry."
        )

    # Validate monetary investments
    clean_investments: List[float] = []
    for ticker, amount in zip(clean_tickers, investments):
        if amount is None:
            raise PortfolioValidationError(f"Investment amount for {ticker} cannot be empty.")

        try:
            val = float(amount)
        except (ValueError, TypeError):
            raise PortfolioValidationError(
                f"Investment amount for {ticker} must be a valid numeric value."
            )

        if math.isnan(val) or math.isinf(val):
            raise PortfolioValidationError(
                f"Investment amount for {ticker} must be a finite number."
            )

        if val < MIN_INVESTMENT_PER_STOCK:
            raise PortfolioValidationError(
                f"Investment amount for {ticker} must be at least ₹{MIN_INVESTMENT_PER_STOCK:,.2f}."
            )

        clean_investments.append(val)

    total_capital = sum(clean_investments)
    if total_capital <= 0:
        raise PortfolioValidationError("Total portfolio investment must be strictly positive.")

    return clean_tickers, clean_investments
