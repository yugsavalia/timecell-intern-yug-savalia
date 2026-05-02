"""
Task 2 - Live Market Data Fetch
=================================
Fetches current prices for BTC, NIFTY 50, and Gold and displays them in INR.
"""

import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytz
import requests
import yfinance as yf
from tabulate import tabulate

TROY_OUNCE_IN_GRAMS = Decimal("31.1034768")


def get_goldapi_key() -> str | None:
    """Read GoldAPI key from environment or .env file."""
    if value := os.getenv("GOLDAPI_API_KEY", "").strip():
        return value

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == "GOLDAPI_API_KEY":
            return val.strip().strip("\"'")
    return None


def fetch_prices() -> list[tuple[str, Decimal | None]]:
    """Fetch BTC (USD), NIFTY50 (INR), and Gold (INR/gram) prices."""
    results = []

    # BTC via CoinGecko
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "inr"}, timeout=10,
        )
        resp.raise_for_status()
        results.append(("BTC", Decimal(str(resp.json()["bitcoin"]["inr"]))))
    except Exception as e:
        print(f"  [ERROR] BTC: {e}")
        results.append(("BTC", None))

    # NIFTY 50 via yfinance
    try:
        price = yf.Ticker("^NSEI").fast_info["lastPrice"]
        results.append(("NIFTY50", Decimal(str(price))))
    except Exception as e:
        print(f"  [ERROR] NIFTY50: {e}")
        results.append(("NIFTY50", None))

    # Gold via GoldAPI.io
    try:
        key = get_goldapi_key()
        if not key:
            raise ValueError("Missing GoldAPI key — set GOLDAPI_API_KEY in env or .env")
        resp = requests.get(
            "https://www.goldapi.io/api/XAU/INR",
            headers={"x-access-token": key, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        gram_price = next(
            (Decimal(str(data[k])) for k in ("price_gram_24k", "price_gram_24K", "price_gram") if k in data),
            Decimal(str(data["price"])) / TROY_OUNCE_IN_GRAMS,
        )
        results.append(("GOLD", gram_price))
    except Exception as e:
        print(f"  [ERROR] GOLD: {e}")
        results.append(("GOLD", None))

    return results


def format_inr(price: Decimal) -> str:
    """Format a number with Indian-style comma grouping (e.g. 2,84,345.60)."""
    value = price.quantize(Decimal("0.01"))
    integer_part, decimal_part = f"{value:.2f}".split(".")
    sign = "-" if value < 0 else ""
    s = str(abs(int(value)))

    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        remaining = s[:-3]
        groups = []
        while remaining:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        groups.reverse()
        formatted = ",".join(groups) + "," + last_three

    return f"{sign}{formatted}.{decimal_part}"


def main():
    prices = fetch_prices()
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")

    rows = [
        [name, format_inr(price), "INR" if name != "GOLD" else "INR/gram"]
        for name, price in prices if price is not None
    ]

    print(f"\nAsset Prices — fetched at {ist_now}\n")
    if rows:
        print(tabulate(rows, headers=["Asset", "Price", "Currency"], tablefmt="pretty"))
    else:
        print("  No data could be fetched. Check your network connection.")


if __name__ == "__main__":
    main()