"""
Task 1 - Portfolio Risk Calculator
=========================
Computes key risk metrics for a wealth management portfolio under
severe and moderate crash scenarios, and provides an ASCII bar-chart
visualisation of asset allocations.
"""

from __future__ import annotations

import math
from typing import Any

def _scenario_metrics(
    total_value: float,
    monthly_expenses: float,
    assets: list[dict[str, Any]],
    crash_severity: float = 1.0,
) -> dict[str, Any]:
    """Return post-crash value, runway, ruin test, risk asset, and concentration warning.

    Args:
        total_value:       Total portfolio value in INR.
        monthly_expenses:  Monthly household expenses in INR.
        assets:            List of asset dicts (name, allocation_pct, expected_crash_pct).
        crash_severity:    Fraction of the expected crash to apply (1.0 = full, 0.5 = moderate).
    """
    post_crash_value: float = 0.0
    max_risk_magnitude: float = -1.0
    largest_risk_asset: str = ""
    concentration_warning: bool = False

    for asset in assets:
        allocation_pct: float = asset["allocation_pct"]
        expected_crash_pct: float = asset["expected_crash_pct"]

        # Current value of this asset slice
        current_value = total_value * (allocation_pct / 100)

        # Adjusted crash for this scenario
        adjusted_crash_pct = expected_crash_pct * crash_severity
        multiplier = 1 + adjusted_crash_pct / 100
        post_crash_value += current_value * multiplier

        # Risk magnitude (always based on the *original* expected crash)
        risk_magnitude = allocation_pct * abs(expected_crash_pct) 
        if risk_magnitude > max_risk_magnitude:
            max_risk_magnitude = risk_magnitude
            largest_risk_asset = asset["name"]

        # Concentration check
        if allocation_pct > 40:
            concentration_warning = True

    # Runway months (guard against division by zero)
    if monthly_expenses == 0:
        runway_months = math.inf
    else:
        runway_months = post_crash_value / monthly_expenses

    ruin_test = "PASS" if runway_months > 12 else "FAIL"

    return {
        "post_crash_value": round(post_crash_value, 2),
        "runway_months": round(runway_months, 2) if math.isfinite(runway_months) else runway_months,
        "ruin_test": ruin_test,
        "largest_risk_asset": largest_risk_asset,
        "concentration_warning": concentration_warning,
    }


def compute_risk_metrics(portfolio: dict[str, Any]) -> dict[str, Any]:
    """Compute risk metrics for both severe and moderate crash scenarios.

    Args:
        portfolio: Dictionary containing total_value_inr, monthly_expenses_inr,
                   and a list of asset dicts under the key 'assets'.

    Returns:
        Dictionary with 'severe' and 'moderate' sub-dictionaries, each holding
        post_crash_value, runway_months, ruin_test, largest_risk_asset, and
        concentration_warning.
    """
    total_value: float = portfolio["total_value_inr"]
    monthly_expenses: float = portfolio["monthly_expenses_inr"]
    assets: list[dict[str, Any]] = portfolio["assets"]

    total_allocation = sum(asset["allocation_pct"] for asset in assets)
    
    if not math.isclose(total_allocation, 100.0, abs_tol=1e-5):
        raise ValueError(f"Total allocation must sum to 100. Current sum is {total_allocation}.")

    severe = _scenario_metrics(total_value, monthly_expenses, assets, crash_severity=1.0)
    moderate = _scenario_metrics(total_value, monthly_expenses, assets, crash_severity=0.5)

    return {
        "severe": severe,
        "moderate": moderate,
    }


# ---------------------------------------------------------------------------
# CLI bar-chart visualisation
# ---------------------------------------------------------------------------


def visualize_allocations(portfolio: dict[str, Any]) -> None:
    """Print an ASCII bar chart of asset allocations to the console."""
    assets = portfolio.get("assets", [])
    if not assets:
        print("No assets to display.")
        return

    # Find the longest asset name for alignment
    max_name_len = max(len(a["name"]) for a in assets)

    print("\nAsset Allocation Chart")
    # Header width adjusted for a max bar of 100 characters
    print("=" * (max_name_len + 115))

    for asset in assets:
        name = asset["name"].ljust(max_name_len)
        raw_pct = asset["allocation_pct"]
        
        # Round to nearest integer and cap at 99 to ensure it is < 100
        display_pct = round(raw_pct)
        if display_pct >= 100:
            display_pct = 99
            
        bar = "#" * display_pct
        print(f"  {name} | {bar} {raw_pct}%")

    print("=" * (max_name_len + 115))
    print()


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the calculator with sample portfolio data and print results."""
    portfolio = {
        "total_value_inr": 10_000_000,
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct": 0},
        ],
    }

    results = compute_risk_metrics(portfolio)

    for label, scenario in results.items():
        print(f"\n--- {label.upper()} Scenario ---")
        for key, value in scenario.items():
            print(f"  {key:25s}: {value}")

    visualize_allocations(portfolio)


if __name__ == "__main__":
    main()
