"""
Task 3 - AI-Powered Portfolio Explainer
=========================================
Uses Google Gemini to generate a plain-English explanation of a portfolio's
risk profile, written in the tone of a friendly-but-honest financial advisor.
"""

import os
import json
import sys
from typing import Literal
from google import genai
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    sys.exit(
        "ERROR: Please set the GEMINI_API_KEY environment variable.\n"
    )

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

Tone = Literal["beginner", "experienced", "expert"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Engineering
# ─────────────────────────────────────────────────────────────────────────────

TONE_INSTRUCTIONS = {
    "beginner": (
        "The client is new to investing. Use simple language and relatable analogies. "
        "Strictly avoid jargon. If a financial term is unavoidable, define it "
        "immediately in parentheses. Relate concepts to familiar Indian instruments "
        "like Fixed Deposits or physical gold where appropriate."
    ),
    "experienced": (
        "The client has active market experience. Use standard terminology such as "
        "'volatility', 'asset allocation', 'drawdown', and 'concentration risk' "
        "without providing definitions. Keep the assessment direct and focus on "
        "portfolio strategy."
    ),
    "expert": (
        "The client is highly sophisticated. Focus on quantitative mechanics. "
        "Use advanced terminology like tail risk, correlation coefficients, "
        "and risk-adjusted returns. Completely skip basic explanations. Deliver "
        "a strictly analytical and objective assessment."
    ),
}


def get_calculated_risk(portfolio: dict):
    """Attempts to run the external risk calculator. Returns None if it fails."""
    try:
        import Task1_Portfolio_risk_calculator
        return Task1_Portfolio_risk_calculator.scenario_matrix(portfolio)
    
    except ImportError:
        return None
    except AttributeError:
        return None
    except Exception as e:
        print(f"Risk calculation skipped due to error: {e}")
        return None

def build_explainer_prompt(portfolio: dict, tone: str) -> str:
    """Construct the prompt sent to Gemini for the initial explanation."""
    portfolio_str = json.dumps(portfolio, indent=2)
    tone_directive = TONE_INSTRUCTIONS.get(tone, "")
    
    risk_data = get_calculated_risk(portfolio)
    
    risk_context = ""
    if risk_data:
        risk_context = f"\nCALCULATED RISK METRICS:\n{json.dumps(risk_data, indent=2)}\n"

    return f"""You are a friendly but honest financial advisor speaking with
an Indian client about their portfolio. You advise them for what is best in the long term and you do so truthfully so if they are making a mistake you tell them. 
You cater your response in the language that the client understands so respond according to the tone guidance given below.

TONE GUIDANCE
{tone_directive}

YOUR TASK
Analyse the portfolio and respond with ONLY a valid JSON object. DO NOT USE markdown
fences, no commentary before or after just with exactly these six keys:

{{
  "reasoning": "Briefly calculate portfolio values, check concentration risks, and think step-by-step about the advice before writing the fields below.",
  "summary": "A 3-4 sentence plain-English summary of the portfolio's overall risk level. Mention concentration, crash exposure, and runway if relevant.",
  "doing_well": "One specific thing the investor is doing well. Name the asset or behaviour.",
  "should_change": "One specific thing they should consider changing AND the reasoning behind it. Name the asset, explain the risk, suggest a direction (not a precise number).",
  "verdict": "A one-line verdict classifying the portfolio as 'Aggressive', 'Balanced', or 'Conservative'.",
  "recommended_allocation": "A JSON object mapping each asset name to its recommended allocation percentage. The percentages MUST sum to exactly 100. Keep the same asset names from the input portfolio. Example: {{\\"BTC\\": 5, \\"NIFTY50\\": 55, \\"GOLD\\": 20, \\"CASH\\": 20}}. Base this on your analysis — reduce overexposed assets, increase underweighted safe havens, and aim for a more balanced risk profile."
}}

Rules:
- Base every claim on the actual numbers in the portfolio above.
- If CALCULATED RISK METRICS are provided below, you must incorporate them into your reasoning and assessment.
- Do not invent assets or facts.
- The 'verdict' field MUST be a single line starting with 'Aggressive', 'Balanced', or 'Conservative' followed by a brief explanation of why.
- Output raw JSON only.

EXAMPLES OF GOOD RESPONSES
Use these examples to understand the expected structure and tone. Do not copy them line for line. Your summary and advice must be unique and specifically adapted to the client's actual numbers.

Example 1 (All Crypto - Very Risky):
Input Portfolio: {{"total_value_inr": 10000000, "monthly_expenses_inr": 80000, "assets": [{{"name": "BTC", "allocation_pct": 100, "expected_crash_pct": -80}}]}}
Output:
{{
  "reasoning": "100% allocation in BTC. An 80% crash wipes out 80L, leaving 20L. With 80k expenses, the runway shrinks drastically. This is highly concentrated and reckless.",
  "summary": "Your entire portfolio is tied to a single highly volatile asset. While the potential upside is high, a severe market crash could wipe out the majority of your wealth overnight. This severely threatens your long term financial security.",
  "doing_well": "You have managed to accumulate a substantial total corpus.",
  "should_change": "You must urgently diversify. Sell a large portion of your Bitcoin and move it into more stable assets like NIFTY50 and Cash to protect your capital from extreme crypto volatility.",
  "verdict": "Aggressive. A 100% allocation to a highly volatile asset class exposes you to unacceptable risk of ruin.",
  "recommended_allocation": {{"BTC": 10, "NIFTY50": 50, "GOLD": 20, "CASH": 20}}
}}

Example 2 (All Cash - No Growth):
Input Portfolio: {{"total_value_inr": 10000000, "monthly_expenses_inr": 80000, "assets": [{{"name": "CASH", "allocation_pct": 100, "expected_crash_pct": 0}}]}}
Output:
{{
  "reasoning": "100% in cash means 0% crash risk, but guaranteed loss of purchasing power due to inflation. 1Cr / 80k gives roughly 125 months of runway but no growth.",
  "summary": "Your portfolio is completely insulated from market crashes, but it is heavily exposed to inflation risk. By holding only cash, your wealth is silently losing its purchasing power every year. You are trading short term safety for long term stagnation.",
  "doing_well": "You have zero market risk and an excellent emergency buffer.",
  "should_change": "You need to start allocating capital to growth assets like NIFTY50. Cash generates no real returns, so investing in equities will help your corpus grow faster than inflation.",
  "verdict": "Conservative. Holding 100% cash guarantees no nominal loss but guarantees a loss of real purchasing power over time.",
  "recommended_allocation": {{"CASH": 20, "NIFTY50": 50, "GOLD": 20, "BTC": 10}}
}}

Example 3 (Balanced - Optimal):
Input Portfolio: {{"total_value_inr": 10000000, "monthly_expenses_inr": 80000, "assets": [{{"name": "BTC", "allocation_pct": 5, "expected_crash_pct": -80}}, {{"name": "NIFTY50", "allocation_pct": 55, "expected_crash_pct": -40}}, {{"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15}}, {{"name": "CASH", "allocation_pct": 20, "expected_crash_pct": 0}}]}}
Output:
{{
  "reasoning": "55% equity provides growth. 20% gold hedges inflation. 20% cash provides a 25-month runway (20L / 80k). 5% crypto is a controlled speculative bet. Excellent balance.",
  "summary": "You have built a highly resilient and well diversified portfolio. It balances strong growth potential from equities with solid downside protection from gold and cash. This setup ensures you can weather market downturns comfortably without sacrificing long term returns.",
  "doing_well": "Your asset allocation is excellent, particularly the 20% cash buffer which covers over two years of living expenses.",
  "should_change": "Consider setting strict rebalancing rules. As NIFTY50 or BTC grows, trim the profits to maintain these target allocation percentages so risk does not slowly creep up.",
  "verdict": "Balanced. The portfolio smartly blends high growth assets with sufficient safe haven buffers to mitigate severe drawdowns.",
  "recommended_allocation": {{"BTC": 5, "NIFTY50": 55, "GOLD": 20, "CASH": 20}}
}}

THE CLIENT'S PORTFOLIO
{portfolio_str}
{risk_context}
Notes on the data:
- 'monthly_expenses_inr' is what the client spends each month.
- 'total_value_inr' is the total value of the portfolio of client.
- 'allocation_pct' is the share of the portfolio held in that asset.
- 'expected_crash_pct' is how much that asset is expected to lose in a
  severe but plausible crash (negative number = loss).
"""


def build_critique_prompt(portfolio: dict, explanation: dict, expected_tone: str, tone_directive: str) -> str:
    """Prompt for the second LLM call that critiques the first explanation."""
    portfolio_str = json.dumps(portfolio, indent=2)
    explanation_str = json.dumps(explanation, indent=2)

    return f"""You are a senior portfolio reviewer at a wealth management firm.
A junior advisor just produced the explanation below for the client portfolio
shown. Your job is to critique that explanation honestly.

THE ASSIGNMENT
The junior advisor was instructed to write the explanation using the following tone:
Tone Name: {expected_tone}
Tone Guidelines: {tone_directive}

CLIENT PORTFOLIO
{portfolio_str}

JUNIOR ADVISOR'S EXPLANATION
{explanation_str}

YOUR TASK
Review the explanation against the portfolio data and the assigned tone. Respond with ONLY a valid JSON object. Do not use markdown fences or add commentary. Use exactly these seven keys:

{{
  "reasoning": "First, verify the math and risk claims against the portfolio data. Then, analyze the language used against the expected tone guidelines. Finally, evaluate the recommended_allocation for soundness.",
  "factual_accuracy": "Are the numbers and risk claims correct? Point out any miscalculations, hallucinations, or misinterpretations of the data.",
  "missing_points": "What critical risks or strengths did the junior advisor fail to mention based on the portfolio numbers?",
  "tone_assessment": "Did the advisor successfully adopt the requested '{expected_tone}' tone? Quote specific words or phrases that hit or missed the mark. Keep this concise so in 1-2 sentences.",
  "redistribution_review": "Evaluate the recommended_allocation. Does it address the identified risks? Does it improve diversification? Is the suggested shift too aggressive or too conservative? Would the new allocation survive the expected crash scenario better than the original?",
  "redistribution_verdict": "One of: 'Good' (allocation is sound), 'Needs Adjustment' (directionally right but amounts are off), or 'Bad' (allocation makes things worse or ignores key risks). Follow with a brief reason.",
  "one_line_verdict": "A single sentence stating if the explanation should be 'Approved', 'Edited', or 'Rewritten', and why."
}}

Rules:
- Be highly critical of factual errors or fabricated data.
- Ensure the critique is based strictly on the provided portfolio numbers.
- When reviewing the redistribution, calculate the expected crash loss under both the original and recommended allocations and compare them.
- Output raw JSON only.
"""


def validate_redistribution(portfolio: dict, recommended_allocation: dict) -> dict:
    """
    Programmatically validate the recommended redistribution against the
    original portfolio. Returns a dict of validation results with pass/fail
    checks and computed risk metrics.
    """
    results = {
        "checks": [],
        "original_crash_loss_pct": 0.0,
        "new_crash_loss_pct": 0.0,
        "passed": True,
    }

    # --- Check 1: Allocations sum to 100% ---
    total_alloc = sum(recommended_allocation.values())
    sum_ok = abs(total_alloc - 100.0) < 0.01
    results["checks"].append({
        "name": "Allocations sum to 100%",
        "passed": sum_ok,
        "detail": f"Sum = {total_alloc:.2f}%"
    })
    if not sum_ok:
        results["passed"] = False

    # --- Check 2: No negative allocations ---
    negatives = {k: v for k, v in recommended_allocation.items() if v < 0}
    no_neg = len(negatives) == 0
    results["checks"].append({
        "name": "No negative allocations",
        "passed": no_neg,
        "detail": f"Negative assets: {negatives}" if negatives else "All allocations >= 0"
    })
    if not no_neg:
        results["passed"] = False

    # --- Check 3: All original assets accounted for ---
    original_names = {a["name"].upper() for a in portfolio.get("assets", [])}
    recommended_names = {k.upper() for k in recommended_allocation}
    missing = original_names - recommended_names
    all_present = len(missing) == 0
    results["checks"].append({
        "name": "All original assets present",
        "passed": all_present,
        "detail": f"Missing: {missing}" if missing else "All assets accounted for"
    })
    if not all_present:
        results["passed"] = False

    # --- Check 4: Crash-scenario risk comparison ---
    # Build a lookup for crash percentages
    crash_lookup = {}
    for a in portfolio.get("assets", []):
        crash_lookup[a["name"].upper()] = a.get("expected_crash_pct", 0)

    # Original weighted crash loss
    original_loss = 0.0
    for a in portfolio.get("assets", []):
        original_loss += a["allocation_pct"] * abs(a.get("expected_crash_pct", 0)) / 100.0
    results["original_crash_loss_pct"] = round(original_loss, 2)

    # New weighted crash loss
    new_loss = 0.0
    for asset_name, alloc_pct in recommended_allocation.items():
        crash = abs(crash_lookup.get(asset_name.upper(), 0))
        new_loss += alloc_pct * crash / 100.0
    results["new_crash_loss_pct"] = round(new_loss, 2)

    risk_reduced = new_loss <= original_loss
    results["checks"].append({
        "name": "Crash-scenario risk reduced or maintained",
        "passed": risk_reduced,
        "detail": f"Original crash loss: {original_loss:.2f}%, New crash loss: {new_loss:.2f}%"
    })
    # This is a soft check — failing it doesn't invalidate the redistribution
    # but it's flagged as a warning

    # --- Check 5: No single asset > 70% (concentration risk) ---
    max_alloc = max(recommended_allocation.values()) if recommended_allocation else 0
    max_asset = max(recommended_allocation, key=recommended_allocation.get) if recommended_allocation else "N/A"
    concentration_ok = max_alloc <= 70
    results["checks"].append({
        "name": "No extreme concentration (>70% in one asset)",
        "passed": concentration_ok,
        "detail": f"Highest: {max_asset} at {max_alloc:.1f}%"
    })
    if not concentration_ok:
        results["passed"] = False

    # --- Check 6: Cash allocation covers at least 6 months of expenses ---
    total_value = portfolio.get("total_value_inr", 0)
    monthly_expenses = portfolio.get("monthly_expenses_inr", 1)
    cash_pct = 0
    for k, v in recommended_allocation.items():
        if k.upper() == "CASH":
            cash_pct = v
            break
    cash_value = total_value * cash_pct / 100.0
    months_runway = cash_value / monthly_expenses if monthly_expenses > 0 else 0
    runway_ok = months_runway >= 6
    results["checks"].append({
        "name": "Cash covers >= 6 months expenses",
        "passed": runway_ok,
        "detail": f"Cash = ₹{cash_value:,.0f}, Runway = {months_runway:.1f} months"
    })
    # Soft check — warning only

    return results


# ─────────────────────────────────────────────────────────────────────────────
# API Calls
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str | None:
    """Send a prompt to Gemini and return the raw text response, handling server errors."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
            },
        )
        return response.text
    except Exception as e:
        print("\n[ERROR] The Gemini API server is currently busy or unreachable. Please try again later.")
        print(f"Technical details: {e}\n")
        return None


def parse_json_safely(raw: str) -> dict | None:
    """Parse JSON, stripping markdown fences if the model added them."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"\n[WARN] Could not parse JSON: {e}")
        return None

def explain_portfolio(portfolio: dict, tone: Tone = "beginner") -> dict | None:
    """Generate a plain English explanation of the portfolio's risk."""
    prompt = build_explainer_prompt(portfolio, tone)
    raw = call_gemini(prompt)

    # Check if the API call failed before trying to parse
    if raw is None:
        return None

    parsed = parse_json_safely(raw)
    if parsed is None:
        return None

    EXPLANATION_KEYS = {"reasoning", "summary", "doing_well", "should_change", "verdict", "recommended_allocation"}
    if not isinstance(parsed, dict):
        print(f"\n[WARN] Expected a JSON object from explainer, got {type(parsed).__name__}.")
        return None
    missing = EXPLANATION_KEYS - parsed.keys()
    if missing:
        print(f"\n[WARN] Explainer response is missing keys: {missing}")
        return None
    STRING_KEYS = {"reasoning", "summary", "doing_well", "should_change", "verdict"}
    if not all(isinstance(parsed[k], str) for k in STRING_KEYS):
        print("\n[WARN] Explainer response contains non-string values for expected keys.")
        return None

    # Validate recommended_allocation field
    rec_alloc = parsed.get("recommended_allocation")
    if not isinstance(rec_alloc, (dict, str)):
        print("\n[WARN] recommended_allocation must be a JSON object or string.")
        return None
    # If the LLM returned it as a JSON string, parse it
    if isinstance(rec_alloc, str):
        try:
            rec_alloc = json.loads(rec_alloc)
            parsed["recommended_allocation"] = rec_alloc
        except json.JSONDecodeError:
            print("\n[WARN] Could not parse recommended_allocation string as JSON.")
            return None

    print("=" * 70)
    print("STRUCTURED EXPLANATION")
    print("=" * 70)
    print(f"Verdict: {parsed.get('verdict', 'N/A')}\n")
    print(f"Summary: {parsed.get('summary', 'N/A')}\n")
    print(f"Doing Well: {parsed.get('doing_well', 'N/A')}\n")
    print(f"Should Change: {parsed.get('should_change', 'N/A')}\n")

    # Display the recommended redistribution
    rec_alloc = parsed.get("recommended_allocation", {})
    if rec_alloc:
        print("-" * 40)
        print("RECOMMENDED ASSET REDISTRIBUTION")
        print("-" * 40)
        for asset_name, pct in rec_alloc.items():
            # Find original allocation for comparison
            original_pct = None
            for a in portfolio.get("assets", []):
                if a["name"].upper() == asset_name.upper():
                    original_pct = a["allocation_pct"]
                    break
            if original_pct is not None:
                delta = pct - original_pct
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                print(f"  {asset_name:>10s}: {original_pct:5.1f}%  →  {pct:5.1f}%  ({arrow} {abs(delta):.1f}%)")
            else:
                print(f"  {asset_name:>10s}: (new)  →  {pct:5.1f}%")
        print()

    return parsed


def critique_explanation(portfolio: dict, explanation: dict, expected_tone: str, tone_directive: str) -> dict | None:
    """Second LLM call - reviews the first explanation for accuracy.
    Also runs programmatic validation on the recommended redistribution."""
    prompt = build_critique_prompt(portfolio, explanation, expected_tone, tone_directive)
    raw = call_gemini(prompt)

    # Check if the API call failed before trying to parse
    if raw is None:
        return None

    parsed = parse_json_safely(raw)
    if parsed is None:
        return None

    CRITIQUE_KEYS = {"reasoning", "factual_accuracy", "missing_points", "tone_assessment",
                     "redistribution_review", "redistribution_verdict", "one_line_verdict"}
    if not isinstance(parsed, dict):
        print(f"\n[WARN] Expected a JSON object from critique, got {type(parsed).__name__}.")
        return None
    missing = CRITIQUE_KEYS - parsed.keys()
    if missing:
        print(f"\n[WARN] Critique response is missing keys: {missing}")
        return None
    if not all(isinstance(parsed[k], str) for k in CRITIQUE_KEYS):
        print("\n[WARN] Critique response contains non-string values for expected keys.")
        return None

    print("=" * 70)
    print("STRUCTURED CRITIQUE")
    print("=" * 70)
    print(f"Factual Accuracy: {parsed.get('factual_accuracy', 'N/A')}\n")
    print(f"Missing Points: {parsed.get('missing_points', 'N/A')}\n")
    print(f"Tone Assessment: {parsed.get('tone_assessment', 'N/A')}\n")
    print(f"Redistribution Review: {parsed.get('redistribution_review', 'N/A')}\n")
    print(f"Redistribution Verdict: {parsed.get('redistribution_verdict', 'N/A')}\n")
    print(f"Verdict: {parsed.get('one_line_verdict', 'N/A')}\n")

    # ── Programmatic redistribution validation ──
    rec_alloc = explanation.get("recommended_allocation", {})
    if isinstance(rec_alloc, str):
        try:
            rec_alloc = json.loads(rec_alloc)
        except json.JSONDecodeError:
            rec_alloc = {}

    if rec_alloc:
        print("=" * 70)
        print("PROGRAMMATIC REDISTRIBUTION VALIDATION")
        print("=" * 70)
        validation = validate_redistribution(portfolio, rec_alloc)

        for check in validation["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {status}  {check['name']}")
            print(f"         {check['detail']}")

        print()
        print(f"  Original crash-scenario loss: {validation['original_crash_loss_pct']:.2f}%")
        print(f"  New crash-scenario loss:      {validation['new_crash_loss_pct']:.2f}%")
        improvement = validation['original_crash_loss_pct'] - validation['new_crash_loss_pct']
        if improvement > 0:
            print(f"  Risk improvement:             ↓ {improvement:.2f}% (better)")
        elif improvement < 0:
            print(f"  Risk change:                  ↑ {abs(improvement):.2f}% (worse)")
        else:
            print(f"  Risk change:                  → 0% (unchanged)")
        print()

        overall = "REDISTRIBUTION VALIDATED" if validation["passed"] else " REDISTRIBUTION HAS ISSUES (see failed checks above)"
        print(f"  {overall}\n")
    else:
        print("\n[INFO] No recommended_allocation found in explanation — skipping validation.\n")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Demo & Interactive Setup
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PORTFOLIO = {
    "total_value_inr": 10_000_000,
    "monthly_expenses_inr": 80_000,
    "assets": [
        {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
        {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
        {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
        {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct":   0},
    ],
}

def get_user_portfolio() -> tuple[dict, str]:
    """Interactively asks the user for portfolio details and explanation tone."""
    print("\n" + "=" * 70)
    print("PORTFOLIO SETUP")
    print("=" * 70)
    
    choice = input("Would you like to enter a custom portfolio? (y/N): ").strip().lower()
    if choice != 'y':
        print("Using the default SAMPLE_PORTFOLIO.\n")
        active_portfolio = SAMPLE_PORTFOLIO
    else:
        print("\n--- Entering Custom Portfolio ---")
        try:
            total_value = int(input("Total Portfolio Value (INR) [e.g., 10000000]: "))
            monthly_expenses = int(input("Monthly Expenses (INR) [e.g., 80000]: "))
            
            assets = []
            print("\nEnter assets one by one. Leave 'Asset Name' blank and press Enter when finished.")
            
            while True:
                name = input("\nAsset Name (e.g., BTC, EQUITY, CASH): ").strip()
                if not name:
                    if not assets:
                        print("You must enter at least one asset. Please try again.")
                        continue
                    break
                
                alloc_pct = float(input(f"Allocation percentage for {name} (0-100): "))
                crash_pct = float(input(f"Expected crash percentage for {name} (e.g., -40 for a 40% drop): "))
                
                assets.append({
                    "name": name.upper(),
                    "allocation_pct": alloc_pct,
                    "expected_crash_pct": crash_pct
                })
                
            active_portfolio = {
                "total_value_inr": total_value,
                "monthly_expenses_inr": monthly_expenses,
                "assets": assets
            }
            
            total_alloc = sum(a["allocation_pct"] for a in assets)
            if total_alloc != 100:
                print(f"\n[WARN] Total allocation equals {total_alloc}%, not 100%. The AI will analyze it as-is.")
                
        except ValueError:
            print("\n[ERROR] Invalid number format entered. Falling back to the SAMPLE_PORTFOLIO.")
            active_portfolio = SAMPLE_PORTFOLIO

    print("\n" + "=" * 70)
    print("EXPLANATION SETTINGS")
    print("=" * 70)
    print("How would you like your portfolio explained?")
    print("1. Beginner (Simple language, analogies, no jargon)")
    print("2. Experienced (Standard terminology, direct assessment)")
    print("3. Expert (Quantitative mechanics, advanced terminology)")
    
    tone_choice = input("\nSelect a tone (1/2/3) [Default: 1]: ").strip()
    
    if tone_choice == '2':
        chosen_tone = "experienced"
    elif tone_choice == '3':
        chosen_tone = "expert"
    else:
        chosen_tone = "beginner"
        
    return active_portfolio, chosen_tone


def main():
    # 1. Ask the user for their portfolio and desired tone
    active_portfolio, chosen_tone = get_user_portfolio()
    tone_directive = TONE_INSTRUCTIONS.get(chosen_tone, "")

    print("\n" + "#" * 70)
    print("# STEP 1 — Generate the portfolio explanation")
    print("#" * 70 + "\n")
    
    # 2. Pass the interactive portfolio to the explainer
    explanation = explain_portfolio(active_portfolio, tone=chosen_tone)

    if explanation is None:
        print("Explanation failed to parse. Skipping critique step.")
        return

    print("\n" + "#" * 70)
    print("# STEP 2 — Critique the explanation (second LLM call)")
    print("#" * 70 + "\n")
    
    # 3. Pass the interactive portfolio to the critique logic
    critique_explanation(active_portfolio, explanation, chosen_tone, tone_directive)


if __name__ == "__main__":
    main()
