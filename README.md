# Timecell Intern Technical Test 2026

This repository contains my Python solutions for the Timecell.ai Engineering Intern technical assessment. The project focuses on portfolio risk, live market data, and LLM-powered financial explanations for high-net-worth Indian family portfolios.

## Project Structure

| File | Purpose |
| --- | --- |
| `Task1_Portfolio_risk_calculator.py` | Task 01: deterministic portfolio crash-risk calculator with severe and moderate scenarios plus a CLI allocation chart. |
| `Task2_Live_market_data_fetch.py` | Task 02: live market data fetcher for BTC, NIFTY50, and Gold, with formatted terminal output and graceful API failure handling. |
| `Task3_Portfolio_explainer.py` | Task 03 and Task 04: Gemini-powered portfolio explainer with configurable tone, structured JSON output, critique pass, and LLM-based redistribution validation. |

## Setup

Install dependencies:

```bash
pip install requests yfinance tabulate pytz google-genai
```

Environment variables:

```bash
# Required for Task 03 LLM and redistribution features
GEMINI_API_KEY=your_gemini_api_key

# Required only for the Gold line in Task 02
GOLDAPI_API_KEY=your_goldapi_key
```

Task 02 still continues if one data source fails. For example, if the Gold API key is missing, BTC and NIFTY50 are still fetched and printed when their APIs are available.

## Task 01 - Portfolio Risk Calculator

File: `Task1_Portfolio_risk_calculator.py`

The calculator accepts a portfolio dictionary with total value, monthly expenses, and a list of assets. Each asset has:

- `name`
- `allocation_pct`
- `expected_crash_pct`

The main function is:

```python
compute_risk_metrics(portfolio)
```

It computes the required metrics from the PDF:

- `post_crash_value`: total portfolio value after the crash.
- `runway_months`: post-crash value divided by monthly expenses.
- `ruin_test`: `PASS` if runway is more than 12 months, otherwise `FAIL`.
- `largest_risk_asset`: asset with the largest `allocation_pct * abs(expected_crash_pct)`.
- `concentration_warning`: `True` if any single asset is above 40 percent.

I also implemented both optional bonuses:

- A moderate crash scenario where each asset loses only 50 percent of its expected crash magnitude.
- A simple CLI allocation bar chart without external plotting libraries.

Exceptions and edge cases are handled properly. The function validates that allocations sum to 100 percent before calculating risk, raises a clear `ValueError` when the input allocation is invalid, and handles zero monthly expenses by returning infinite runway rather than crashing on division by zero.

Example result for the sample portfolio:

```text
SEVERE:
post_crash_value      = 5700000.0
runway_months         = 71.25
ruin_test             = PASS
largest_risk_asset    = BTC
concentration_warning = False

MODERATE:
post_crash_value      = 7850000.0
runway_months         = 98.12
ruin_test             = PASS
largest_risk_asset    = BTC
concentration_warning = False
```

Run it with:

```bash
python Task1_Portfolio_risk_calculator.py
```

## Task 02 - Live Market Data Fetch

File: `Task2_Live_market_data_fetch.py`

This script fetches current prices for three assets:

- BTC through CoinGecko.
- NIFTY50 through Yahoo Finance using `yfinance`.
- Gold through GoldAPI, converted to INR per gram.

The output is a clean terminal table containing:

- asset name
- price
- currency
- timestamp of fetch in IST

Implementation details:

- Uses `Decimal` for safer money formatting.
- Formats INR with Indian comma grouping.
- Wraps each API call in its own `try/except` block.
- Logs the failed asset and continues with the remaining assets.
- Uses `.env` fallback support for `GOLDAPI_API_KEY`.
- Exceptions are handled properly for all external calls, so one failed API does not crash the whole script.

Run it with:

```bash
python Task2_Live_market_data_fetch.py
```

## Task 03 - AI-Powered Portfolio Explainer

File: `Task3_Portfolio_explainer.py`

This task uses Google Gemini through the `google-genai` SDK. I used Gemini because it was accessible, fast for this JSON-heavy workflow, and supports response configuration for JSON output.

The script accepts either:

- the default sample portfolio, or
- a custom portfolio entered interactively in the terminal.

It then asks Gemini to produce a structured financial explanation with these fields:

- `reasoning`
- `summary`
- `doing_well`
- `should_change`
- `verdict`
- `recommended_allocation`

The user can choose one of three explanation tones:

- `beginner`: simple language, relatable examples, no jargon unless defined.
- `experienced`: standard market terminology and direct portfolio strategy.
- `expert`: quantitative, analytical, and advanced terminology.

### Prompt Approach

For prompt structuring, I referred to official Gemini/Google documentation such as the Vertex AI prompt design strategies guide (`https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies`) and supporting learning material from GeeksforGeeks.

I built the prompt in stages:

1. I first started with normal prompting without a specific output format. It gave decent plain-English results, but the response shape changed between runs.
2. I then included explicit reasoning steps so the model would calculate portfolio value, concentration risk, crash exposure, and runway before writing advice.
3. After that, I added a few examples for few-shot prompting. This made the answers more consistent because the model could see what a good risky, conservative, and balanced portfolio explanation looked like.
4. Finally, I structured the output as a JSON response and wrote separate parsing and validation functions to standardise the results.

What worked best:

- Giving the model an exact JSON schema.
- Telling it to return raw JSON only, with no markdown fences.
- Including clear tone-specific instructions.
- Adding examples of good responses for risky, overly conservative, and balanced portfolios.
- Passing the actual portfolio numbers directly into the prompt.
- Asking for a recommended allocation in the same response so the advice becomes actionable.

Exceptions are handled properly throughout the LLM workflow. API failures are caught, JSON parsing failures are handled safely, missing keys are detected, and invalid `recommended_allocation` values are rejected instead of being used blindly.

### Redistribution Logic Included Here

The Task 04 redistribution idea is implemented inside `Task3_Portfolio_explainer.py`.

The first LLM call returns `recommended_allocation`. The second LLM call then reviews that redistribution through:

- `redistribution_review`
- `redistribution_verdict`

The script also validates the recommendation programmatically using `validate_redistribution()`. This validator checks:

- allocations sum to 100 percent
- no negative allocations
- all original assets are accounted for
- crash-scenario risk before and after redistribution
- no asset remains extremely concentrated above 70 percent
- cash allocation covers at least 6 months of expenses

This matters because LLM advice should not be trusted blindly. The model can suggest a redistribution, but deterministic checks should verify whether it actually reduces risk.

Run it with:

```bash
python Task3_Portfolio_explainer.py
```

## Task 04 - Open Problem

Primary idea: AI-guided portfolio redistribution based on the LLM's response, with deterministic validation.

The open problem I chose was: Timecell can identify portfolio risk, but a client or wealth manager naturally asks, "What should I change now?" I implemented this in `Task3_Portfolio_explainer.py` by making the LLM return a `recommended_allocation` alongside the explanation.

Task 04 workflow inside `Task3_Portfolio_explainer.py`:

1. The LLM explains the portfolio risk and recommends how the assets should be redistributed.
2. The redistribution is reviewed by a second LLM critique call.
3. The recommendation is validated with deterministic Python checks.
4. The script compares the original crash-scenario loss with the new suggested allocation.
5. The script flags invalid or unsafe recommendations instead of accepting them blindly.

Why this is worth building:

- It moves the product from descriptive analytics to decision support.
- It keeps the human-facing advice explainable.
- It pairs LLM judgment with deterministic risk checks.
- It gives a wealth manager a clear before/after view instead of a vague recommendation.
- Exceptions are handled properly: failed API calls, invalid JSON, missing response keys, invalid allocation formats, and unsafe allocation checks are all handled explicitly.

## AI Usage Disclosure

The assessment explicitly allows and encourages AI tools. I used AI assistance for:

- brainstorming the solution structure
- improving prompt design
- thinking through edge cases
- reviewing readability and README clarity

I reviewed and adapted the code so I can explain the implementation, the portfolio math, the API choices, and the prompt decisions.

For runtime LLM functionality, the code uses Google Gemini through the `google-genai` Python SDK. The model configured in the code is `gemini-2.5-flash`.

## Hardest Part

The hardest part was making the LLM output useful but still reliable. A plain advisor prompt gives nice prose, but it is hard to validate. The final approach evolved from normal prompting, to reasoning steps, to few-shot examples, and finally to structured JSON with separate parsing functions, a critique pass, and deterministic redistribution checks so the AI output is both readable and auditable.

## Submission Checklist

- Public GitHub repo named in the required format: `timecell-intern-<your-name>`.
- README explaining the approach for each task.
- AI usage disclosed.
- LLM API choice and prompt approach documented.
- 3-5 minute Loom or screen recording walkthrough.
- Email submission includes repo link, video link, and one paragraph on the hardest part.
