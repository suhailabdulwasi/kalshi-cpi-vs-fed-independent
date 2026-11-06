# No-trade benchmark and out-of-sample guardrail

**Scope:** an additional evaluation of the Phase 6 CPI trading experiment, not a
retroactive change to the reported baseline.

## Question

The existing `choose_theta` routine selects the training threshold with the
highest aggregate net P&L (subject to a minimum trade count) even when every
qualifying strategy loses money. The reported held-out gain therefore describes
a **forced-to-trade** rule, not what a decision-maker with the option to hold
cash would have implemented.

## Alternative rule

1. Use the same historical train/test split, quotes, fills, fees, candidate
   thresholds and 20-trade minimum as the existing Phase 6 analysis.
2. Rank eligible thresholds by *training-only* net P&L, with the lower theta
   breaking ties.
3. If none qualifies or the best training P&L is nonpositive, **do not trade**.
4. Otherwise freeze the winning theta and evaluate it once on the held-out
   period. Never use held-out data to select whether to trade.

The rule is implemented in
[`src/analysis/no_trade_validation.py`](../src/analysis/no_trade_validation.py),
with regression tests in
[`tests/test_no_trade_validation.py`](../tests/test_no_trade_validation.py).

## Interpretation

The published README states that **0 of 31 thresholds were profitable in
training**. Under this explicitly specified alternative, the decision would be
to hold cash and make zero trades, rather than carrying the least-bad threshold
to the test period. The reported +4.9 cents per contract is still the
*historical held-out performance of the forced-to-trade baseline*, not a
realized result under this new no-trade rule.

This observation **does not prove** that trading cannot be profitable or that
the original backtest has lookahead bias. It distinguishes a descriptive
strategy comparison from an implementable policy that can abstain. The
new rule was proposed after seeing the published summary, so it should be
described as a **post-hoc robustness check**, not as a preregistered strategy.

## Reproduce

```bash
python -m pytest tests/test_no_trade_validation.py
```

To evaluate on refreshed data, pass the Phase 6 `sweep` results to
`select_with_cash_option`, then pass the unchanged forecast frame (with its
`month` column) and frozen selection to `evaluate_cash_option`.
No PostgreSQL connection is required for the unit tests. The baseline source,
published output CSVs and dashboard are intentionally unchanged.
