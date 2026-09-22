# Crack Spread Hedging Pressure -> Mean Reversion (Factor Book)

Multi-factor mean-reversion book on energy refined-margin stress, expressed
in five nearly independent markets. Single self-contained file.

## Strategy

Refiners are structurally short the crack spread: they buy crude, sell
products, and capture the margin. When a refined-product margin is crushed
below its normal level for the season, the physical economy is forced to
adjust (run cuts, yield shifts, imports). The margin then reverts. This file
trades that reversion in five factors:

| Factor | Market | Construction |
| --- | --- | --- |
| F1a | WTI 3:2:1 crack | seasonal crush, long |
| F1b | WTI heating-oil crack | seasonal crush, long |
| F2 | WTI complex | cross-sectional: long the most-crushed leg |
| F3 | Natural gas | seasonal crush, long |
| F4 | Brent-WTI basis | z-reversion, both sides |

Factors are combined with inverse-volatility weights. Daily-return
correlations between factors are 0.00-0.25 in the sample.

All rolling statistics use data strictly before bar t (inputs shifted one
bar), so no lookahead.

## Run

```bash
pip install numpy pandas yfinance
python factor_book.py               # 3y default window
python factor_book.py --start 2016-01-01 --end 2026-09-08
```

Prints per-factor stats, correlation matrix, and the combined book.

## Honest caveats

**AUDIT UPDATE (2026-09-22) — read this before using any number.**

- Return basis fixed. P&L was priced as `level.pct_change()`, which explodes
  when a spread crosses zero (`bzwti` crosses ~548 times). The file now uses
  `diff / rolling_mean(|level|)`, matching the audit-corrected engine.
- The price panel is yfinance continuous front-month futures, which are
  **not** back-adjusted. The legs roll on different dates, so the crack
  series gains **+3.915 $/bbl every March, in 18 of 18 years**, and repays it
  across the other months. Those sessions were **40.2%** of the measured P&L
  in a later walk-forward. This is contract construction, not refining margin.
- On roll-free spot prices (EIA), at measured real cost (16-24 bps/side, not
  5), the edge is **not statistically significant**: ann +4% to +8%,
  Sharpe 0.39 to 0.67, block t 1.0 to 2.3, DSR 0.04 to 0.23, on an
  architecture selected in-sample.
- Full record: `algoterminal-strategy-v2/findings/artifact_audit.md`.

Older caveats, still true:

- 3-year sample default, in-sample, parameters are mid-plateau picks,
  not peaks.
- No costs, slippage, or futures roll modeling in this file.
- Small trade counts per factor (crack legs ~9-11 trades in 2.75 years).
- Combined Sharpe ~2.6 in-sample is a backtest artifact. Do not size
  capital to it.