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

- 3-year sample (2023-09 to 2026-09), in-sample, parameters are mid-plateau
  picks, not peaks.
- No costs, slippage, or futures roll modeling.
- Small trade counts per factor (crack legs ~9-11 trades in 2.75 years).
- Combined Sharpe ~2.6 in-sample is a backtest artifact risk until confirmed
  out-of-sample; do not size capital to it yet.
- Next steps: 10y validation, transaction-cost/roll model, options/tail
  overlay.