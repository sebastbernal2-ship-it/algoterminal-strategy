"""Multi-factor book engine for the crack complex.

Combines four distinct, nearly uncorrelated factors into one portfolio:

  F1  Seasonal mean reversion (WTI product complex)
      - Legs: WTI 3:2:1 crack, WTI heating-oil crack
      - Long when a leg is seasonally crushed (deseasonalized z < -ENTRY)
      - Absolute-value reversion of refining margins
  F2  Cross-sectional crush ranking (WTI complex)
      - Long the single most-crushed leg among {3:2:1, gasoline, heating}
      - Conditional concentration where the reversion is strongest
  F3  Seasonal mean reversion (natgas)
      - Same seasonal-crush logic on natural gas (different commodity)
  F4  Brent-WTI crude convergence
      - Z-score reversion of the BZ-CL crude spread, both sides
      - Weak edge but essentially zero correlation with the rest

The combination is the point: each factor trades a different assumption
(refining-margin pressure vs relative crush strength vs a different
commodity's winter demand vs crude geography), so a failure of any one
assumption is covered by the others. Daily-return correlations between the
factors are 0.00-0.17 in the 3y sample.

Usage:
    python factor_book.py [--start 2023-09-08] [--end 2026-09-08]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

GALLONS = 42.0

# --- parameters (mid-plateau picks, same philosophy as strategy_v4) -------
SMR_Z_LOOKBACK = 90
SMR_ENTRY = 0.75
SMR_EXIT = -0.5
SMR_MIN_OBS = 45
SEASON_MIN_OBS = 10
VOL_LOOKBACK = 20
VT_F1 = 0.50
VT_F2 = 0.50
VT_F3 = 0.50
VT_F4 = 0.15
MAX_LEV = 1.0
STOP_LOOKBACK = 10
STOP_SIGMA = 1.25
TRAILING_STOP_ON = {
    # which factors keep the trailing stop: the crack legs benefit (caps the
    # 2024-09 deep drawdown); NG and Brent-WTI are hurt by it (their vol
    # clusters make the fixed-vol-distance stop cut trades early).
    "crack_321": True,
    "crack_ho": True,
    "cross_sectional": False,
    "ng": False,
    "bzwti": False,
}
DAY_STD_LOOKBACK = 20
DAILY_LOSS_SIGMA = 3.0
COOLDOWN_BARS = 5
HARD_STOP_PCT = 0.20
XS_MIN_Z = -0.5      # require the chosen leg to be at least this crushed
F4_ENTRY = 1.0
F4_EXIT = 0.0


def fetch_panel(start: str, end: str) -> pd.DataFrame:
    tickers = {"CL": "CL=F", "BZ": "BZ=F", "RB": "RB=F", "HO": "HO=F", "NG": "NG=F"}
    raw = {}
    for name, t in tickers.items():
        d = yf.download(
            t,
            start=start,
            end=pd.Timestamp(end) + pd.Timedelta(days=1),
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
        if d.empty:
            print(f"WARN: no data for {name} ({t})")
            continue
        raw[name] = d["Close"]
    return pd.DataFrame(raw).sort_index()


def build_levels(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "crack_321": (2 * df.RB + df.HO) / 3 * GALLONS - df.CL,
        "crack_gas": df.RB * GALLONS - df.CL,
        "crack_ho": df.HO * GALLONS - df.CL,
        "ng": df.NG,
        "bzwti": df.BZ - df.CL,
    }


def seasonal_mean(s: pd.Series, minobs: int = SEASON_MIN_OBS) -> pd.Series:
    out = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        idx = s.index[s.index.month == m]
        for i in range(len(idx)):
            t = idx[i]
            past = s.loc[: t - pd.Timedelta(days=1)]
            past = past[past.index.month == m]
            if len(past) >= minobs:
                out.loc[t] = past.mean()
    return out


def seasonal_z(s: pd.Series) -> pd.Series:
    prev = s.shift(1)
    adj = prev - seasonal_mean(prev)
    mean = adj.rolling(SMR_Z_LOOKBACK, min_periods=SMR_MIN_OBS).mean()
    std = adj.rolling(SMR_Z_LOOKBACK, min_periods=SMR_MIN_OBS).std()
    min_std = 1e-4 * mean.abs()
    valid = std.fillna(0.0) > min_std.fillna(0.0)
    z = (adj - mean) / std.where(valid)
    return z.replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0)


def vol_scale(s: pd.Series, vt: float) -> pd.Series:
    level = s.abs().rolling(VOL_LOOKBACK, min_periods=10).mean().shift(1)
    rel = s.diff() / level
    rv = rel.rolling(VOL_LOOKBACK, min_periods=10).std().shift(1).replace(0.0, np.nan) * np.sqrt(252)
    scale = (vt / rv).clip(upper=MAX_LEV)
    uncond = rel.expanding(min_periods=10).std().shift(1).replace(0.0, np.nan) * np.sqrt(252)
    fallback = (vt / uncond).clip(upper=MAX_LEV)
    return scale.fillna(fallback).fillna(0.5).clip(upper=MAX_LEV)


def apply_leg_risk(pos: pd.Series, level: pd.Series, trailing_stop: bool = False) -> pd.Series:
    """Circuit breaker + hard level stop + cooldown on one leg.

    trailing_stop: optional volatility-distance trend guard. Off by default
    because for vol-clustered instruments (NG, Brent-WTI) it cuts trades on
    normal retracements; on for the crack legs where it caps deep drawdowns.
    """
    out = pos.fillna(0.0).clip(-MAX_LEV, MAX_LEV)
    s = level.to_numpy(dtype=float)
    arr = out.to_numpy(dtype=float).copy()
    prev_held = out.shift(1).fillna(0.0).to_numpy(dtype=float)
    move = level.diff().to_numpy(dtype=float)
    day_std = level.diff().rolling(DAY_STD_LOOKBACK, min_periods=10).std()
    vol = day_std.shift(1).replace(0.0, np.nan).to_numpy(dtype=float)

    maxv = level.shift(1).rolling(STOP_LOOKBACK, min_periods=10).max().to_numpy(dtype=float)
    minv = level.shift(1).rolling(STOP_LOOKBACK, min_periods=10).min().to_numpy(dtype=float)
    dist = vol * STOP_SIGMA * np.sqrt(STOP_LOOKBACK)
    if trailing_stop:
        stop_hit = ((arr > 0.0) & (s < maxv - dist)) | ((arr < 0.0) & (s > minv + dist))
    else:
        stop_hit = np.zeros(len(arr), dtype=bool)
    sigma_move = move / vol
    cb_hit = (prev_held * sigma_move) <= -DAILY_LOSS_SIGMA

    entry_level = np.full(len(arr), np.nan)
    cur_entry = np.nan
    for i in range(len(arr)):
        if arr[i] > 0.0 and np.isnan(cur_entry):
            cur_entry = s[i]
        elif arr[i] == 0.0:
            cur_entry = np.nan
        entry_level[i] = cur_entry
    hard_hit = (arr > 0.0) & (s < entry_level * (1.0 - HARD_STOP_PCT))

    event = np.asarray(stop_hit | cb_hit | hard_hit, dtype=bool)
    arr[event] = 0.0
    n = len(arr)
    for event_i in np.flatnonzero(event):
        hi = min(event_i + 1 + COOLDOWN_BARS, n)
        arr[event_i + 1:hi] = 0.0
    return pd.Series(arr, index=level.index)


def f1_positions(levels: dict[str, pd.Series]) -> dict[str, pd.Series]:
    pos = {}
    for leg in ["crack_321", "crack_ho"]:
        z = seasonal_z(levels[leg])
        zz = z.to_numpy(dtype=float)
        vals = np.zeros(len(levels[leg]))
        state = 0.0
        for i in range(len(levels[leg])):
            if np.isnan(zz[i]):
                vals[i] = 0.0
                continue
            if state == 0.0 and zz[i] <= -SMR_ENTRY:
                state = 1.0
            elif state == 1.0 and zz[i] >= SMR_EXIT:
                state = 0.0
            vals[i] = state
        sig = pd.Series(vals, index=levels[leg].index)
        pos[leg] = apply_leg_risk(sig * vol_scale(levels[leg], VT_F1), levels[leg], trailing_stop=TRAILING_STOP_ON[leg])
    return pos


def f2_positions(levels: dict[str, pd.Series]) -> dict[str, pd.Series]:
    legs = ["crack_321", "crack_gas", "crack_ho"]
    zdf = pd.DataFrame({k: seasonal_z(levels[k]) for k in legs})
    valid = zdf.notna().all(axis=1)
    arr = zdf.to_numpy(dtype=float)
    cols = list(zdf.columns)
    chosen = pd.Series(np.nan, index=zdf.index, dtype=float)
    chosen_level = pd.Series(np.nan, index=zdf.index, dtype=float)
    for i in range(len(zdf)):
        if valid.iloc[i]:
            row = arr[i]
            if np.isnan(row).all():
                continue
            k = cols[int(np.nanargmin(row))]
            if row[int(np.nanargmin(row))] < XS_MIN_Z:
                chosen.iloc[i] = legs.index(k)
                chosen_level.iloc[i] = levels[k].iloc[i]
    on = chosen.notna()
    sig = pd.Series(0.0, index=zdf.index)
    sig[on] = 1.0
    # Vol size on the chosen leg's level
    scale = vol_scale(chosen_level.fillna(levels["crack_321"]), VT_F2).where(on, 0.0)
    pos = sig * scale
    return {"cross_sectional": apply_leg_risk(pos, chosen_level.fillna(levels["crack_321"]), trailing_stop=TRAILING_STOP_ON["cross_sectional"])}


def f3_positions(levels: dict[str, pd.Series]) -> dict[str, pd.Series]:
    ng = levels["ng"]
    z = seasonal_z(ng)
    zz = z.to_numpy(dtype=float)
    vals = np.zeros(len(ng))
    state = 0.0
    for i in range(len(ng)):
        if np.isnan(zz[i]):
            vals[i] = 0.0
            continue
        if state == 0.0 and zz[i] <= -SMR_ENTRY:
            state = 1.0
        elif state == 1.0 and zz[i] >= SMR_EXIT:
            state = 0.0
        vals[i] = state
    sig = pd.Series(vals, index=ng.index)
    pos = sig * vol_scale(ng, VT_F3)
    return {"ng": apply_leg_risk(pos, ng, trailing_stop=TRAILING_STOP_ON["ng"])}


def f4_positions(levels: dict[str, pd.Series]) -> dict[str, pd.Series]:
    bw = levels["bzwti"]
    prev = bw.shift(1)
    mean = prev.rolling(60, min_periods=30).mean()
    std = prev.rolling(60, min_periods=30).std()
    z = ((prev - mean) / std).replace([np.inf, -np.inf], np.nan)
    zz = z.to_numpy(dtype=float)
    vals = np.zeros(len(bw))
    state = 0.0
    for i in range(len(bw)):
        if np.isnan(zz[i]):
            vals[i] = 0.0
            continue
        if state == 0.0:
            if zz[i] < -F4_ENTRY:
                state = 1.0
            elif zz[i] > F4_ENTRY:
                state = -1.0
        elif state == 1.0:
            if zz[i] >= F4_EXIT:
                state = 0.0
        else:
            if zz[i] <= F4_EXIT:
                state = 0.0
        vals[i] = state
    sig = pd.Series(vals, index=bw.index)
    pos = sig * vol_scale(bw, VT_F4)
    return {"bzwti": apply_leg_risk(pos, bw, trailing_stop=TRAILING_STOP_ON["bzwti"])}


def stats(r: pd.Series) -> tuple[float, float, float]:
    eq = (1 + r).cumprod()
    c = (eq.iloc[-1] ** (252 / len(r)) - 1) * 100 if len(r) else 0.0
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0
    dd = (eq / eq.cummax() - 1).min() * 100
    return c, sh, dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-09-08")
    ap.add_argument("--end", default="2026-09-08")
    args = ap.parse_args()

    print("Fetching panel...")
    df = fetch_panel(args.start, args.end)
    levels = build_levels(df)

    factors = {}
    factors.update(f1_positions(levels))
    factors.update(f2_positions(levels))
    factors.update(f3_positions(levels))
    factors.update(f4_positions(levels))

    # Recompute per-factor returns with the correct held level for cross-sectional
    factor_rets = {}
    for name, pos in factors.items():
        if name == "cross_sectional":
            legs = {"crack_321": levels["crack_321"], "crack_gas": levels["crack_gas"], "crack_ho": levels["crack_ho"]}
            pos_idx = pos.index
            # Build held-level series from the chosen leg each day
            zdf = pd.DataFrame({k: seasonal_z(lvl) for k, lvl in legs.items()})
            arr = zdf.to_numpy(dtype=float)
            cols = list(zdf.columns)
            held_level = pd.Series(np.nan, index=pos_idx)
            for i in range(len(zdf)):
                if pos.iloc[i] != 0.0 and not np.isnan(arr[i]).all():
                    held_level.iloc[i] = levels[cols[int(np.nanargmin(arr[i]))]].iloc[i]
            prev_held = held_level.shift(1)
            r = pos.shift(1).fillna(0.0) * prev_held.pct_change().fillna(0.0)
            factor_rets[name] = r.fillna(0.0)
        else:
            level = levels[name]
            r = pos.shift(1).fillna(0.0) * level.pct_change().fillna(0.0)
            factor_rets[name] = r.fillna(0.0)

    print("\n=== FACTOR BOOK (2023-09 to 2026-09) ===")
    fr = pd.DataFrame(factor_rets)
    fr = fr.loc[fr.index >= "2024-01-01"]  # skip warm-up
    print("\nPer-factor stats (post-warm-up):")
    for col in fr.columns:
        c, sh, dd = stats(fr[col])
        days = (fr[col] != 0).mean() * 100
        print("  %-14s CAGR=%7.2f%% Sharpe=%5.2f MaxDD=%7.2f%% daysOn=%5.1f%%" % (col, c, sh, dd, days))
    print("\nCorrelation matrix:")
    print(fr.corr().round(2).to_string())

    # Book: equal-vol (inverse-vol) weights
    vols = fr.std()
    w = (1.0 / vols.replace(0.0, np.nan))
    w = w / w.sum()
    book = fr.mul(w, axis=1).sum(axis=1)
    c, sh, dd = stats(book)
    print("\n=== COMBINED BOOK (inverse-vol weights) ===")
    print("weights: %s" % ", ".join("%s=%.2f" % (k, v) for k, v in w.items()))
    print("CAGR=%7.2f%% Sharpe=%5.2f MaxDD=%7.2f%%  worstDay=%+.2f%%" % (c, sh, dd, book.min() * 100))
    print("annualized book vol: %.0f%%" % (book.std() * np.sqrt(252) * 100))


if __name__ == "__main__":
    main()