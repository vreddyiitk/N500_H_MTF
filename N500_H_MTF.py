"""
nifty500_hourly_pipeline.py
===========================
STEP 1 — Download the Nifty 500 constituent list directly from NSE's official
          CSV (no browser / Selenium required for this step):
          https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv

STEP 2 — Generate hourly candlestick charts (last 58 days ≈ 350+ bars)
          for every Nifty 500 stock with:
            • Hourly  EMA9  — yellow  solid line   (computed on hourly close)
            • Daily   EMA9  — orange  solid line   (daily close, forward-filled)
            • Weekly  EMA9  — purple  dashed line  (weekly close, forward-filled)
            • MACD (12, 26, 9) sub-panel
            • MACD histogram turn circles on price panel (● green / ● red)
            • Price pills on right axis (Close / Hourly EMA9 / Daily EMA9 / Weekly EMA9)
            • Title: symbol | NSE | Hourly | Nifty 500 | latest bar timestamp

Requirements:
    pip install requests yfinance pandas matplotlib

Usage:
    python nifty500_hourly_pipeline.py              # normal run
    python nifty500_hourly_pipeline.py --from-csv FILE  # skip download, use saved CSV
"""

import sys
import os
import argparse
import datetime
import traceback
import warnings
from datetime import timedelta

import requests
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import yfinance as yf

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

# ── Nifty 500 CSV (official NSE direct download — no Selenium needed) ────────
NIFTY500_CSV_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
)

EXCHANGE_SFX          = ".NS"
OUTPUT_DIR            = "Nifty500_Hourly_Charts"

HOURLY_LOOKBACK_DAYS  = 58     # yfinance hourly limit ≈ 730 days; 58 is safe
DAILY_LOOKBACK_DAYS   = 365    # daily EMA9 warm-up
WEEKLY_LOOKBACK_DAYS  = 730    # weekly EMA9 warm-up

EMA_PERIOD   = 9
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
MAX_RETRIES  = 3
RETRY_DELAY  = 5               # seconds between retry attempts

# ── Colours ──────────────────────────────────────────────────────────────────
HOURLY_EMA_COLOR = "#FFEB3B"   # bright yellow — solid   (hourly EMA9)
DAILY_EMA_COLOR  = "#FF9800"   # orange        — solid   (daily EMA9 ffilled)
WEEKLY_EMA_COLOR = "#E040FB"   # purple        — dashed  (weekly EMA9 ffilled)
MACD_TURN_BULL   = "#00C853"   # green dot: MACD histogram turns positive
MACD_TURN_BEAR   = "#D50000"   # red dot:   MACD histogram turns negative

STYLE = {
    "bg":          "#131722",
    "panel_bg":    "#1E222D",
    "grid":        "#2A2E39",
    "up_candle":   "#26A69A",
    "down_candle": "#EF5350",
    "macd_line":   "#2962FF",
    "signal_line": "#FF6D00",
    "hist_up":     "#26A69A",
    "hist_down":   "#EF5350",
    "text":        "#D1D4DC",
    "subtext":     "#787B86",
    "border":      "#2A2E39",
}


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — FETCH NIFTY 500 SYMBOL LIST
# ═══════════════════════════════════════════════════════════════

def download_nifty500_csv(url: str) -> pd.DataFrame:
    """
    Download the official Nifty 500 constituents CSV from NSE archives.
    The CSV has columns: Company Name, Industry, Symbol, Series, ISIN Code
    Returns a DataFrame with at least a 'Symbol' column.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    print(f"  Downloading: {url}")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df.columns = df.columns.str.strip()
            print(f"  ✔  Downloaded — {len(df)} rows, columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"  [WARN] Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                import time; time.sleep(RETRY_DELAY)
    return pd.DataFrame()


def get_nifty500_symbols(from_csv: str = None) -> list:
    """
    Returns a clean list of NSE stock symbols from the Nifty 500 index.
    Priority: --from-csv arg → downloaded CSV from NSE archives.
    """
    if from_csv:
        print(f"\n  Loading from local CSV: {from_csv}")
        try:
            df = pd.read_csv(from_csv)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"  [ERROR] Cannot read CSV: {e}")
            sys.exit(1)
    else:
        df = download_nifty500_csv(NIFTY500_CSV_URL)
        if df.empty:
            print("\n  [ERROR] Could not download Nifty 500 list.")
            print("  MANUAL FALLBACK:")
            print(f"  1. Open {NIFTY500_CSV_URL} in your browser and save the file")
            print("  2. Run: python nifty500_hourly_pipeline.py --from-csv ind_nifty500list.csv")
            sys.exit(1)

    # Locate symbol column (NSE CSV uses 'Symbol')
    sym_col = next(
        (c for c in df.columns if c.strip().lower() == "symbol"),
        None
    )
    if sym_col is None:
        # Fallback: try first column
        sym_col = df.columns[0]
        print(f"  [WARN] 'Symbol' column not found — using first column: '{sym_col}'")

    symbols = (
        df[sym_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    # Basic validation: NSE symbols are 2–20 alphanumeric chars (allow & - .)
    symbols = [s for s in symbols if 2 <= len(s) <= 20 and s not in ("SYMBOL", "NAN", "")]

    # Save a local copy for reference / future --from-csv use
    pd.DataFrame({"Symbol": symbols}).to_csv("nifty500_symbols.csv", index=False)
    print(f"  Saved : nifty500_symbols.csv  ({len(symbols)} symbols)")
    return symbols


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — INDICATORS
# ═══════════════════════════════════════════════════════════════

def ema_s(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (Wilder / EWM, adjust=False)."""
    return series.ewm(span=span, adjust=False).mean()


def macd_calc(close: pd.Series):
    """Returns (macd_line, signal_line, histogram)."""
    ml = ema_s(close, MACD_FAST) - ema_s(close, MACD_SLOW)
    sl = ema_s(ml, MACD_SIGNAL)
    return ml, sl, ml - sl


def _dl_close(ticker: str, interval: str, lookback_days: int) -> pd.Series:
    """
    Download OHLC for the given interval and return the raw Close series
    (auto_adjust=False → unadjusted prices, matching NSE / TradingView).
    Strips timezone so the index is tz-naive for easy alignment.
    """
    import time as _time
    end_dt   = datetime.datetime.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=lookback_days)
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=False,   # ← unadjusted — no split/demerger distortion
                progress=False,
            )
            if df is None or df.empty:
                return pd.Series(dtype=float)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            return close
        except Exception:
            if attempt < MAX_RETRIES - 1:
                _time.sleep(RETRY_DELAY)
    return pd.Series(dtype=float)


def fetch_htf_ema9(ticker: str, hourly_index: pd.DatetimeIndex):
    """
    Compute EMA9 on daily and weekly closes, then forward-fill both
    onto the hourly bar index (step-hold between HTF closes).

    Returns
    ───────
    daily_ema9_h  : pd.Series  — daily EMA9 aligned to hourly_index
    weekly_ema9_h : pd.Series  — weekly EMA9 aligned to hourly_index
    """
    daily_close  = _dl_close(ticker, "1d",  DAILY_LOOKBACK_DAYS)
    weekly_close = _dl_close(ticker, "1wk", WEEKLY_LOOKBACK_DAYS)

    daily_ema9  = ema_s(daily_close,  EMA_PERIOD) if not daily_close.empty  \
                  else pd.Series(dtype=float)
    weekly_ema9 = ema_s(weekly_close, EMA_PERIOD) if not weekly_close.empty \
                  else pd.Series(dtype=float)

    def _ffill(htf: pd.Series) -> pd.Series:
        """Union-index forward-fill then reindex to hourly bars."""
        if htf.empty:
            return pd.Series(np.nan, index=hourly_index)
        merged = htf.index.union(hourly_index).sort_values()
        return htf.reindex(merged).ffill().reindex(hourly_index)

    return _ffill(daily_ema9), _ffill(weekly_ema9)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — CHART
# ═══════════════════════════════════════════════════════════════

def plot_chart(
    symbol: str,
    hourly_df: pd.DataFrame,    # tz-naive DatetimeIndex, columns OHLCV
    hourly_ema9: pd.Series,     # EMA9 computed on hourly close
    daily_ema9: pd.Series,      # daily EMA9 forward-filled to hourly
    weekly_ema9: pd.Series,     # weekly EMA9 forward-filled to hourly
    output_path: str,
):
    """
    TradingView-style dark hourly chart:
      Price pane : Candlesticks
                   + Hourly EMA9  (yellow solid)
                   + Daily EMA9   (orange solid, forward-filled from daily close)
                   + Weekly EMA9  (purple dashed, forward-filled from weekly close)
                   + MACD turn circles (green ↑ / red ↓)
      MACD pane  : Histogram + MACD line + Signal line
    """
    s  = STYLE
    n  = len(hourly_df)
    xs = np.arange(n)

    macd_l, sig, hist = macd_calc(hourly_df["Close"])
    hv = hist.values

    # ── MACD histogram direction-change markers ───────────────────────────────
    macd_bull_turn = np.zeros(n, bool)
    macd_bear_turn = np.zeros(n, bool)
    for i in range(1, n):
        if hv[i - 1] <= 0 < hv[i]:
            macd_bull_turn[i] = True
        if hv[i - 1] >= 0 > hv[i]:
            macd_bear_turn[i] = True

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 11), facecolor=s["bg"])
    gs  = gridspec.GridSpec(
        2, 1, height_ratios=[7, 3],
        hspace=0.04, top=0.93, bottom=0.07, left=0.04, right=0.94
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor(s["panel_bg"])
        ax.tick_params(colors=s["subtext"], labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(s["border"])
        ax.grid(True, color=s["grid"], linewidth=0.4, alpha=0.6)

    # ── Candlestick rendering ─────────────────────────────────────────────────
    opens  = hourly_df["Open"].values
    closes = hourly_df["Close"].values
    highs  = hourly_df["High"].values
    lows   = hourly_df["Low"].values

    bull_mask  = closes >= opens
    bear_mask  = ~bull_mask
    body_tops  = np.maximum(opens, closes)
    body_bots  = np.minimum(opens, closes)
    avg_range  = (highs - lows).mean()
    min_body_h = avg_range * 0.04
    doji       = (body_tops - body_bots) < min_body_h
    body_tops  = np.where(doji, body_bots + min_body_h, body_tops)
    BODY_W     = 0.6

    # Wicks
    if bull_mask.any():
        ax1.vlines(xs[bull_mask], lows[bull_mask], highs[bull_mask],
                   color=s["up_candle"], linewidth=1.0, zorder=2)
    if bear_mask.any():
        ax1.vlines(xs[bear_mask], lows[bear_mask], highs[bear_mask],
                   color=s["down_candle"], linewidth=1.0, zorder=2)
    # Bodies
    for i in range(n):
        ax1.add_patch(Rectangle(
            (xs[i] - BODY_W / 2, body_bots[i]),
            BODY_W, body_tops[i] - body_bots[i],
            facecolor=s["up_candle"] if bull_mask[i] else s["down_candle"],
            edgecolor="none", linewidth=0, zorder=3,
        ))

    # ── Hourly EMA9  (yellow solid) ───────────────────────────────────────────
    ax1.plot(xs, hourly_ema9.values,
             color=HOURLY_EMA_COLOR, linewidth=1.6, linestyle="-",
             label=f"Hourly EMA{EMA_PERIOD}", zorder=5)

    # ── Daily EMA9 forward-filled (orange solid) ──────────────────────────────
    d_vals = daily_ema9.reindex(hourly_df.index, method="ffill").values
    ax1.plot(xs, d_vals,
             color=DAILY_EMA_COLOR, linewidth=1.8, linestyle="-",
             label=f"Daily EMA{EMA_PERIOD}", zorder=4)

    # ── Weekly EMA9 forward-filled (purple dashed) ────────────────────────────
    w_vals = weekly_ema9.reindex(hourly_df.index, method="ffill").values
    ax1.plot(xs, w_vals,
             color=WEEKLY_EMA_COLOR, linewidth=1.8, linestyle="--",
             label=f"Weekly EMA{EMA_PERIOD}", zorder=4, alpha=0.90)

    # ── MACD turn circles on price pane ───────────────────────────────────────
    price_range = highs.max() - lows.min()
    circ_off    = price_range * 0.022
    bull_xs = xs[macd_bull_turn]
    bull_ys = lows[macd_bull_turn] - circ_off
    bear_xs = xs[macd_bear_turn]
    bear_ys = highs[macd_bear_turn] + circ_off
    if len(bull_xs):
        ax1.scatter(bull_xs, bull_ys, marker="o", s=40,
                    color=MACD_TURN_BULL, edgecolors=s["bg"],
                    linewidths=0.7, zorder=7)
    if len(bear_xs):
        ax1.scatter(bear_xs, bear_ys, marker="o", s=40,
                    color=MACD_TURN_BEAR, edgecolors=s["bg"],
                    linewidths=0.7, zorder=7)

    # ── Axis styling ──────────────────────────────────────────────────────────
    ax1.set_xlim(-1, n + 1)
    pad = price_range * 0.06
    ax1.set_ylim(lows.min() - pad, highs.max() + pad * 1.5)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.set_ylabel("Price (₹)", color=s["text"], fontsize=9)

    # ── Price pills on right axis ─────────────────────────────────────────────
    last_close  = closes[-1]
    last_h_ema  = hourly_ema9.iloc[-1]
    last_d_ema  = d_vals[-1]
    last_w_ema  = w_vals[-1]
    close_col   = s["up_candle"] if last_close >= opens[-1] else s["down_candle"]

    for val, col, lbl in [
        (last_close, close_col,        f"₹{last_close:,.2f}"),
        (last_h_ema, HOURLY_EMA_COLOR, f"H {last_h_ema:,.2f}"),
        (last_d_ema, DAILY_EMA_COLOR,  f"D {last_d_ema:,.2f}"),
        (last_w_ema, WEEKLY_EMA_COLOR, f"W {last_w_ema:,.2f}"),
    ]:
        if not np.isnan(val):
            ax1.annotate(
                lbl,
                xy=(1, val), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, fontweight="bold", color=s["bg"],
                ha="left", va="center", annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor=col, edgecolor="none", alpha=0.95),
            )

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = [
        mpatches.Patch(facecolor=s["up_candle"],    label="Bullish"),
        mpatches.Patch(facecolor=s["down_candle"],  label="Bearish"),
        Line2D([0], [0], color=HOURLY_EMA_COLOR, lw=1.8, ls="-",
               label=f"Hourly EMA{EMA_PERIOD}"),
        Line2D([0], [0], color=DAILY_EMA_COLOR,  lw=1.8, ls="-",
               label=f"Daily EMA{EMA_PERIOD}"),
        Line2D([0], [0], color=WEEKLY_EMA_COLOR, lw=1.8, ls="--",
               label=f"Weekly EMA{EMA_PERIOD}"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MACD_TURN_BULL, ms=7, ls="None",
               label="MACD turns +ve"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MACD_TURN_BEAR, ms=7, ls="None",
               label="MACD turns -ve"),
    ]
    ax1.legend(handles=leg, loc="upper left", fontsize=7,
               framealpha=0.6, facecolor=s["bg"],
               edgecolor=s["border"], labelcolor=s["text"], ncol=2)

    # ── MACD sub-panel ────────────────────────────────────────────────────────
    hcols = [s["hist_up"] if v >= 0 else s["hist_down"] for v in hv]
    ax2.bar(xs, hv,             color=hcols, alpha=0.85, width=0.7, zorder=2)
    ax2.plot(xs, macd_l.values, color=s["macd_line"],   lw=1.2, zorder=3,
             label="MACD")
    ax2.plot(xs, sig.values,    color=s["signal_line"], lw=1.0, zorder=3,
             label="Signal")
    ax2.axhline(0, color=s["subtext"], lw=0.5, linestyle="--")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("MACD", color=s["text"], fontsize=9)
    ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.6,
               facecolor=s["bg"], edgecolor=s["border"], labelcolor=s["text"])

    # ── X-axis: date on day change, time otherwise ────────────────────────────
    step   = max(n // 14, 1)
    ticks  = xs[::step]
    labels = []
    prev_d = None
    for i in range(0, n, step):
        ts = hourly_df.index[i]
        try:
            ts = ts.tz_convert("Asia/Kolkata") if getattr(ts, "tzinfo", None) else ts
            ts = ts.replace(tzinfo=None)
        except Exception:
            pass
        if prev_d is None or ts.date() != prev_d:
            labels.append(ts.strftime("%d %b\n%H:%M"))
        else:
            labels.append(ts.strftime("%H:%M"))
        prev_d = ts.date()

    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=7, color=s["subtext"],
                        ha="center", linespacing=1.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Title block ───────────────────────────────────────────────────────────
    lc0  = closes[0]
    pct  = (last_close - lc0) / lc0 * 100
    sign = "+" if pct >= 0 else ""
    ccol = s["up_candle"] if pct >= 0 else s["down_candle"]

    try:
        last_ts = hourly_df.index[-1]
        last_ts = last_ts.tz_convert("Asia/Kolkata") \
                  if getattr(last_ts, "tzinfo", None) else last_ts
        ts_str  = last_ts.strftime("%d %b %Y  %H:%M IST")
    except Exception:
        ts_str = str(hourly_df.index[-1])

    n_bturn = int(macd_bull_turn.sum())
    n_rturn = int(macd_bear_turn.sum())

    fig.text(0.05, 0.957,
             f"{symbol}  |  NSE  |  Hourly  |  Nifty 500",
             color=s["text"], fontsize=13, fontweight="bold")
    fig.text(0.05, 0.937,
             f"₹{last_close:,.2f}   {sign}{pct:.2f}%  ({HOURLY_LOOKBACK_DAYS}d)",
             color=ccol, fontsize=10)
    fig.text(0.94, 0.957,
             f"Latest: {ts_str}",
             color=s["text"], fontsize=9, ha="right", fontweight="bold")
    fig.text(0.94, 0.937,
             f"MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"
             f"  |  EMA{EMA_PERIOD}: Hourly (H) · Daily (D) · Weekly (W)"
             f"  |  MACD turns ●{n_bturn} ●{n_rturn}  |  Bars:{n}",
             color=s["subtext"], fontsize=7.5, ha="right")

    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=s["bg"], edgecolor="none")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — BATCH CHART GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_charts(symbols: list):
    import time as _time
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    end_dt   = datetime.datetime.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=HOURLY_LOOKBACK_DAYS)
    success, failed = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols, 1):
        ticker = sym if sym.endswith(EXCHANGE_SFX) else sym + EXCHANGE_SFX
        print(f"\n[{idx:>4}/{total}]  {ticker:<22}", end="  ", flush=True)

        try:
            # ── Download hourly OHLC (unadjusted) ────────────────────────────
            df = pd.DataFrame()
            for attempt in range(MAX_RETRIES):
                df = yf.download(
                    ticker,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    interval="60m",
                    auto_adjust=False,   # ← raw prices, no split adjustment
                    progress=False,
                )
                if not df.empty:
                    break
                if attempt < MAX_RETRIES - 1:
                    _time.sleep(RETRY_DELAY)

            min_bars = MACD_SLOW + MACD_SIGNAL + 5
            if df.empty or len(df) < min_bars:
                print(f"✗  Insufficient data ({len(df)} rows, need {min_bars}+)")
                failed.append(sym)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only raw OHLCV — drop Adj Close
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            # Convert to IST and filter to NSE market hours 09:15–15:30
            try:
                df.index = df.index.tz_convert("Asia/Kolkata")
                df = df.between_time("09:15", "15:30")
            except Exception:
                pass

            if len(df) < min_bars:
                print(f"✗  Too few bars after market filter ({len(df)})")
                failed.append(sym)
                continue

            # ── Strip tz for chart index (x-axis is integer-based) ───────────
            naive_idx = pd.DatetimeIndex([
                t.tz_localize(None) if getattr(t, "tzinfo", None) else t
                for t in df.index
            ])
            df_chart       = df.copy()
            df_chart.index = naive_idx

            # ── Hourly EMA9 (computed directly on hourly close) ───────────────
            h_ema9 = ema_s(df_chart["Close"], EMA_PERIOD)

            # ── Daily + Weekly EMA9 (unadjusted, forward-filled) ─────────────
            d_ema9, w_ema9 = fetch_htf_ema9(ticker, naive_idx)

            # ── Render chart ──────────────────────────────────────────────────
            out = os.path.join(OUTPUT_DIR, f"{sym}.png")
            plot_chart(sym, df_chart, h_ema9, d_ema9, w_ema9, out)
            print(f"✔  {len(df_chart)} bars  →  {out}")
            success.append(sym)

        except Exception:
            print("✗  Exception")
            traceback.print_exc()
            failed.append(sym)

    return success, failed


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nifty 500 → Hourly Charts (Hourly + Daily + Weekly EMA9 + MACD)"
    )
    parser.add_argument(
        "--from-csv", metavar="FILE",
        help="Skip NSE download — load Nifty 500 symbols from a local CSV file",
    )
    args = parser.parse_args()

    run_time = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
    print(f"\n{'═'*68}")
    print(f"  Nifty 500 Hourly Pipeline  —  {run_time}")
    print(f"  Step 1 : Download Nifty 500 list from NSE archives (direct CSV)")
    print(f"  Step 2 : Hourly charts  →  {OUTPUT_DIR}/")
    print(f"           EMA9: Hourly (yellow) · Daily (orange) · Weekly (purple)")
    print(f"           MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) sub-panel + turn circles")
    print(f"           Prices: UNADJUSTED (auto_adjust=False)")
    print(f"{'═'*68}")

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  STEP 1  —  Nifty 500 symbol list")
    print(f"{'─'*68}")
    symbols = get_nifty500_symbols(from_csv=getattr(args, "from_csv", None))

    if not symbols:
        print("\n  ✗  Symbol list is empty — cannot proceed.")
        sys.exit(1)

    print(f"\n  Total Nifty 500 stocks : {len(symbols)}")
    print(f"  First 10               : {symbols[:10]}")

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  STEP 2  —  Generating hourly charts  →  {OUTPUT_DIR}/")
    print(f"  Timeframe    : 1H  |  Lookback : {HOURLY_LOOKBACK_DAYS} days")
    print(f"  Hourly EMA9  : yellow  solid   (computed on hourly close)")
    print(f"  Daily  EMA9  : orange  solid   (daily close, forward-filled)")
    print(f"  Weekly EMA9  : purple  dashed  (weekly close, forward-filled)")
    print(f"{'─'*68}")

    success, failed = generate_charts(symbols)

    print(f"\n{'═'*68}")
    print(f"  PIPELINE COMPLETE  —  {run_time}")
    print(f"  Nifty 500 stocks : {len(symbols)}")
    print(f"  Charts saved     : {len(success)}  →  {OUTPUT_DIR}/")
    if failed:
        print(f"  Charts failed    : {len(failed)}: "
              + ", ".join(failed[:15])
              + (" …" if len(failed) > 15 else ""))
    print(f"  nifty500_symbols.csv — full symbol list")
    print(f"  Prices           : UNADJUSTED (splits/demergers not applied)")
    print(f"{'═'*68}\n")


if __name__ == "__main__":
    main()