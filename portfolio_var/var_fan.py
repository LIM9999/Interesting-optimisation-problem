"""
Portfolio VaR + 1-year fan chart (GBP base).

Data: tries yfinance. Full history is pulled once into an archive CSV; later runs
only pull the last ~60 days and append. If no data source is reachable, falls back
to the assumed vol / correlation matrix below (stated in the output).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "price_archive.csv")

# ---- Holdings (GBP market value, 24-Nov-2026 screenshots) --------------------
H = pd.Series({
    "MCD":   3854.72,
    "VUAG":  3462.63,
    "SSLN":  1764.99,
    "BTC":   1423.08,
    "GOOGL": 1374.15,
    "SGLN":   775.62,
})
TICKERS = {"MCD": "MCD", "VUAG": "VUAG.L", "SSLN": "SSLN.L", "BTC": "BTC-GBP",
           "GOOGL": "GOOGL", "SGLN": "SGLN.L"}
USD_ASSETS = ["MCD", "GOOGL"]

# ---- Fallback: trailing ~1y vols from published sources (Nov-2026), GBP-adjusted ----
# BTC 43% (Glassnode 1y realised), MCD 19% USD (Trefis 12m) +FX -> 20%,
# GOOGL 32% (Trefis 1y realised) +FX -> 33%, S&P ~15-18% (VIX avg 18) -> 16%,
# silver ~37% long-run, 2026 elevated -> 40%, gold Q1 >20%, spiked >50% mid-year -> 22%
VOL_ASSUMED = pd.Series({"MCD": .20, "VUAG": .16, "SSLN": .40, "BTC": .43,
                         "GOOGL": .33, "SGLN": .22})
CORR_ASSUMED = pd.DataFrame(
    #  MCD   VUAG  SSLN  BTC   GOOGL SGLN
    [[1.00, 0.40, 0.08, 0.08, 0.25, 0.08],
     [0.40, 1.00, 0.25, 0.30, 0.68, 0.15],
     [0.08, 0.25, 1.00, 0.15, 0.15, 0.75],
     [0.08, 0.30, 0.15, 1.00, 0.25, 0.10],
     [0.25, 0.68, 0.15, 0.25, 1.00, 0.10],
     [0.08, 0.15, 0.75, 0.10, 0.10, 1.00]],
    index=VOL_ASSUMED.index, columns=VOL_ASSUMED.index)

MU_ANNUAL = 0.06      # assumed arithmetic drift for the fan chart (set 0 for pure risk view)
TRADING_DAYS = 252
N_SIMS = 100_000
SEED = 42


def load_prices():
    try:
        import yfinance as yf
    except ImportError:
        return None
    syms = list(TICKERS.values()) + ["GBPUSD=X"]
    try:
        if os.path.exists(ARCHIVE):
            arch = pd.read_csv(ARCHIVE, index_col=0, parse_dates=True)
            new = yf.download(syms, period="60d", auto_adjust=True, progress=False)["Close"]
            px = pd.concat([arch, new]).groupby(level=0).last().sort_index()
        else:
            px = yf.download(syms, period="5y", auto_adjust=True, progress=False)["Close"]
        if px.dropna(how="all").empty:
            return None
        px.to_csv(ARCHIVE)
        return px
    except Exception:
        return None


def returns_from_prices(px, lookback_years=3):
    px = px.ffill().dropna()
    px = px[px.index >= px.index.max() - pd.DateOffset(years=lookback_years)]
    out = pd.DataFrame(index=px.index)
    for k, t in TICKERS.items():
        s = px[t]
        if k in USD_ASSETS:
            s = s / px["GBPUSD=X"]          # USD price -> GBP
        out[k] = s
    # BTC trades 7 days; aligning on business days handles it
    return np.log(out).diff().dropna()[H.index]


def main():
    px = load_prices()
    if px is not None:
        r = returns_from_prices(px)
        cov_d = r.cov().values
        vol = pd.Series(np.sqrt(np.diag(cov_d) * TRADING_DAYS), index=H.index)
        corr = r.corr()
        src = f"historical, {r.index.min():%Y-%m-%d} to {r.index.max():%Y-%m-%d}"
    else:
        r = None
        vol, corr = VOL_ASSUMED[H.index], CORR_ASSUMED.loc[H.index, H.index]
        cov_d = (np.outer(vol, vol) * corr.values) / TRADING_DAYS
        src = "Vols from published 1y figures (Nov-2026); correlations assumed"

    V = H.sum()
    w = (H / V).values
    cov_a = cov_d * TRADING_DAYS
    sig_p = float(np.sqrt(w @ cov_a @ w))
    sig_wavg = float(w @ vol.values)

    print(f"Source: {src}")
    print(f"Portfolio value: £{V:,.2f}\n")
    print(pd.DataFrame({"Value £": H, "Weight": w.round(4), "Vol (ann)": vol.round(3)}).to_string())
    print(f"\nWeighted-average vol (no diversification): {sig_wavg:.2%}")
    print(f"Portfolio vol (with correlations):         {sig_p:.2%}")
    print(f"Diversification benefit:                   {sig_wavg - sig_p:.2%}\n")

    # ---- Parametric VaR / ES (delta-normal, zero mean) ----
    rows = []
    for h_lbl, h in [("1 day", 1), ("10 day", 10), ("1 month", 21), ("1 year", 252)]:
        s = sig_p * np.sqrt(h / TRADING_DAYS)
        for c in (0.95, 0.99):
            z = norm.ppf(c)
            rows.append([h_lbl, f"{c:.0%}", V * z * s, V * s * norm.pdf(z) / (1 - c)])
    pv = pd.DataFrame(rows, columns=["Horizon", "Conf", "VaR £", "ES £"])
    print("Parametric VaR / Expected Shortfall")
    print(pv.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # ---- Component VaR (1y 95%) ----
    z95 = norm.ppf(0.95)
    marg = (cov_a @ w) / sig_p
    comp = H.values * marg * z95
    cv = pd.DataFrame({"Comp VaR £ (1y 95%)": comp, "% of VaR": comp / comp.sum()}, index=H.index)
    print("\nComponent VaR (Euler decomposition, 1y 95%)")
    print(cv.to_string(float_format=lambda x: f"{x:,.2f}"))

    # ---- Historical VaR (if data) ----
    if r is not None:
        pnl = (np.exp(r.values) - 1) @ H.values
        for c in (0.95, 0.99):
            print(f"Historical 1d VaR {c:.0%}: £{-np.percentile(pnl, (1-c)*100):,.0f}")

    # ---- Monte Carlo 1y, multi-asset, buy-and-hold, lognormal ----
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(cov_a)
    Z = rng.standard_normal((N_SIMS, len(w))) @ L.T
    mu_i = MU_ANNUAL - 0.5 * np.diag(cov_a)
    end = (H.values * np.exp(mu_i + Z)).sum(axis=1)
    pnl1y = end - V
    print("\nMonte Carlo 1y (multi-asset, drift {:.0%})".format(MU_ANNUAL))
    for c in (0.95, 0.99):
        q = np.percentile(pnl1y, (1 - c) * 100)
        es = pnl1y[pnl1y <= q].mean()
        print(f"  {c:.0%}: VaR £{-q:,.0f}   ES £{-es:,.0f}")
    print(f"  P(loss after 1y): {np.mean(pnl1y < 0):.1%}")

    # ---- Fan chart: single-factor GBM at portfolio vol ----
    t = np.linspace(0, 1, 253)
    pcts = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    drift = (MU_ANNUAL - 0.5 * sig_p**2) * t
    band = {p: V * np.exp(drift + norm.ppf(p) * sig_p * np.sqrt(t)) for p in pcts}
    dates = pd.bdate_range("2026-11-24", periods=253)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    base = "#1f5fa8"
    for lo, hi, a in [(0.05, 0.95, .15), (0.10, 0.90, .22), (0.25, 0.75, .35)]:
        ax.fill_between(dates, band[lo], band[hi], color=base, alpha=a, lw=0,
                        label=f"{int(lo*100)}–{int(hi*100)}th pct")
    ax.plot(dates, band[0.5], color=base, lw=2, label="Median")
    ax.axhline(V, color="grey", lw=1, ls="--", label=f"Today £{V:,.0f}")
    for p in pcts:
        ax.annotate(f"P{int(p*100)}  £{band[p][-1]:,.0f}", (dates[-1], band[p][-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_title(f"1-year projection  |  σp = {sig_p:.1%}, μ = {MU_ANNUAL:.0%}, GBM", loc="left")
    ax.set_ylabel("Portfolio value (£)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=60))
    fig.text(0.01, 0.01, f"Source: {src}", fontsize=7, color="grey")
    fig.tight_layout()
    out = os.path.join(HERE, "fan_chart.png")
    fig.savefig(out)
    print("\n1y percentile values:", {f"P{int(p*100)}": round(band[p][-1]) for p in pcts})
    print("Saved", out)


if __name__ == "__main__":
    main()
