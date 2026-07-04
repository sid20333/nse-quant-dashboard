# NSE Momentum Dashboard

A self-updating GitHub Pages site showing, for a large/mid-cap NSE universe:

- **Top momentum picks** — the top-ranked 12-month momentum names (the only
  signal that survived the research in [`../backtests`](../backtests); ~+5%/yr
  alpha vs equal-weight, but lumpy and crash-prone — **not trading advice**).
- **Current price + 1-day change** (end-of-day, via Yahoo Finance).
- **Buy / Hold / Sell tags** vs the **20, 50 and 200-day SMAs**, plus an overall
  signal.
- **ROE** (trailing-twelve-month) and **ROCE** (EBIT ÷ capital employed,
  annualised from the latest quarterly statement). ROCE is blank for banks — it
  isn't a meaningful metric for a financial's balance sheet.

## How it updates

`dashboard/generate.py` fetches fresh EOD data and rewrites `docs/index.html`
and `docs/data.json`. The GitHub Actions workflow
[`.github/workflows/update-dashboard.yml`](../.github/workflows/update-dashboard.yml)
runs it every weekday at 11:00 UTC (after NSE close) and commits the refresh;
GitHub Pages serves `docs/`.

## One-time setup on GitHub

1. Push this repo to GitHub.
2. **Settings → Pages** → Source = *Deploy from a branch*, Branch = `main`,
   Folder = `/docs`. Save. Your page will be at
   `https://<user>.github.io/<repo>/`.
3. **Settings → Actions → General** → Workflow permissions = *Read and write*
   (so the scheduled job can commit the refresh).
4. Trigger the first build: **Actions → Update dashboard → Run workflow**
   (or wait for the next weekday 11:00 UTC).

## Run it locally

```bash
pip install yfinance pandas numpy
python dashboard/generate.py            # full universe (~a few minutes)
python dashboard/generate.py --limit 10 # quick test on 10 names
open docs/index.html
```

## Notes / honesty

- EOD data only. "Current price" = latest close, not intraday.
- Yahoo data can be delayed, adjusted, or wrong. Verify before acting.
- Fundamentals are cached 7 days (`dashboard/fundamentals_cache.json`) so the
  daily price refresh doesn't re-hit slow endpoints.
- The "recommendations" are a mechanical momentum ranking, not a view on any
  company. See the research writeups in `../backtests` for exactly how much
  (and how little) that edge is worth.
