# NSE Momentum Dashboard

Two dashboards share this logic (see [`../DEPLOY.md`](../DEPLOY.md)):
`streamlit_app.py` (live, with charts + performance tabs) and this static
GitHub Pages page (EOD, cron-refreshed). Both show, for a large/mid-cap NSE
universe:

- **Top momentum picks** — top-ranked **idiosyncratic** (beta-residual) momentum
  names, the best signal from the research in [`../backtests`](../backtests)
  (~+5%/yr alpha vs equal-weight, lumpy & crash-prone — **not trading advice**).
- **Price + 1-day change**, and **Buy/Hold/Sell** vs the **20/50/200-day SMAs**
  with an overall signal.
- **Context:** distance below 52-week high, 60-day annualised volatility, and
  average daily traded value (₹cr) — a real liquidity filter (picks must be
  ≥₹50 and ≥₹5cr/day, so penny/illiquid names can't top the list).
- **Sector**, **ROE** (TTM) and **ROCE** (EBIT ÷ capital employed, last quarter;
  "n/m" when distorted, e.g. banks / negative equity).
- **Live-forward track record** (`docs/track_record.json`) — a paper portfolio
  the daily job marks-to-market, so git history is an auditable record of what
  was recommended and how it did. The Streamlit **Performance** tab plots it
  alongside the backtested equity curve.

## Roadmap / not included
- **Alerts:** set a repo secret `ALERT_WEBHOOK` (Slack/Discord URL) to get a
  post when picks/signals change. Off by default.
- **Point-in-time / de-survivorship data:** the one genuine limitation — the
  universe is *today's* constituents, so backtest numbers are survivorship-
  inflated. Fixing it needs a paid data vendor; see `../backtests` for the
  honest analysis of how much that biases the results.

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
