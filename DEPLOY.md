# Deploying the live dashboard (Streamlit Community Cloud)

The live version is `streamlit_app.py`. It fetches prices server-side (no CORS
issues) and auto-refreshes every 15s, so you get near-live ticks during market
hours. Fundamentals (ROE/ROCE) are read from the committed
`dashboard/fundamentals_cache.json` — refresh that occasionally with
`python dashboard/generate.py` and commit.

## Deploy (one time, free)

1. Push this repo to GitHub.
2. Go to **https://share.streamlit.io** → sign in with GitHub → **New app**.
3. Pick this repo, branch `main`, main file `streamlit_app.py`. Deploy.
4. Your app is live at `https://<something>.streamlit.app`.

Streamlit Cloud installs `requirements.txt` automatically. First load takes a
few seconds (it pulls 2y of history for the universe, cached hourly); after that
only the latest price is refetched each refresh.

Notes:
- The app **sleeps after inactivity** on the free tier and wakes on next visit
  (takes ~30s to spin up).
- Yahoo prices can be delayed ~15 min for NSE; the badge shows market open/closed.
- To refresh fundamentals: `python dashboard/generate.py` locally, then commit
  the updated `dashboard/fundamentals_cache.json`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## The two dashboards

| File | Hosting | Data | Best for |
|---|---|---|---|
| `streamlit_app.py` | Streamlit Cloud | **Live** (server-side, auto-refresh) | live ticks |
| `dashboard/generate.py` → `docs/` | GitHub Pages | EOD (cron-refreshed) | a static, always-on page with no cold start |

They share the same logic (momentum picks, SMA Buy/Hold/Sell, ROE/ROCE, and the
≥₹50 / "n/m" artifact fixes). Keep whichever you prefer, or both.
