"""
track_record.py — Live-forward, auditable paper track record.

Each daily run advances a paper portfolio: hold the top-N momentum picks equal-
weight, rebalance monthly, mark-to-market daily. The NAV series (and an index
NAV for comparison) is written to docs/track_record.json and committed, so git
history is an immutable, timestamped record of what the strategy recommended and
how it actually did — the thing that separates a credible quant tool from a toy.

Gross of costs (monthly turnover is light); treat as indicative. The LIVE curve
starts at 100 on first run and accumulates out-of-sample from there.
"""
import json, os


def _load(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"history": [], "holdings": {}, "last_rebal_month": None, "start": 100.0}


def advance(path, today, picks, prices, idx_level):
    """today: 'YYYY-MM-DD'. picks: ordered top-N symbols. prices: {sym: price}.
    idx_level: current benchmark index level."""
    st = _load(path)

    # mark-to-market existing holdings
    if st["holdings"] and all(prices.get(s) for s in st["holdings"]):
        nav = sum(sh * prices[s] for s, sh in st["holdings"].items())
    elif st["history"]:
        nav = st["history"][-1]["nav"]
    else:
        nav = st["start"]

    # rebalance monthly (or on first run) into current picks, equal weight
    month = today[:7]
    valid = [s for s in picks if prices.get(s, 0) > 0]
    if valid and (not st["holdings"] or st["last_rebal_month"] != month):
        per = nav / len(valid)
        st["holdings"] = {s: per / prices[s] for s in valid}
        st["last_rebal_month"] = month
        nav = sum(sh * prices[s] for s, sh in st["holdings"].items())

    # index NAV: normalise so both curves start at 100 on the first logged day
    if not st["history"]:
        st["idx_base"] = idx_level or 1.0
    idx_nav = (idx_level / st.get("idx_base", idx_level or 1.0)) * st["start"] if idx_level else None

    if not st["history"] or st["history"][-1]["date"] != today:
        st["history"].append({"date": today, "nav": round(nav, 3),
                              "idx": round(idx_nav, 3) if idx_nav else None})
    else:  # same-day rerun: update in place
        st["history"][-1] = {"date": today, "nav": round(nav, 3),
                             "idx": round(idx_nav, 3) if idx_nav else None}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(st, f, indent=2)
    return st


def load_history(path):
    return _load(path).get("history", [])
