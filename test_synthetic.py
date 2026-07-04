"""
test_synthetic.py — Proves the whole pipeline runs end-to-end without errors,
using synthetic data. This validates CODE CORRECTNESS ONLY, not strategy
profitability. Swap SyntheticDataProvider for BreezeDataProvider + real
history before drawing any conclusions about the strategy itself.
"""
import sys
sys.path.insert(0, "/home/claude")

from quant_engine.data_provider import SyntheticDataProvider
from quant_engine import valuation as val
from quant_engine import technical as tech
from quant_engine import zones as zn
from quant_engine import vwap as vw
from quant_engine.knowledge_base import StaticKnowledgeBase
from quant_engine.engine import StockInputs, evaluate_stock, run_weekly_scan
from quant_engine.backtest import walk_forward_backtest

print("=" * 70)
print("1. VALUATION LAYER")
print("=" * 70)
dcf = val.dcf_intrinsic_value(
    fcf_projections=[100, 112, 125, 138, 150],
    terminal_growth=0.05,
    discount_rate=0.12,
    shares_outstanding=50,
    net_debt=200,
)
graham = val.graham_intrinsic_value(eps=25, growth_rate_pct=12, aaa_bond_yield_pct=7.2)
epv = val.epv(normalized_ebit=180, tax_rate=0.25, wacc=0.12, maintenance_capex=30, shares_outstanding=50, net_debt=200)
blended = val.blended_valuation(market_value=450, dcf_value=dcf, graham_value=graham, epv_value=epv)
print(f"DCF/share: {dcf:.1f} | Graham: {graham:.1f} | EPV/share: {epv:.1f}")
print(f"Blended IV: {blended.blended_iv:.1f} | MoS: {blended.margin_of_safety:.1%} | Spread: {blended.agreement_spread:.1%}")

print("\n" + "=" * 70)
print("2. DATA PROVIDER (synthetic)")
print("=" * 70)
provider = SyntheticDataProvider(seed=7)
df = provider.get_daily_ohlcv("TESTCO", "2022-01-01", "2026-06-30")
print(f"Generated {len(df)} bars from {df['date'].min().date()} to {df['date'].max().date()}")
print(df.tail(3).to_string(index=False))

print("\n" + "=" * 70)
print("3. TECHNICAL LAYER (squeeze detection)")
print("=" * 70)
sq = tech.detect_triple_squeeze(df)
n_squeezes = sq["triple_squeeze"].sum()
print(f"Triple-squeeze bars found: {n_squeezes} out of {len(sq)}")
print(f"Latest bandwidth percentile ranks -> 20d:{sq['bb20_pct_rank'].iloc[-1]:.2f} "
      f"50d:{sq['bb50_pct_rank'].iloc[-1]:.2f} 100d:{sq['bb100_pct_rank'].iloc[-1]:.2f}")

print("\n" + "=" * 70)
print("4. ZONES LAYER (S/R clusters + order blocks)")
print("=" * 70)
sr_zones = zn.find_support_resistance_zones(df, min_touches=3)
print(f"S/R zones found: {len(sr_zones)}")
for z in sr_zones[:5]:
    print(f"  {z.zone_type}: {z.price_low:.1f}-{z.price_high:.1f} ({z.touches} touches)")

order_blocks = zn.detect_order_blocks(df)
print(f"Order blocks (drop-base-rally) found: {len(order_blocks)}")
for ob in order_blocks[:5]:
    print(f"  zone {ob.zone_low:.1f}-{ob.zone_high:.1f}, breakout vol ratio {ob.breakout_volume_ratio:.2f}x")

print("\n" + "=" * 70)
print("5. VWAP LAYER (anchored from last major swing low)")
print("=" * 70)
anchor_idx = vw.last_major_swing_low(df)
print(f"Anchor index: {anchor_idx} -> date {df['date'].iloc[anchor_idx].date() if anchor_idx else None}")
if anchor_idx is not None:
    vwap_series = vw.anchored_vwap(df, anchor_idx)
    holding = vw.price_above_vwap(df, vwap_series)
    print(f"Current price {df['close'].iloc[-1]:.1f} vs VWAP {vwap_series.iloc[-1]:.1f} "
          f"| holding above (3-bar confirm): {bool(holding.iloc[-1])}")

print("\n" + "=" * 70)
print("6. KNOWLEDGE BASE LAYER")
print("=" * 70)
kb = StaticKnowledgeBase(pledge_threshold_pct=20, earnings_decline_threshold_pct=15)
kb.upsert("TESTCO", promoter_pledge_pct=5, yoy_earnings_decline_pct=2, forensic_audit_flag=False,
          auditor_resignation_flag=False, last_earnings_date="2026-04-15")
flags = kb.get_red_flags("TESTCO")
print(f"TESTCO red flags: disqualified={flags.disqualified}, reasons={flags.reasons}")

flags_unknown = kb.get_red_flags("UNKNOWNCO")
print(f"UNKNOWNCO (no data) red flags: disqualified={flags_unknown.disqualified}, reasons={flags_unknown.reasons}")

print("\n" + "=" * 70)
print("7. FULL ENGINE (all gates combined)")
print("=" * 70)
inputs = StockInputs(
    symbol="TESTCO",
    ohlcv=df,
    eps=25,
    growth_rate_pct=12,
    aaa_bond_yield_pct=7.2,
    normalized_ebit=180,
    tax_rate=0.25,
    wacc=0.12,
    maintenance_capex=30,
    shares_outstanding=50,
    net_debt=200,
    fcf_projections=[100, 112, 125, 138, 150],
    terminal_growth=0.05,
    discount_rate=0.12,
)
result = evaluate_stock(inputs, kb)
print(f"Symbol: {result.symbol}")
mos_str = f"{result.margin_of_safety:.1%}" if result.margin_of_safety is not None else "n/a"
print(f"  Valuation gate:  {result.passed_valuation} (MoS={mos_str})")
print(f"  KB gate:         {result.passed_kb} {result.kb_reasons}")
print(f"  Trend gate:      {result.passed_trend}")
print(f"  Zone gate:       {result.passed_zone} type={result.zone_type} bounds={result.zone_bounds}")
print(f"  Volatility gate: {result.passed_volatility_trigger}")
print(f"  FULLY QUALIFIED: {result.fully_qualified}")

print("\n" + "=" * 70)
print("8. MULTI-STOCK WEEKLY SCAN")
print("=" * 70)
universe = []
for i, seed in enumerate([1, 2, 3, 4, 5]):
    p = SyntheticDataProvider(seed=seed)
    d = p.get_daily_ohlcv(f"STOCK{i}", "2022-01-01", "2026-06-30")
    kb.upsert(f"STOCK{i}", promoter_pledge_pct=5 + i * 3, yoy_earnings_decline_pct=2,
              forensic_audit_flag=False, auditor_resignation_flag=False, last_earnings_date="2026-04-15")
    universe.append(StockInputs(
        symbol=f"STOCK{i}", ohlcv=d, eps=20 + i, growth_rate_pct=10 + i, aaa_bond_yield_pct=7.2,
        normalized_ebit=150 + i * 10, tax_rate=0.25, wacc=0.12, maintenance_capex=25,
        shares_outstanding=50, net_debt=150, fcf_projections=[90, 100, 112, 124, 136],
        terminal_growth=0.05, discount_rate=0.12,
    ))

scan_results = run_weekly_scan(universe, kb)
for r in scan_results:
    gates = sum([r.passed_valuation, r.passed_kb, r.passed_trend, r.passed_zone, r.passed_volatility_trigger])
    print(f"  {r.symbol}: gates_passed={gates}/5 fully_qualified={r.fully_qualified}")

print("\n" + "=" * 70)
print("9. BACKTEST (walk-forward, synthetic universe)")
print("=" * 70)

symbol_dfs = {inp.symbol: inp.ohlcv for inp in universe}
static_inputs_by_symbol = {inp.symbol: inp for inp in universe}

def signal_fn(symbol, sliced_df):
    base_inputs = static_inputs_by_symbol[symbol]
    inp = StockInputs(
        symbol=symbol, ohlcv=sliced_df, eps=base_inputs.eps, growth_rate_pct=base_inputs.growth_rate_pct,
        aaa_bond_yield_pct=base_inputs.aaa_bond_yield_pct, normalized_ebit=base_inputs.normalized_ebit,
        tax_rate=base_inputs.tax_rate, wacc=base_inputs.wacc, maintenance_capex=base_inputs.maintenance_capex,
        shares_outstanding=base_inputs.shares_outstanding, net_debt=base_inputs.net_debt,
        fcf_projections=base_inputs.fcf_projections, terminal_growth=base_inputs.terminal_growth,
        discount_rate=base_inputs.discount_rate,
    )
    result = evaluate_stock(inp, kb, min_margin_of_safety=0.10, squeeze_percentile_threshold=0.25)
    return result.fully_qualified

bt = walk_forward_backtest(
    symbol_dfs=symbol_dfs,
    signal_fn=signal_fn,
    start_date="2023-06-01",
    end_date="2026-06-30",
    holding_period_days=20,
    stop_loss_pct=0.08,
    max_concurrent_positions=3,
)
print(f"Trades: {bt.num_trades}")
print(f"CAGR: {bt.cagr:.1%}")
print(f"Sharpe: {bt.sharpe:.2f}")
print(f"Max Drawdown: {bt.max_drawdown:.1%}")
print(f"Win rate: {bt.win_rate:.1%}  Avg win: {bt.avg_win_pct:.1%}  Avg loss: {bt.avg_loss_pct:.1%}")

print("\nALL LAYERS RAN END-TO-END WITHOUT ERRORS.")
