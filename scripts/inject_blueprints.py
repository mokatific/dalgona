#!/usr/bin/env python3
"""inject_blueprints.py — Blueprint to TOML Config Bridge

Reads blueprint JSONs from data/blueprints/ and generates
[strategy.blueprint-*] TOML sections for perps.toml.

Bridges Stage 2 (Analyze) to Stage 3 (Implement) of the pipeline.

Usage:
    python scripts/inject_blueprints.py [--blueprints-dir data/blueprints] \
        [--config config/perps.toml] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

STRATEGY_TYPE_MAP = {
    "momentum_scalper": "momentum-scalper",
    "momentum": "momentum-scalper",
    "mean_reversion": "mean-reversion",
    "mean-reversion": "mean-reversion",
    "lp_consumer": "lp-consumption",
    "lp-consumer": "lp-consumption",
    "trend_follower": "trend-follower",
    "trend-follower": "trend-follower",
    # No "grid" entry: there is no grid implementation on the Rust side, and
    # silently remapping to momentum-scalper produces a parameter mismatch
    # (grid wallets trade both directions with mixed exits; scalper logic
    # expects directional velocity). Grid clusters are explicitly skipped
    # in main().
    "swing_trader": "trend-follower",
    "swing-trader": "trend-follower",
}

# Strategy types we know we cannot map cleanly. Listed separately from the
# STRATEGY_TYPE_MAP miss-case so main() can emit a clear skip reason.
UNSUPPORTED_STRATEGY_TYPES = {"grid", "hft_market_maker", "hft-market-maker"}

DEFAULTS = {
    "clip_size_usd": 100.0,
    "max_hold_secs": 3600,
    "take_profit_pct": 1.0,
    "stop_loss_pct": 0.5,
    "trailing_stop_pct": 0.0,
    "trailing_activation_pct": 0.0,
    "cooldown_after_loss_secs": 300,
    "leverage": 3.0,
    "scale_in_clips": 1,
}

# Sanity bounds. Blueprints that fail these are rejected — the Rust engine
# will trade real money against these values, so silent garbage-in is worse
# than dropping a cluster.
SANITY = {
    # Reject if TP <= SL * this multiplier. Fees + slippage eat ~0.2-0.3%
    # round-trip on Flash perps, so TP must clear SL by enough margin to
    # have positive expectancy at break-even win rate.
    "tp_sl_ratio_min": 1.2,
    # Hard cap on hold time. Anything above this is almost certainly an
    # outlier or a stuck position rather than an intentional strategy.
    "max_hold_hours_cap": 24.0,
    # Cap on entry threshold (in percent). A 5% velocity entry is already
    # an extreme move; anything above is likely a units error upstream.
    "entry_threshold_pct_cap": 5.0,
    # Floor on TP — sub-fee TPs guarantee bleed even on wins.
    "min_take_profit_pct": 0.15,
}


def sanity_check(bp: dict, cluster_id: str) -> list:
    """Return a list of human-readable rejection reasons, or [] if the
    blueprint is structurally sound.

    These checks defend against pipeline aggregation noise — they do not
    second-guess strategy logic, only refuse to ship blueprints whose
    numeric parameters guarantee negative expectancy or are dimensionally
    impossible (e.g. SL > TP, 50h scalper holds).
    """
    reasons = []

    stype = bp.get("strategy_type", "unknown")
    if stype in UNSUPPORTED_STRATEGY_TYPES:
        reasons.append(f"strategy_type '{stype}' has no Rust implementation")

    exit_c = bp.get("exit_conditions", {})
    tp_raw = exit_c.get("take_profit_pct", 0)
    sl_raw = exit_c.get("stop_loss_pct", 0)
    # Normalize to percent units for the sanity comparison only — the
    # underlying blueprint keeps decimals.
    tp_pct = tp_raw * 100.0 if 0 < tp_raw < 1.0 else tp_raw
    sl_pct = sl_raw * 100.0 if 0 < sl_raw < 1.0 else sl_raw

    if tp_pct <= 0:
        reasons.append(f"take_profit_pct={tp_pct:.4f} is non-positive")
    if sl_pct <= 0:
        reasons.append(f"stop_loss_pct={sl_pct:.4f} is non-positive")
    if tp_pct > 0 and sl_pct > 0 and tp_pct < sl_pct * SANITY["tp_sl_ratio_min"]:
        reasons.append(
            f"TP/SL ratio {tp_pct / sl_pct:.2f} below {SANITY['tp_sl_ratio_min']}x "
            f"(TP={tp_pct:.3f}% vs SL={sl_pct:.3f}%) — negative expectancy after fees"
        )
    if tp_pct > 0 and tp_pct < SANITY["min_take_profit_pct"]:
        reasons.append(
            f"take_profit_pct={tp_pct:.3f}% below min {SANITY['min_take_profit_pct']}% — "
            f"sub-fee target"
        )

    hold_h = exit_c.get("max_hold_hours", 0)
    if hold_h > SANITY["max_hold_hours_cap"]:
        reasons.append(
            f"max_hold_hours={hold_h:.1f} exceeds cap {SANITY['max_hold_hours_cap']}h — "
            f"likely outlier-contaminated"
        )

    ep = bp.get("entry_conditions", {}).get("parameters", {})
    vel = ep.get("price_velocity_threshold", 0)
    if vel > SANITY["entry_threshold_pct_cap"]:
        reasons.append(
            f"price_velocity_threshold={vel:.2f}% exceeds cap "
            f"{SANITY['entry_threshold_pct_cap']}% — likely units error"
        )
    if tp_pct > 0 and vel > tp_pct:
        reasons.append(
            f"entry threshold {vel:.3f}% > take profit {tp_pct:.3f}% — "
            f"strategy requires move larger than its own target"
        )

    return reasons


def blueprint_to_toml(bp: dict, cluster_id: str) -> str:
    """Convert a blueprint JSON to a TOML config section string."""
    stype = bp.get("strategy_type", "unknown")
    rust_strat = STRATEGY_TYPE_MAP.get(stype, "momentum-scalper")
    section = f"blueprint-{cluster_id}"
    entry = bp.get("entry_conditions", {})
    exit_c = bp.get("exit_conditions", {})
    risk = bp.get("risk_parameters", {})
    sample = bp.get("sample_size", {})
    conf = bp.get("confidence_score", 0.0)
    direction = bp.get("direction", "mixed")
    market = bp.get("primary_market", "UNKNOWN")
    trace = bp.get("parameter_traceability", {})

    dir_bias = direction if direction in ("long", "short") else "neutral"
    clip = risk.get("clip_size_usd", DEFAULTS["clip_size_usd"])
    if clip <= 0:
        clip = DEFAULTS["clip_size_usd"]

    tp_raw = exit_c.get("take_profit_pct", 0)
    sl_raw = exit_c.get("stop_loss_pct", 0)
    tp_pct = tp_raw * 100.0 if tp_raw < 1.0 else tp_raw
    sl_pct = sl_raw * 100.0 if sl_raw < 1.0 else sl_raw
    if tp_pct <= 0:
        tp_pct = DEFAULTS["take_profit_pct"]
    if sl_pct <= 0:
        sl_pct = DEFAULTS["stop_loss_pct"]

    hold_h = exit_c.get("max_hold_hours", 0)
    hold_s = int(hold_h * 3600) if hold_h > 0 else DEFAULTS["max_hold_secs"]

    trailing = exit_c.get("trailing_stop", False)
    t_stop = tp_pct * 0.4 if trailing and rust_strat == "trend-follower" else 0.0
    t_act = tp_pct * 0.6 if trailing and rust_strat == "trend-follower" else 0.0

    ep = entry.get("parameters", {})
    vel = ep.get("price_velocity_threshold", 0.15)
    lookback = entry.get("lookback_candles", 6) * 5
    nw = sample.get("wallets", 0)
    nt = sample.get("total_trades", 0)

    L = []
    L.append("")
    L.append(f"# Data-Driven Strategy from {cluster_id}")
    L.append(f"# Source: data/blueprints/{cluster_id}.json")
    L.append(f"# {nw} wallets, {nt} trades, confidence {conf:.2f}")
    L.append(f"# Market: {market}, Direction: {direction}, Type: {stype} -> {rust_strat}")
    L.append(f"[strategy.{section}]")

    if rust_strat == "momentum-scalper":
        L.append(f"momentum_threshold_pct = {vel}")
        L.append(f"lookback_count = {lookback}")
    elif rust_strat == "mean-reversion":
        L.append(f"mean_lookback = {lookback}")
        L.append(f"deviation_threshold_pct = {vel}")
        L.append(f"reversal_confirmation_ticks = 2")
        L.append(f"mean_tolerance_pct = 0.3")
    elif rust_strat == "lp-consumption":
        L.append(f"consumption_velocity_threshold = {max(vel, 0.5)}")
        L.append(f"lp_concentration_min = 0.7")
        L.append(f"confirmation_ticks = 3")
        L.append(f"max_utilization = 0.9")
    elif rust_strat == "trend-follower":
        L.append(f"breakout_threshold_pct = {max(vel, 0.25)}")
        L.append(f"confirmation_ticks = 4")
        L.append(f"min_price_count = 30")

    L.append(f"clip_size_usd = {clip}")
    L.append(f"take_profit_pct = {tp_pct:.4f}")
    L.append(f"stop_loss_pct = {sl_pct:.4f}")
    L.append(f"max_hold_secs = {hold_s}")
    L.append(f"trailing_stop_pct = {t_stop:.4f}")
    L.append(f"trailing_activation_pct = {t_act:.4f}")
    L.append(f'direction_bias = "{dir_bias}"')
    L.append(f"scale_in_clips = {DEFAULTS['scale_in_clips']}")
    L.append(f"cooldown_after_loss_secs = {DEFAULTS['cooldown_after_loss_secs']}")
    L.append(f"use_native_tp_sl = true")
    L.append(f"leverage = {DEFAULTS['leverage']}")
    return "\n".join(L) + "\n"


def load_blueprints(bdir: str):
    bps = []
    bp = Path(bdir)
    if not bp.exists():
        print(f"Error: {bdir} not found")
        sys.exit(1)
    for f in sorted(bp.glob("cluster-*.json")):
        try:
            with open(f) as fp:
                bps.append((f.stem, json.load(fp)))
        except Exception as e:
            print(f"Warning: {f}: {e}")
    return bps


def inject_into_config(cfg_path: str, sections: list, dry_run=False):
    with open(cfg_path) as f:
        txt = f.read()

    # Remove existing auto-generated sections
    txt = re.sub(
        r'\n# =+\s*\n#\s*AUTO-GENERATED.*?(?=\n\[risk\]|\Z)',
        '', txt, flags=re.DOTALL
    ).rstrip() + "\n"

    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    block = "\n"
    block += "# " + "=" * 77 + "\n"
    block += f"# AUTO-GENERATED BLUEPRINT STRATEGIES ({now})\n"
    block += "# Data lineage: fills -> clusters -> medians -> blueprint -> config\n"
    block += "# DO NOT EDIT MANUALLY — re-run inject_blueprints.py to regenerate\n"
    block += "# " + "=" * 77 + "\n"
    for s in sections:
        block += s

    m = re.search(r'^\[risk\]', txt, re.MULTILINE)
    if m:
        txt = txt[:m.start()] + block + "\n" + txt[m.start():]
    else:
        txt += block

    if not dry_run:
        tmp = cfg_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(txt)
        os.replace(tmp, cfg_path)
        print(f"Config updated: {cfg_path}")
    return txt


def main():
    ap = argparse.ArgumentParser(description="Inject blueprints into config")
    ap.add_argument("--blueprints-dir", default="data/blueprints")
    ap.add_argument("--config", default="config/perps.toml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--min-trades", type=int, default=100)
    args = ap.parse_args()

    print("=== Blueprint -> Config Injection ===")
    bps = load_blueprints(args.blueprints_dir)
    if not bps:
        print("No blueprints found. Run analysis first.")
        sys.exit(1)

    print(f"Found {len(bps)} blueprints")
    qualified = []
    rejected = []
    for cid, bp in bps:
        c = bp.get("confidence_score", 0)
        t = bp.get("sample_size", {}).get("total_trades", 0)
        st = bp.get("strategy_type", "unknown")
        if c < args.min_confidence:
            print(f"  SKIP {cid}: confidence {c:.2f} < {args.min_confidence}")
            continue
        if t < args.min_trades:
            print(f"  SKIP {cid}: {t} trades < {args.min_trades}")
            continue
        if st == "unknown":
            print(f"  SKIP {cid}: unknown strategy type")
            continue
        if st not in STRATEGY_TYPE_MAP:
            print(f"  SKIP {cid}: strategy_type '{st}' has no Rust mapping")
            continue
        sanity_reasons = sanity_check(bp, cid)
        if sanity_reasons:
            print(f"  REJECT {cid}: {st}, {t} trades, conf={c:.2f}")
            for r in sanity_reasons:
                print(f"         ↳ {r}")
            rejected.append((cid, sanity_reasons))
            continue
        print(f"  OK   {cid}: {st}, {t} trades, conf={c:.2f}")
        qualified.append((cid, bp))

    if rejected:
        print(f"\nRejected {len(rejected)} blueprint(s) on sanity checks.")
        print("Tune blueprint_generator.py thresholds or fix source data before retry.")

    if not qualified:
        print("No blueprints passed quality gates.")
        sys.exit(1)

    sections = []
    names = []
    for cid, bp in qualified:
        sections.append(blueprint_to_toml(bp, cid))
        names.append(f"blueprint-{cid}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for s in sections:
            print(s)
    else:
        inject_into_config(args.config, sections)

    print(f"\nStrategies: {', '.join(names)}")
    print(f"Next: cargo run -- --backtest --strategies {','.join(names)} --markets BTC,SOL,ETH")
    return names


if __name__ == "__main__":
    main()
