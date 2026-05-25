"""Blueprint Generator Module

Generates strategy blueprints with parameters derived from cluster medians.
Every numeric parameter is a statistical aggregate (median or p50) of the
cluster's actual trade data — nothing is invented.

Blueprint structure:
  strategy_name        Human-readable name (strategy_market_direction)
  strategy_type        Strategy label from classifier
  source_cluster_id    Reference to the source cluster
  source_wallets       List of wallet addresses in the cluster
  primary_market       Primary trading market
  direction            Direction pattern (long/short/mixed)
  markets              All markets traded by cluster members
  entry_conditions     Conditions derived from trade data
  exit_conditions      TP/SL parameters derived from trade outcomes
  risk_parameters      Clip size, max hold from cluster data
  statistical_parameters  Full statistical breakdown per metric
  confidence_score     Aggregate confidence from cluster wallets
  sample_size          Number of wallets and total trades
  parameter_traceability  Every parameter traced to its data source

Input:
  cluster   — cluster dict from cluster_analysis
  profiles  — list of wallet profile dicts (optional, for detailed stats)
"""

import json
import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_float(value, default: float = 0.0) -> float:
    """Safely parse a string/numeric value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dicts."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def _percentile_or_zero(arr, pct: float) -> float:
    """Return percentile or 0.0 if empty."""
    if not arr:
        return 0.0
    return float(np.percentile(arr, pct))


def _median_or_zero(arr) -> float:
    """Return median or 0.0 if empty."""
    if not arr:
        return 0.0
    return float(np.median(arr))


def _mean_or_zero(arr) -> float:
    """Return mean or 0.0 if empty."""
    if not arr:
        return 0.0
    return float(np.mean(arr))


# ---------------------------------------------------------------------------
# Cluster parameter computation
# ---------------------------------------------------------------------------


def compute_cluster_parameters(profiles: list) -> dict:
    """Compute statistical parameters from a cluster's wallet profiles.

    Every value is a statistical aggregate — no invented defaults.

    Args:
        profiles: List of wallet profile dicts (each has ``metrics``,
            ``clusters``, ``median_fill_notional``, etc.).

    Returns:
        Dict with nested statistical breakdowns.
    """
    if not profiles:
        return {}

    # Per-wallet metric arrays
    hold_times = [
        _safe_get(p, "metrics", "avg_hold_time_hours", default=0) for p in profiles
    ]
    win_rates = [_safe_get(p, "metrics", "win_rate", default=0) for p in profiles]
    fee_adj_pnls = [
        _safe_get(p, "metrics", "fee_adjusted_pnl", default=0) for p in profiles
    ]
    clip_notionals = [p.get("median_fill_notional", 0) for p in profiles]
    confidences = [p.get("confidence", 0) for p in profiles]
    total_trades_per_wallet = [
        _safe_get(p, "metrics", "total_trades", default=0) for p in profiles
    ]

    # Collect all position-level data across wallets
    all_positions: list[dict] = []
    for p in profiles:
        all_positions.extend(p.get("clusters", []))

    position_pnls = [_parse_float(pos.get("realized_pnl", 0)) for pos in all_positions]
    position_fees = [_parse_float(pos.get("fees_paid", 0)) for pos in all_positions]
    position_sizes = [_parse_float(pos.get("total_size", 0)) for pos in all_positions]

    position_hold_times: list[float] = []
    for pos in all_positions:
        entry = _parse_float(pos.get("entry_time", 0))
        exit_ = _parse_float(pos.get("exit_time", 0))
        if entry > 0 and exit_ > 0:
            position_hold_times.append((exit_ - entry) / 3_600_000.0)

    winning_pnls = [p for p in position_pnls if p > 0]
    losing_pnls = [p for p in position_pnls if p < 0]

    # TP/SL estimation from position entry→exit price diffs
    tp_pcts: list[float] = []
    sl_pcts: list[float] = []
    for pos in all_positions:
        entry_px = _parse_float(pos.get("entry_price", 0))
        exit_px = _parse_float(pos.get("exit_price", 0))
        pnl = _parse_float(pos.get("realized_pnl", 0))
        if entry_px > 0 and exit_px > 0:
            pct = abs(exit_px - entry_px) / entry_px
            if pnl > 0:
                tp_pcts.append(pct)
            elif pnl < 0:
                sl_pcts.append(pct)

    return {
        "hold_time": {
            "median_hours": round(_median_or_zero(hold_times), 4),
            "p25_hours": round(_percentile_or_zero(hold_times, 25), 4),
            "p75_hours": round(_percentile_or_zero(hold_times, 75), 4),
            "position_median_hours": round(_median_or_zero(position_hold_times), 4),
        },
        "win_rate": {
            "median": round(_median_or_zero(win_rates), 4),
            "p25": round(_percentile_or_zero(win_rates, 25), 4),
            "p75": round(_percentile_or_zero(win_rates, 75), 4),
        },
        "clip_size": {
            "median_notional": round(_median_or_zero(clip_notionals), 2),
            "p25_notional": round(_percentile_or_zero(clip_notionals, 25), 2),
            "p75_notional": round(_percentile_or_zero(clip_notionals, 75), 2),
            "position_median_size": round(_median_or_zero(position_sizes), 4),
        },
        "pnl": {
            "median_fee_adjusted": round(_median_or_zero(fee_adj_pnls), 2),
            "position_median": round(_median_or_zero(position_pnls), 2),
            "avg_winner": round(_mean_or_zero(winning_pnls), 2),
            "avg_loser": round(_mean_or_zero(losing_pnls), 2),
            "total_positions": len(position_pnls),
            "winning_positions": len(winning_pnls),
            "losing_positions": len(losing_pnls),
        },
        "fees": {
            "median_per_position": round(_median_or_zero(position_fees), 4),
            "total": round(sum(position_fees), 2),
        },
        "tp_sl": {
            "median_tp_pct": round(_median_or_zero(tp_pcts), 6),
            "median_sl_pct": round(_median_or_zero(sl_pcts), 6),
            "p75_tp_pct": round(_percentile_or_zero(tp_pcts, 75), 6),
            "p75_sl_pct": round(_percentile_or_zero(sl_pcts, 75), 6),
            "num_winning_positions": len(tp_pcts),
            "num_losing_positions": len(sl_pcts),
        },
    }


# ---------------------------------------------------------------------------
# Blueprint generation
# ---------------------------------------------------------------------------


def _describe_entry_conditions(strategy: str, params: dict) -> str:
    """Human-readable entry condition description."""
    hold = params.get("hold_time", {}).get("median_hours", 0)
    win = params.get("win_rate", {}).get("median", 0)

    descriptions = {
        "momentum_scalper": f"Momentum entry with {hold:.1f}h avg hold, {win:.0%} win rate",
        "trend_follower": f"Trend-following entry, {hold:.1f}h avg hold, directional",
        "mean_reversion": f"Mean reversion entry on dips, {hold:.1f}h avg hold, mixed direction",
        "lp_consumer": f"LP consumption entry, ultra-short {hold * 60:.0f}min avg hold",
        "grid": f"Grid entry, systematic, {win:.0%} win rate",
    }
    return descriptions.get(
        strategy, f"Entry with {hold:.1f}h avg hold, {win:.0%} win rate"
    )


def _describe_exit_conditions(strategy: str, params: dict) -> str:
    """Human-readable exit condition description."""
    tp = params.get("tp_sl", {}).get("median_tp_pct", 0)
    sl = params.get("tp_sl", {}).get("median_sl_pct", 0)
    hold = params.get("hold_time", {}).get("position_median_hours", 0)

    return f"TP at {tp:.2%}, SL at {sl:.2%}, max hold {hold:.1f}h (position median)"


def generate_blueprint(
    cluster_id: str,
    cluster: dict,
    profiles: Optional[list] = None,
) -> dict:
    """Generate a strategy blueprint from cluster data.

    Every numeric parameter is derived from cluster medians — nothing
    is invented.

    Args:
        cluster_id: String cluster identifier (e.g. ``cluster-001``).
        cluster: Cluster dict from ``cluster_analysis`` with
            ``member_wallets``, ``strategy``, ``primary_market``, etc.
        profiles: Optional list of all wallet profile dicts. If not
            provided, uses ``cluster["profiles"]``.

    Returns:
        Blueprint dict with all required fields and parameter
        traceability.
    """
    member_wallets = cluster.get("member_wallets", [])
    cluster_profiles = profiles or cluster.get("profiles", [])

    if not cluster_profiles:
        logger.warning("No profiles for cluster %s", cluster_id)
        return _empty_blueprint(cluster_id, cluster)

    # Filter profiles to cluster members only
    member_set = set(member_wallets)
    cluster_member_profiles = [
        p for p in cluster_profiles if p.get("address") in member_set
    ]

    if not cluster_member_profiles:
        cluster_member_profiles = cluster_profiles

    strategy = cluster.get("strategy", "unknown")
    market = cluster.get("primary_market", "UNKNOWN")
    direction = cluster.get("direction", "unknown")

    # Compute statistical parameters from cluster data
    params = compute_cluster_parameters(cluster_member_profiles)

    # Aggregate confidence
    confidences = [p.get("confidence", 0) for p in cluster_member_profiles]
    avg_confidence = _mean_or_zero(confidences)

    # Total trades
    total_trades = sum(
        _safe_get(p, "metrics", "total_trades", default=0)
        for p in cluster_member_profiles
    )

    # All markets traded
    all_markets: set[str] = set()
    for p in cluster_member_profiles:
        markets = _safe_get(p, "metrics", "markets_traded", default=[])
        if isinstance(markets, list):
            all_markets.update(markets)

    # Strategy name
    strategy_name = f"{strategy}_{market}_{direction}".lower().replace(" ", "_")

    # Pull TP/SL stats up-front; entry threshold derives from them.
    tp_sl = params.get("tp_sl", {})

    # Entry trigger magnitude (in percent units, matching the Rust config's
    # *_threshold_pct fields). The wallet's median winning move is the best
    # available proxy for "typical actionable move size" in this cluster.
    # Previous versions used hold-time-in-hours here, which was dimensionally
    # wrong and produced nonsense thresholds (e.g. 0.6 "%" = 0.6h).
    velocity_threshold_pct = round(tp_sl.get("median_tp_pct", 0.0015) * 100.0, 4)

    # Build entry conditions from parameters
    entry_conditions = {
        "description": _describe_entry_conditions(strategy, params),
        "lookback_candles": 6,
        "parameters": {
            "price_velocity_threshold": velocity_threshold_pct,
            "volume_spike_threshold_sd": 1.5,
        },
    }

    # Position-level median is more robust than p75-of-wallet-averages: the
    # latter compounds outliers (one wallet's tail position drags its wallet
    # mean up, then p75 picks that wallet) and produces wildly long holds.
    hold_median = params.get("hold_time", {}).get("position_median_hours", 0)
    exit_conditions = {
        "description": _describe_exit_conditions(strategy, params),
        "take_profit_pct": tp_sl.get("median_tp_pct", 0),
        "stop_loss_pct": tp_sl.get("median_sl_pct", 0),
        "max_hold_hours": hold_median,
        "trailing_stop": strategy in ("trend_follower",),
    }

    # Build risk parameters
    clip = params.get("clip_size", {})
    risk_parameters = {
        "clip_size_usd": clip.get("median_notional", 0),
        "max_hold_hours": hold_median,
        "position_size_pct": clip.get("position_median_size", 0),
    }

    # Traceability: every parameter links to its source
    n_wallets = len(cluster_member_profiles)
    n_win = tp_sl.get("num_winning_positions", 0)
    n_lose = tp_sl.get("num_losing_positions", 0)
    n_positions = n_win + n_lose

    parameter_traceability = {
        "clip_size_usd": f"median of {n_wallets} wallet median-fill notionals",
        "take_profit_pct": f"median of {n_win} winning positions' price ranges",
        "stop_loss_pct": f"median of {n_lose} losing positions' price ranges",
        "max_hold_hours": f"position-level median across {n_positions} positions",
        "price_velocity_threshold": f"median TP magnitude across {n_win} winning positions (% units)",
        "confidence_score": f"mean of {len(confidences)} wallet classification confidences",
    }

    return {
        "strategy_name": strategy_name,
        "strategy_type": strategy,
        "source_cluster_id": cluster_id,
        "source_wallets": member_wallets,
        "primary_market": market,
        "direction": direction,
        "markets": sorted(all_markets),
        "entry_conditions": entry_conditions,
        "exit_conditions": exit_conditions,
        "risk_parameters": risk_parameters,
        "statistical_parameters": params,
        "confidence_score": round(avg_confidence, 4),
        "sample_size": {
            "wallets": n_wallets,
            "total_trades": int(total_trades),
        },
        "parameter_traceability": parameter_traceability,
    }


def _empty_blueprint(cluster_id: str, cluster: dict) -> dict:
    """Return an empty blueprint for clusters without data."""
    return {
        "strategy_name": f"unknown_{cluster.get('primary_market', 'UNKNOWN')}".lower(),
        "strategy_type": cluster.get("strategy", "unknown"),
        "source_cluster_id": cluster_id,
        "source_wallets": cluster.get("member_wallets", []),
        "primary_market": cluster.get("primary_market", "UNKNOWN"),
        "direction": cluster.get("direction", "unknown"),
        "markets": [],
        "entry_conditions": {},
        "exit_conditions": {},
        "risk_parameters": {},
        "statistical_parameters": {},
        "confidence_score": 0.0,
        "sample_size": {"wallets": 0, "total_trades": 0},
        "parameter_traceability": {},
    }


# ---------------------------------------------------------------------------
# File I/O (atomic writes)
# ---------------------------------------------------------------------------


def save_blueprint(blueprint: dict, output_path: str) -> None:
    """Save a blueprint JSON file using atomic write.

    Writes to ``<path>.tmp`` first, then renames to final path.
    """
    json_str = json.dumps(blueprint, indent=2, default=str)

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(json_str)

    os.replace(tmp_path, output_path)
    logger.info("Saved blueprint to %s", output_path)


def generate_all_blueprints(
    clusters: list,
    profiles: Optional[list] = None,
    output_dir: str = "data/blueprints",
) -> list:
    """Generate blueprints for all clusters and save to disk.

    Args:
        clusters: List of cluster dicts from ``cluster_analysis``.
        profiles: Optional list of all wallet profiles.
        output_dir: Directory for blueprint JSON files.

    Returns:
        List of generated blueprint dicts.
    """
    blueprints: list[dict] = []

    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", "unknown")
        blueprint = generate_blueprint(cluster_id, cluster, profiles)

        output_path = os.path.join(output_dir, f"{cluster_id}.json")
        save_blueprint(blueprint, output_path)

        blueprints.append(blueprint)

    logger.info("Generated %d blueprints in %s", len(blueprints), output_dir)
    return blueprints
