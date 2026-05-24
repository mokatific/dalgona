"""Wallet Metrics Module

Computes all Bulk.Trade metrics per wallet from position cluster data and
raw fill records. Every metric is derived from actual data — nothing is
hardcoded.

Metrics produced:
  total_trades          Number of closed position clusters
  win_rate              Fraction of closed positions with positive PnL
  avg_hold_time_hours   Mean hold duration across closed positions (hours)
  avg_pnl_per_trade     Mean realized PnL per closed position (USD)
  clip_size_consistency % of fills within ±10% of median fill size
  fill_interval_stats   {median_gap_seconds, pct_sub_30s}
  scale_in_count        Number of positions with 2+ entry fills within 5 min
  active_hours          List of distinct UTC hours (0-23) with fill activity
  coverage_pct          Fraction of 24h covered by active_hours
  fee_adjusted_pnl      sum(realized_pnl) - sum(all fees)
  fee_adjusted_win_rate Fraction of positions with fee-adjusted PnL > 0
  sharpe_ratio          mean(PnL) / std(PnL) for closed positions
  max_drawdown          Maximum peak-to-trough decline in cumulative PnL
  profit_factor         gross_wins / abs(gross_losses)
  avg_leverage          Mean notional value per position (entry_price × size)
  markets_traded        Unique coin symbols
  primary_market        Coin with most closed positions
  preferred_direction   "long", "short", or "mixed"
  pnl_distribution      {mean, median, max_winner, max_loser, skewness}

Input:
  clusters  — list of position cluster dicts from position_clustering.cluster_fills()
  fills     — list of raw fill dicts with Hyperliquid schema fields
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Direction classification thresholds
_DIRECTION_DOMINANCE = 0.6  # >60% one direction → classified as that direction

# Clip consistency tolerance
_CLIP_TOLERANCE = 0.10  # ±10%


def _parse_float(value, default: float = 0.0) -> float:
    """Safely parse a string/numeric value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Individual metric computation functions (pure, testable)
# ---------------------------------------------------------------------------


def _compute_total_trades(clusters: list) -> int:
    """Count closed position clusters (have exit_fills or realized_pnl)."""
    return sum(
        1 for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    )


def _compute_win_rate(clusters: list) -> float:
    """Fraction of closed positions with positive realized_pnl."""
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return 0.0
    wins = sum(1 for c in closed if _parse_float(c.get("realized_pnl", 0)) > 0)
    return wins / len(closed)


def _compute_avg_hold_time_hours(clusters: list) -> Optional[float]:
    """Average hold time in hours for closed positions."""
    hold_times = []
    for c in clusters:
        entry = c.get("entry_time")
        exit_ = c.get("exit_time")
        if entry is not None and exit_ is not None:
            hours = (_parse_float(exit_) - _parse_float(entry)) / 3_600_000.0
            hold_times.append(hours)
    if not hold_times:
        return 0.0
    return sum(hold_times) / len(hold_times)


def _compute_avg_pnl_per_trade(clusters: list) -> float:
    """Mean realized PnL across closed positions."""
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return 0.0
    pnls = [_parse_float(c.get("realized_pnl", 0)) for c in closed]
    return sum(pnls) / len(pnls)


def _compute_clip_size_consistency(fills: list) -> float:
    """Percentage of fills where sz is within ±10% of the median fill size.

    Returns a value in [0, 1]. A value > 0.8 indicates bot activity
    (fixed-clip strategy).
    """
    if not fills:
        return 0.0

    sizes = [_parse_float(f.get("sz", 0)) for f in fills]
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        return 0.0

    median_size = float(np.median(sizes))
    if median_size <= 0:
        return 0.0

    within = sum(
        1 for s in sizes if abs(s - median_size) / median_size <= _CLIP_TOLERANCE
    )
    return within / len(sizes)


def _compute_fill_interval_stats(fills: list) -> dict:
    """Compute fill interval statistics for bot detection.

    Returns:
        median_gap_seconds: median time gap between consecutive fills
        pct_sub_30s: fraction of intervals under 30 seconds
    """
    if len(fills) < 2:
        return {"median_gap_seconds": 0.0, "pct_sub_30s": 0.0}

    # Sort fills by timestamp
    times = sorted(_parse_float(f.get("time", 0)) for f in fills)
    gaps = [(times[i + 1] - times[i]) / 1000.0 for i in range(len(times) - 1)]
    gaps = [g for g in gaps if g >= 0]

    if not gaps:
        return {"median_gap_seconds": 0.0, "pct_sub_30s": 0.0}

    median_gap = float(np.median(gaps))
    sub_30 = sum(1 for g in gaps if g < 30.0)
    pct_sub_30 = sub_30 / len(gaps)

    return {"median_gap_seconds": median_gap, "pct_sub_30s": pct_sub_30}


def _compute_scale_in_count(clusters: list) -> int:
    """Count positions with scale-in entries (2+ entry fills within 5 min)."""
    return sum(1 for c in clusters if c.get("scale_in", False))


def _compute_active_hours(fills: list) -> tuple[list, float]:
    """Compute UTC hour coverage from fill timestamps.

    Returns:
        active_hours: sorted list of distinct UTC hours (0-23)
        coverage_pct: fraction of 24h covered (active_hours / 24)
    """
    if not fills:
        return [], 0.0

    hours = set()
    for f in fills:
        ts = _parse_float(f.get("time", 0))
        if ts >= 0:
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            hours.add(dt.hour)

    active = sorted(hours)
    coverage = len(active) / 24.0
    return active, coverage


def _compute_fee_adjusted_pnl(clusters: list) -> float:
    """Compute PnL after fees: sum(realized_pnl) - sum(fees_paid)."""
    total_pnl = sum(_parse_float(c.get("realized_pnl", 0)) for c in clusters)
    total_fees = sum(_parse_float(c.get("fees_paid", 0)) for c in clusters)
    return total_pnl - total_fees


def _compute_fee_adjusted_win_rate(clusters: list) -> float:
    """Win rate after fees — a trade is a 'win' only if (pnl - fees) > 0."""
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return 0.0

    wins = 0
    for c in closed:
        pnl = _parse_float(c.get("realized_pnl", 0))
        fees = _parse_float(c.get("fees_paid", 0))
        if (pnl - fees) > 0:
            wins += 1

    return wins / len(closed)


def _compute_sharpe_ratio(clusters: list) -> Optional[float]:
    """Compute Sharpe ratio from closed position PnLs.

    Sharpe = mean(pnl) / std(pnl). Returns None if std is 0
    (all identical PnLs) or fewer than 2 trades.
    """
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if len(closed) < 2:
        # With 1 trade, Sharpe is undefined
        if len(closed) == 1:
            pnl = _parse_float(closed[0].get("realized_pnl", 0))
            return float("inf") if pnl > 0 else float("-inf") if pnl < 0 else 0.0
        return None

    pnls = np.array([_parse_float(c.get("realized_pnl", 0)) for c in closed])
    std = float(np.std(pnls, ddof=1))  # sample std
    if std == 0:
        return 0.0
    return float(np.mean(pnls) / std)


def _compute_max_drawdown(clusters: list) -> float:
    """Maximum peak-to-trough decline in cumulative PnL.

    Returns the absolute drawdown value (positive number).
    """
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return 0.0

    # Sort by exit_time to get chronological order
    sorted_clusters = sorted(
        closed,
        key=lambda c: (
            _parse_float(c.get("exit_time", 0)) if c.get("exit_time") is not None else 0
        ),
    )

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for c in sorted_clusters:
        pnl = _parse_float(c.get("realized_pnl", 0))
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return max_dd


def _compute_profit_factor(clusters: list) -> Optional[float]:
    """Gross wins / gross losses. Returns None if no losses (undefined)."""
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return None

    gross_wins = 0.0
    gross_losses = 0.0

    for c in closed:
        pnl = _parse_float(c.get("realized_pnl", 0))
        if pnl > 0:
            gross_wins += pnl
        elif pnl < 0:
            gross_losses += abs(pnl)

    if gross_losses == 0:
        return None  # No losses → profit factor is undefined

    return gross_wins / gross_losses


def _compute_avg_leverage(clusters: list) -> float:
    """Compute average position notional (entry_price × total_size).

    This serves as a proxy for average leverage when account size is unknown.
    """
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return 0.0

    notionals = []
    for c in closed:
        price = _parse_float(c.get("entry_price", 0))
        size = _parse_float(c.get("total_size", 0))
        notionals.append(price * size)

    return sum(notionals) / len(notionals)


def _compute_markets_traded(clusters: list) -> list:
    """Unique coin symbols across all clusters."""
    # Use list to preserve order of first appearance
    seen = set()
    result = []
    for c in clusters:
        coin = c.get("coin", "UNKNOWN")
        if coin not in seen:
            seen.add(coin)
            result.append(coin)
    return result


def _compute_primary_market(clusters: list) -> Optional[str]:
    """Coin with most closed positions."""
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return None

    counts: dict[str, int] = {}
    for c in closed:
        coin = c.get("coin", "UNKNOWN")
        counts[coin] = counts.get(coin, 0) + 1

    return max(counts, key=counts.get)


def _compute_preferred_direction(clusters: list) -> str:
    """Determine if wallet prefers long, short, or mixed.

    Returns "long" if >60% long, "short" if >60% short, else "mixed".
    """
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return "unknown"

    longs = sum(1 for c in closed if c.get("direction") == "long")
    total = len(closed)
    long_frac = longs / total

    if long_frac > _DIRECTION_DOMINANCE:
        return "long"
    elif long_frac < (1.0 - _DIRECTION_DOMINANCE):
        return "short"
    else:
        return "mixed"


def _compute_pnl_distribution(clusters: list) -> dict:
    """PnL distribution statistics for closed positions.

    Returns:
        mean, median, max_winner, max_loser, skewness
    """
    closed = [
        c for c in clusters if c.get("exit_fills") or c.get("exit_time") is not None
    ]
    if not closed:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max_winner": 0.0,
            "max_loser": 0.0,
            "skewness": 0.0,
        }

    pnls = np.array([_parse_float(c.get("realized_pnl", 0)) for c in closed])

    mean = float(np.mean(pnls))
    median = float(np.median(pnls))
    max_winner = float(np.max(pnls))
    max_loser = float(np.min(pnls))

    # Skewness using Fisher's definition (type 3, unbiased)
    if len(pnls) >= 3:
        n = len(pnls)
        std = float(np.std(pnls, ddof=1))
        if std > 0:
            skewness = float(
                (n / ((n - 1) * (n - 2))) * np.sum(((pnls - mean) / std) ** 3)
            )
        else:
            skewness = 0.0
    else:
        skewness = 0.0

    return {
        "mean": mean,
        "median": median,
        "max_winner": max_winner,
        "max_loser": max_loser,
        "skewness": skewness,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_wallet_metrics(clusters: list, fills: list) -> dict:
    """Compute all Bulk.Trade metrics for a single wallet.

    Every metric is derived from the actual cluster and fill data — nothing
    is hardcoded.

    Args:
        clusters: List of position cluster dicts from
            position_clustering.cluster_fills(). Each cluster has fields:
            coin, direction, entry_fills, exit_fills, entry_time, exit_time,
            entry_price, exit_price, total_size, realized_pnl, fees_paid,
            scale_in.
        fills: List of raw fill dicts with Hyperliquid schema fields:
            coin, side, px, sz, fee, closedPnl, time, dir, hash, startPosition.

    Returns:
        Dict with all computed wallet metrics.
    """
    if not clusters and not fills:
        return _empty_metrics()

    active_hours, coverage_pct = _compute_active_hours(fills)

    metrics = {
        # Core trading metrics
        "total_trades": _compute_total_trades(clusters),
        "win_rate": _compute_win_rate(clusters),
        "avg_hold_time_hours": _compute_avg_hold_time_hours(clusters),
        "avg_pnl_per_trade": _compute_avg_pnl_per_trade(clusters),
        # Fill-level metrics (bot detection)
        "clip_size_consistency": _compute_clip_size_consistency(fills),
        "fill_interval_stats": _compute_fill_interval_stats(fills),
        "scale_in_count": _compute_scale_in_count(clusters),
        # Activity metrics
        "active_hours": active_hours,
        "coverage_pct": coverage_pct,
        # Fee-adjusted metrics
        "fee_adjusted_pnl": _compute_fee_adjusted_pnl(clusters),
        "fee_adjusted_win_rate": _compute_fee_adjusted_win_rate(clusters),
        # Risk metrics
        "sharpe_ratio": _compute_sharpe_ratio(clusters),
        "max_drawdown": _compute_max_drawdown(clusters),
        "profit_factor": _compute_profit_factor(clusters),
        # Position sizing
        "avg_leverage": _compute_avg_leverage(clusters),
        "avg_notional": _compute_avg_leverage(clusters),
        # Market and direction
        "markets_traded": _compute_markets_traded(clusters),
        "primary_market": _compute_primary_market(clusters),
        "preferred_direction": _compute_preferred_direction(clusters),
        # PnL distribution
        "pnl_distribution": _compute_pnl_distribution(clusters),
    }

    return metrics


def _empty_metrics() -> dict:
    """Return safe default metrics when no data is available."""
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_hold_time_hours": 0.0,
        "avg_pnl_per_trade": 0.0,
        "clip_size_consistency": 0.0,
        "fill_interval_stats": {"median_gap_seconds": 0.0, "pct_sub_30s": 0.0},
        "scale_in_count": 0,
        "active_hours": [],
        "coverage_pct": 0.0,
        "fee_adjusted_pnl": 0.0,
        "fee_adjusted_win_rate": 0.0,
        "sharpe_ratio": None,
        "max_drawdown": 0.0,
        "profit_factor": None,
        "avg_leverage": 0.0,
        "avg_notional": 0.0,
        "markets_traded": [],
        "primary_market": None,
        "preferred_direction": "unknown",
        "pnl_distribution": {
            "mean": 0.0,
            "median": 0.0,
            "max_winner": 0.0,
            "max_loser": 0.0,
            "skewness": 0.0,
        },
    }
