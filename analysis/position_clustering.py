"""Position Clustering Module

Clusters individual Hyperliquid fills into open→close position cycles.
Uses the `startPosition` field changes and `dir` field (Open Long, Close Long,
Open Short, Close Short) to determine position boundaries.

Handles:
  - Partial closes (multiple exit fills reducing a position)
  - Scale-ins (multiple entry fills within 5 minutes)
  - Direction reversals (close long → open short)
  - Orphaned fills (close without open, open without close)
  - Multiple markets (fills grouped by coin)
  - Edge cases (<10 trades, empty data)

Output: structured position clusters with entry/exit times, VWAP prices,
realized PnL, and fees.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Scale-in window: fills within this many milliseconds are considered scale-ins
SCALE_IN_WINDOW_MS = 5 * 60 * 1000  # 5 minutes


@dataclass
class PositionCluster:
    """Represents a single open→close position cycle.

    Attributes:
        coin: Market symbol (e.g., "BTC", "ETH").
        direction: "long" or "short".
        entry_fills: List of fill dicts that opened / scaled into the position.
        exit_fills: List of fill dicts that closed the position.
        entry_time: Timestamp (ms) of the first entry fill.
        exit_time: Timestamp (ms) of the last exit fill, or None if still open.
        entry_price: VWAP of entry fills (size-weighted).
        exit_price: VWAP of exit fills (size-weighted), or None if still open.
        total_size: Total position size from entry fills.
        realized_pnl: Sum of closedPnl from exit fills.
        fees_paid: Sum of all fees (entry + exit).
        scale_in: True if position has 2+ entry fills within SCALE_IN_WINDOW_MS.
    """

    coin: str
    direction: str
    entry_fills: list = field(default_factory=list)
    exit_fills: list = field(default_factory=list)
    entry_time: Optional[int] = None
    exit_time: Optional[int] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    total_size: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    scale_in: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "coin": self.coin,
            "direction": self.direction,
            "entry_fills": self.entry_fills,
            "exit_fills": self.exit_fills,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "total_size": self.total_size,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "scale_in": self.scale_in,
        }


def _parse_float(value, default: float = 0.0) -> float:
    """Safely parse a string/numeric value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_vwap(fills: list) -> Optional[float]:
    """Compute volume-weighted average price from a list of fill dicts.

    Returns None if fills list is empty or total size is zero.
    """
    if not fills:
        return None
    total_notional = 0.0
    total_size = 0.0
    for f in fills:
        px = _parse_float(f.get("px", 0))
        sz = _parse_float(f.get("sz", 0))
        total_notional += px * sz
        total_size += sz
    if total_size == 0.0:
        return None
    return total_notional / total_size


def _sum_fees(fills: list) -> float:
    """Sum fee values across fills."""
    return sum(_parse_float(f.get("fee", 0)) for f in fills)


def _sum_realized_pnl(fills: list) -> float:
    """Sum closedPnl across fills."""
    return sum(_parse_float(f.get("closedPnl", 0)) for f in fills)


def _direction_from_dir(dir_str: str) -> str:
    """Extract 'long' or 'short' from a dir string like 'Open Long'."""
    dir_lower = dir_str.lower()
    if "long" in dir_lower:
        return "long"
    elif "short" in dir_lower:
        return "short"
    return "unknown"


def _is_open(dir_str: str) -> bool:
    """Return True if the dir indicates an opening fill."""
    return dir_str.lower().startswith("open")


def _is_close(dir_str: str) -> bool:
    """Return True if the dir indicates a closing fill."""
    return dir_str.lower().startswith("close")


def _is_scale_in(entry_fills: list) -> bool:
    """Determine if entry fills constitute a scale-in (2+ fills within 5 min)."""
    if len(entry_fills) < 2:
        return False
    times = sorted(_parse_float(f.get("time", 0)) for f in entry_fills)
    # Check if first and last entry are within the scale-in window
    return (times[-1] - times[0]) <= SCALE_IN_WINDOW_MS


def _finalize_cluster(cluster: PositionCluster) -> dict:
    """Compute derived fields and convert cluster to dict."""
    cluster.entry_time = (
        min(_parse_float(f.get("time", 0)) for f in cluster.entry_fills)
        if cluster.entry_fills
        else None
    )
    if cluster.exit_fills:
        cluster.exit_time = max(
            _parse_float(f.get("time", 0)) for f in cluster.exit_fills
        )
    cluster.entry_price = _compute_vwap(cluster.entry_fills)
    cluster.exit_price = _compute_vwap(cluster.exit_fills)
    cluster.total_size = sum(_parse_float(f.get("sz", 0)) for f in cluster.entry_fills)
    cluster.realized_pnl = _sum_realized_pnl(cluster.exit_fills)
    cluster.fees_paid = _sum_fees(cluster.entry_fills) + _sum_fees(cluster.exit_fills)
    cluster.scale_in = _is_scale_in(cluster.entry_fills)
    return cluster.to_dict()


def _cluster_fills_for_coin(coin: str, fills: list) -> list:
    """Cluster fills for a single coin into position cycles.

    Uses the `dir` field to determine open/close and direction.
    Tracks position state to handle scale-ins, partial closes, and reversals.
    """
    if not fills:
        return []

    # Sort fills by time
    sorted_fills = sorted(fills, key=lambda f: _parse_float(f.get("time", 0)))

    clusters: list[dict] = []
    active: Optional[PositionCluster] = None

    for fill in sorted_fills:
        dir_str = fill.get("dir", "")
        fill_direction = _direction_from_dir(dir_str)
        is_open = _is_open(dir_str)
        is_close = _is_close(dir_str)

        if is_open:
            if active is None:
                # Start a new position
                active = PositionCluster(coin=coin, direction=fill_direction)
                active.entry_fills.append(fill)
            elif active.direction == fill_direction:
                # Scale-in: same direction, add to entry fills
                active.entry_fills.append(fill)
            else:
                # Direction reversal: close current, start new
                clusters.append(_finalize_cluster(active))
                active = PositionCluster(coin=coin, direction=fill_direction)
                active.entry_fills.append(fill)

        elif is_close:
            if active is not None:
                # Add to exit fills of the active position
                active.exit_fills.append(fill)

                # Check if position is fully closed via startPosition
                start_pos = _parse_float(fill.get("startPosition", 0))
                sz = _parse_float(fill.get("sz", 0))
                side = fill.get("side", "")

                # Compute position after this fill
                if side == "B":
                    pos_after = start_pos + sz
                else:
                    pos_after = start_pos - sz

                # If position is effectively closed (near zero)
                if abs(pos_after) < 1e-9:
                    clusters.append(_finalize_cluster(active))
                    active = None
            else:
                # Orphaned close fill (no active position)
                # Create a cluster with just the exit fill
                orphan = PositionCluster(coin=coin, direction=fill_direction)
                orphan.exit_fills.append(fill)
                clusters.append(_finalize_cluster(orphan))
        else:
            # Unknown dir — skip with warning
            logger.warning(
                "Unknown dir '%s' in fill for %s at time %s",
                dir_str,
                coin,
                fill.get("time"),
            )

    # Close any remaining active position (open but never closed)
    if active is not None:
        clusters.append(_finalize_cluster(active))

    return clusters


def cluster_fills(fills: list) -> list:
    """Cluster a flat list of fill dicts into position cycles.

    Fills are grouped by coin, then clustered per-coin using dir/startPosition.
    Handles partial closes, scale-ins, direction reversals, and edge cases.

    Args:
        fills: List of fill dicts with Hyperliquid schema fields:
            coin, side, px, sz, fee, closedPnl, time, dir, hash, startPosition

    Returns:
        List of position cluster dicts sorted by entry_time. Each cluster has:
            coin, direction, entry_fills, exit_fills, entry_time, exit_time,
            entry_price, exit_price, total_size, realized_pnl, fees_paid, scale_in
    """
    if not fills:
        return []

    # Group fills by coin
    coins: dict[str, list] = {}
    for f in fills:
        coin = f.get("coin", "UNKNOWN")
        coins.setdefault(coin, []).append(f)

    # Cluster per coin
    all_clusters = []
    for coin, coin_fills in coins.items():
        coin_clusters = _cluster_fills_for_coin(coin, coin_fills)
        all_clusters.extend(coin_clusters)

    # Sort by entry_time (None values go last)
    all_clusters.sort(
        key=lambda c: c["entry_time"] if c["entry_time"] is not None else float("inf")
    )

    return all_clusters


def cluster_wallet_fills(wallet_address: str, fills: list) -> dict:
    """Cluster fills for a single wallet with metadata.

    Provides a higher-level interface that includes wallet address,
    total fills, cluster count, and insufficient-data flag.

    Args:
        wallet_address: The wallet's address string.
        fills: List of fill dicts (same schema as cluster_fills).

    Returns:
        Dict with wallet metadata and position clusters:
            wallet, total_fills, num_clusters, insufficient_data, clusters
    """
    clusters = cluster_fills(fills)
    total_fills = len(fills)
    insufficient_data = total_fills < 10

    return {
        "wallet": wallet_address,
        "total_fills": total_fills,
        "num_clusters": len(clusters),
        "insufficient_data": insufficient_data,
        "clusters": clusters,
    }
