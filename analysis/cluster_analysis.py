"""Cluster Analysis Module

Groups wallets by strategy similarity using multi-criteria matching.
The clustering is deterministic and idempotent: same input always produces
same output. No random seeds are used.

Similarity criteria (from validation contract VAL-ANALYSIS-005):
  1. Same strategy type (from strategy_classifier)
  2. Same primary market
  3. Same direction pattern (long/short/mixed)
  4. Similar hold times (within ±20% of cluster median)
  5. Similar clip sizes (within ±30% of cluster median)

Output per cluster:
  - cluster_id: unique identifier
  - strategy: shared strategy type
  - primary_market: shared primary market
  - direction: shared direction pattern
  - member_wallets: list of wallet addresses
  - shared_parameters: median hold time, clip size, win rate with ranges
  - divergence_metrics: max deviation from cluster median

Pipeline flow:
  normalize_fills → cluster_fills → compute_metrics → classify_wallet
  → group_wallets_by_similarity → clusters
"""

import json
import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOLD_TIME_TOLERANCE = 0.20  # ±20%
CLIP_SIZE_TOLERANCE = 0.30  # ±30%
MIN_CLUSTER_SIZE = 5


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


# ---------------------------------------------------------------------------
# Fill normalization
# ---------------------------------------------------------------------------


def normalize_fills(fills: list) -> list:
    """Normalize fill data from scraper format to analysis format.

    The scraper writes ``closed_pnl`` (snake_case) while the analysis
    modules expect ``closedPnl`` (camelCase, matching Hyperliquid API).
    The scraper also omits ``startPosition``, so we infer it by tracking
    the running position per coin.

    Args:
        fills: List of fill dicts from ``data/wallets-hl.json``.

    Returns:
        List of normalized fill dicts compatible with
        ``position_clustering.cluster_fills()``.
    """
    if not fills:
        return []

    # Group by coin for per-coin position tracking
    by_coin: dict[str, list] = {}
    for f in fills:
        coin = f.get("coin", "UNKNOWN")
        by_coin.setdefault(coin, []).append(f)

    normalized: list[dict] = []
    for coin, coin_fills in by_coin.items():
        # Sort fills within each coin by time
        coin_fills_sorted = sorted(
            coin_fills, key=lambda f: _parse_float(f.get("time", 0))
        )

        position = 0.0  # running position for this coin
        for f in coin_fills_sorted:
            new_f = dict(f)

            # Convert closed_pnl → closedPnl (idempotent)
            if "closed_pnl" in new_f and "closedPnl" not in new_f:
                new_f["closedPnl"] = new_f["closed_pnl"]

            # Infer startPosition from tracked position
            new_f["startPosition"] = position

            # Update position after this fill
            sz = _parse_float(f.get("sz", 0))
            side = f.get("side", "")
            if side == "B":
                position += sz
            else:  # 'A' = sell
                position -= sz

            normalized.append(new_f)

    # Sort all normalized fills by time
    normalized.sort(key=lambda f: _parse_float(f.get("time", 0)))
    return normalized


# ---------------------------------------------------------------------------
# Per-wallet pipeline
# ---------------------------------------------------------------------------


def _compute_median_fill_notional(fills: list) -> float:
    """Compute median fill notional value (px * sz) for a wallet."""
    if not fills:
        return 0.0
    notionals = []
    for f in fills:
        px = _parse_float(f.get("px", 0))
        sz = _parse_float(f.get("sz", 0))
        notional = px * sz
        if notional > 0:
            notionals.append(notional)
    if not notionals:
        return 0.0
    return float(np.median(notionals))


def compute_wallet_profile(wallet_data: dict) -> dict:
    """Run the full analysis pipeline on a single wallet.

    Steps: normalize fills → cluster positions → compute metrics →
    classify strategy.

    Args:
        wallet_data: Dict with ``address`` and ``fills`` keys.

    Returns:
        Dict with address, strategy, confidence, evidence, metrics,
        clusters, and median_fill_notional.
    """
    from analysis.position_clustering import cluster_fills
    from analysis.wallet_metrics import compute_wallet_metrics
    from analysis.strategy_classifier import classify_wallet

    address = wallet_data.get("address", "unknown")
    raw_fills = wallet_data.get("fills", [])

    logger.debug("Processing wallet %s (%d raw fills)", address, len(raw_fills))

    # Normalize fills to expected format
    fills = normalize_fills(raw_fills)

    # Cluster fills into position cycles
    clusters = cluster_fills(fills)

    # Compute wallet metrics
    metrics = compute_wallet_metrics(clusters, fills)

    # Classify strategy
    classification = classify_wallet(metrics, clusters)

    # Compute median fill notional for clustering
    median_fill_notional = _compute_median_fill_notional(fills)

    return {
        "address": address,
        "strategy": classification["strategy"],
        "confidence": classification["confidence"],
        "evidence": classification["evidence"],
        "metrics": metrics,
        "clusters": clusters,
        "num_clusters": len(clusters),
        "median_fill_notional": median_fill_notional,
    }


# ---------------------------------------------------------------------------
# Clustering algorithm
# ---------------------------------------------------------------------------


def _split_by_tolerance(
    sorted_profiles: list,
    key_func,
    tolerance: float,
    min_size: int,
) -> list[list]:
    """Split sorted profiles into sub-groups within tolerance of group median.

    Deterministic greedy algorithm: iterate through sorted profiles, add to
    current group if the profile's value is within ``tolerance`` of the
    group's running median. Otherwise start a new group.

    Only returns sub-groups with ``len >= min_size``.
    """
    if not sorted_profiles:
        return []

    sub_groups: list[list] = []
    current_group = [sorted_profiles[0]]

    for p in sorted_profiles[1:]:
        current_values = [key_func(m) for m in current_group]
        median_val = float(np.median(current_values))
        p_val = key_func(p)

        if median_val > 0 and p_val > 0:
            ratio = p_val / median_val
            if (1 - tolerance) <= ratio <= (1 + tolerance):
                current_group.append(p)
            else:
                sub_groups.append(current_group)
                current_group = [p]
        elif median_val == 0 and p_val == 0:
            # Both zero → considered similar
            current_group.append(p)
        else:
            sub_groups.append(current_group)
            current_group = [p]

    sub_groups.append(current_group)

    # Filter by minimum size
    result = [sg for sg in sub_groups if len(sg) >= min_size]

    # If nothing met min_size but we have enough profiles overall, return
    # the whole group as one (best effort)
    if not result and len(sorted_profiles) >= min_size:
        result = [sorted_profiles]

    return result


def _build_cluster(
    cluster_id: int,
    strategy: str,
    market: str,
    direction: str,
    profiles: list,
) -> dict:
    """Build a cluster dict from a list of wallet profiles."""
    addresses = [p["address"] for p in profiles]

    hold_times = [
        _safe_get(p, "metrics", "avg_hold_time_hours", default=0) for p in profiles
    ]
    clip_notionals = [p.get("median_fill_notional", 0) for p in profiles]
    win_rates = [_safe_get(p, "metrics", "win_rate", default=0) for p in profiles]

    median_hold_time = float(np.median(hold_times)) if hold_times else 0.0
    median_clip = float(np.median(clip_notionals)) if clip_notionals else 0.0
    median_win_rate = float(np.median(win_rates)) if win_rates else 0.0

    ht_min = min(hold_times) if hold_times else 0.0
    ht_max = max(hold_times) if hold_times else 0.0
    clip_min = min(clip_notionals) if clip_notionals else 0.0
    clip_max = max(clip_notionals) if clip_notionals else 0.0

    # Divergence: max fractional deviation from median
    ht_divergence = 0.0
    if median_hold_time > 0:
        ht_divergence = max(
            abs(h - median_hold_time) / median_hold_time for h in hold_times
        )

    clip_divergence = 0.0
    if median_clip > 0:
        clip_divergence = max(
            abs(c - median_clip) / median_clip for c in clip_notionals
        )

    return {
        "cluster_id": f"cluster-{cluster_id:03d}",
        "strategy": strategy,
        "primary_market": market,
        "direction": direction,
        "member_wallets": addresses,
        "size": len(profiles),
        "shared_parameters": {
            "median_hold_time_hours": round(median_hold_time, 4),
            "median_clip_notional": round(median_clip, 2),
            "median_win_rate": round(median_win_rate, 4),
            "hold_time_range": (round(ht_min, 4), round(ht_max, 4)),
            "clip_notional_range": (round(clip_min, 2), round(clip_max, 2)),
        },
        "divergence_metrics": {
            "hold_time_max_divergence": round(ht_divergence, 4),
            "clip_size_max_divergence": round(clip_divergence, 4),
        },
        "profiles": profiles,
    }


def _dominant_market(profiles: list) -> str:
    """Return the most common primary_market among profiles."""
    from collections import Counter

    markets = [
        _safe_get(p, "metrics", "primary_market", default="UNKNOWN") for p in profiles
    ]
    if not markets:
        return "UNKNOWN"
    return Counter(markets).most_common(1)[0][0]


def _try_grouping(profiles: list, key_func, min_cluster_size: int) -> list:
    """Attempt to create clusters by grouping profiles with key_func.

    Applies hold-time and clip-size tolerance splitting within each group.
    Returns a list of (key_tuple, final_group_profiles) pairs for groups
    that meet min_cluster_size.
    """
    groups: dict[tuple, list] = {}
    for p in profiles:
        key = key_func(p)
        groups.setdefault(key, []).append(p)

    results: list[tuple[tuple, list]] = []
    for key, group_profiles in sorted(groups.items()):
        if len(group_profiles) < min_cluster_size:
            continue

        # Sort by hold time for deterministic splitting
        sorted_by_ht = sorted(
            group_profiles,
            key=lambda p: _safe_get(p, "metrics", "avg_hold_time_hours", default=0),
        )

        # Split by hold time tolerance
        ht_sub_groups = _split_by_tolerance(
            sorted_by_ht,
            key_func=lambda p: _safe_get(
                p, "metrics", "avg_hold_time_hours", default=0
            ),
            tolerance=HOLD_TIME_TOLERANCE,
            min_size=min_cluster_size,
        )

        if not ht_sub_groups:
            ht_sub_groups = [group_profiles]

        # Split each sub-group by clip size
        for ht_group in ht_sub_groups:
            sorted_by_clip = sorted(
                ht_group, key=lambda p: p.get("median_fill_notional", 0)
            )

            clip_sub_groups = _split_by_tolerance(
                sorted_by_clip,
                key_func=lambda p: p.get("median_fill_notional", 0),
                tolerance=CLIP_SIZE_TOLERANCE,
                min_size=min_cluster_size,
            )

            if not clip_sub_groups:
                clip_sub_groups = [ht_group]

            for final_group in clip_sub_groups:
                results.append((key, final_group))

    return results


def group_wallets_by_similarity(
    profiles: list,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> list:
    """Group wallets by strategy similarity criteria.

    Uses a hierarchical fallback approach:

    1. Try grouping by ``(strategy, primary_market, direction)`` — most
       specific, keeps same-market wallets together.
    2. If that produces < 3 qualifying clusters, fall back to
       ``(strategy, direction)`` — allows multi-market clusters but
       keeps direction consistency.
    3. If still insufficient, fall back to ``(strategy)`` only —
       broadest grouping.

    Within each grouping level, profiles are further split by hold-time
    similarity (±20%) and clip-size similarity (±30%).

    The algorithm is deterministic: same input always produces same output.

    Args:
        profiles: List of wallet profile dicts from
            ``compute_wallet_profile()``.
        min_cluster_size: Minimum wallets per cluster (default 5).

    Returns:
        List of cluster dicts sorted by size descending, then by
        cluster_id for deterministic ordering.
    """
    # Filter out unclassifiable wallets
    classified = [
        p for p in profiles if p.get("strategy") not in ("insufficient_data", "unknown")
    ]

    if len(classified) < min_cluster_size:
        return []

    # Define grouping levels (most specific → least specific)
    grouping_levels = [
        # Level 0: strategy + market + direction (ideal)
        lambda p: (
            p.get("strategy", "unknown"),
            _safe_get(p, "metrics", "primary_market", default="UNKNOWN"),
            _safe_get(p, "metrics", "preferred_direction", default="unknown"),
        ),
        # Level 1: strategy + direction (broader)
        lambda p: (
            p.get("strategy", "unknown"),
            _safe_get(p, "metrics", "preferred_direction", default="unknown"),
        ),
        # Level 2: strategy only (broadest)
        lambda p: (p.get("strategy", "unknown"),),
    ]

    cluster_results: list[tuple[tuple, list]] = []

    for level, key_func in enumerate(grouping_levels):
        cluster_results = _try_grouping(classified, key_func, min_cluster_size)
        n_clusters = len(cluster_results)
        logger.debug(
            "Grouping level %d produced %d clusters (need >= 3)",
            level,
            n_clusters,
        )
        if n_clusters >= 3:
            break

    # Build cluster dicts from results
    clusters: list[dict] = []
    cluster_id = 0

    for key_tuple, group_profiles in cluster_results:
        cluster_id += 1

        # Extract grouping info from key
        strategy = key_tuple[0]
        direction = key_tuple[-1] if len(key_tuple) > 1 else "mixed"
        market = (
            key_tuple[1] if len(key_tuple) == 3 else _dominant_market(group_profiles)
        )

        cluster = _build_cluster(
            cluster_id, strategy, market, direction, group_profiles
        )
        clusters.append(cluster)

    # Sort by size descending, then cluster_id for stability
    clusters.sort(key=lambda c: (-c["size"], c["cluster_id"]))

    # Re-number after sorting
    for i, c in enumerate(clusters):
        c["cluster_id"] = f"cluster-{i + 1:03d}"

    return clusters


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_clusters(
    wallets_data: list,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> dict:
    """Main entry point: run full pipeline and produce clusters.

    Args:
        wallets_data: List of wallet dicts from ``wallets-hl.json``, each
            with ``address`` and ``fills`` keys.
        min_cluster_size: Minimum wallets per cluster (default 5).

    Returns:
        Dict with:
        - clusters: list of cluster dicts
        - total_wallets: int
        - classified_wallets: int
        - clustered_wallets: int
        - unclustered_wallets: int
        - profiles: list of all wallet profiles
    """
    logger.info("Analyzing %d wallets for strategy clusters", len(wallets_data))

    # Compute profiles for all wallets
    profiles: list[dict] = []
    for wallet_data in wallets_data:
        try:
            profile = compute_wallet_profile(wallet_data)
            profiles.append(profile)
        except Exception as e:
            address = wallet_data.get("address", "unknown")
            logger.warning("Failed to process wallet %s: %s", address, e)

    logger.info("Computed profiles for %d wallets", len(profiles))

    # Group wallets by similarity
    clusters = group_wallets_by_similarity(profiles, min_cluster_size)

    # Statistics
    total_wallets = len(wallets_data)
    classified_wallets = sum(
        1 for p in profiles if p.get("strategy") not in ("unknown", "insufficient_data")
    )
    clustered_wallets = sum(c["size"] for c in clusters)
    unclustered = classified_wallets - clustered_wallets

    logger.info(
        "Found %d clusters covering %d wallets (%d unclustered)",
        len(clusters),
        clustered_wallets,
        unclustered,
    )

    return {
        "clusters": clusters,
        "total_wallets": total_wallets,
        "classified_wallets": classified_wallets,
        "clustered_wallets": clustered_wallets,
        "unclustered_wallets": unclustered,
        "profiles": profiles,
    }


# ---------------------------------------------------------------------------
# File I/O (atomic writes)
# ---------------------------------------------------------------------------


def save_results(results: dict, output_path: str) -> None:
    """Save cluster analysis results using atomic write.

    Writes to ``<path>.tmp`` first, then renames to final path.
    Excludes raw fill/cluster data from profiles to keep output manageable.
    """
    serializable = {
        "total_wallets": results["total_wallets"],
        "classified_wallets": results["classified_wallets"],
        "clustered_wallets": results["clustered_wallets"],
        "unclustered_wallets": results["unclustered_wallets"],
        "clusters": [],
    }

    for cluster in results["clusters"]:
        cluster_out = {
            "cluster_id": cluster["cluster_id"],
            "strategy": cluster["strategy"],
            "primary_market": cluster["primary_market"],
            "direction": cluster["direction"],
            "member_wallets": cluster["member_wallets"],
            "size": cluster["size"],
            "shared_parameters": cluster["shared_parameters"],
            "divergence_metrics": cluster["divergence_metrics"],
        }
        serializable["clusters"].append(cluster_out)

    # Add lightweight profile summaries (no raw fills)
    profile_summaries = []
    for p in results.get("profiles", []):
        profile_summaries.append(
            {
                "address": p["address"],
                "strategy": p["strategy"],
                "confidence": p["confidence"],
                "median_fill_notional": p.get("median_fill_notional", 0),
                "metrics": {
                    k: v
                    for k, v in p.get("metrics", {}).items()
                    if k != "active_hours"  # skip large lists
                },
            }
        )
    serializable["profile_summaries"] = profile_summaries

    # Atomic write
    json_str = json.dumps(serializable, indent=2, default=str)
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(json_str)

    os.replace(tmp_path, output_path)
    logger.info("Saved cluster analysis to %s", output_path)
