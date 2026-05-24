"""Entry Reconstruction Module

Fetches HL candle data before each position entry and identifies common
trigger patterns. Given position clusters and corresponding candle data,
analyzes the market conditions present at each entry point and finds
shared conditions across a wallet's trades.

Key functions:
  - analyze_entry_conditions(): Detects conditions at a single entry point
  - find_common_triggers(): Identifies patterns shared across ≥60% of entries
  - reconstruct_triggers(): Processes all position clusters for triggers
  - reconstruct_wallet_triggers(): High-level per-wallet trigger analysis

Conditions detected:
  - price_velocity: Rate and direction of price change in N candles before entry
  - volume_spike: Volume > N standard deviations above mean
  - volatility: Average true range as percentage of price
  - consecutive_ticks: N consecutive up/down candles before entry
  - time_of_day: UTC hour of entry

Output: Structured trigger description per position, plus common trigger
signature per wallet.

Input:
  clusters         — list of position cluster dicts from position_clustering
  candles_by_coin  — dict mapping coin symbol to list of candle dicts
                      (Hyperliquid candleSnapshot schema: t, T, o, c, h, l, v, n)
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Configuration constants
CANDLES_BEFORE_ENTRY = 6  # Number of candles to analyze before entry
LOOKBACK_MINUTES = 30  # How many minutes before entry to look
MIN_CANDLES_FOR_ANALYSIS = 1  # Minimum candles needed for analysis
COMMON_TRIGGER_THRESHOLD = 0.6  # ≥60% of entries must share a condition

# Volume spike threshold: ratio of latest candle volume to mean must exceed this
VOLUME_SPIKE_SD_THRESHOLD = 1.5  # Volume > mean + 1.5 * std

# Volatility level thresholds (as percentage of price)
VOLATILITY_LOW_THRESHOLD = 0.002  # < 0.2% range = low
VOLATILITY_HIGH_THRESHOLD = 0.01  # > 1% range = high

# Price velocity thresholds
PRICE_VELOCITY_LOW = 0.001  # < 0.1% = flat
PRICE_VELOCITY_HIGH = 0.005  # > 0.5% = strong

# Consecutive tick thresholds
MIN_CONSECUTIVE_TICKS = 2  # Minimum count to be notable


def _parse_float(value, default: float = 0.0) -> float:
    """Safely parse a string/numeric value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_candle_fields(candle: dict) -> dict:
    """Parse and normalize candle dict fields to floats.

    Handles both string and numeric field values from Hyperliquid API.
    """
    return {
        "t": _parse_float(candle.get("t", 0)),
        "T": _parse_float(candle.get("T", 0)),
        "o": _parse_float(candle.get("o", 0)),
        "c": _parse_float(candle.get("c", 0)),
        "h": _parse_float(candle.get("h", 0)),
        "l": _parse_float(candle.get("l", 0)),
        "v": _parse_float(candle.get("v", 0)),
        "n": _parse_float(candle.get("n", 0)),
        "s": candle.get("s", ""),
        "i": candle.get("i", "5m"),
    }


def _get_candles_before_entry(
    candles: list,
    entry_time: float,
    n_candles: int = CANDLES_BEFORE_ENTRY,
    interval_ms: float = 300_000,
) -> list:
    """Select the N candles immediately before entry_time.

    Args:
        candles: List of candle dicts, sorted by time.
        entry_time: Entry timestamp in ms.
        n_candles: Number of candles to retrieve.
        interval_ms: Expected interval between candles in ms.

    Returns:
        List of parsed candle dicts for the period before entry.
    """
    if not candles:
        return []

    parsed = [_parse_candle_fields(c) for c in candles]
    # Filter to candles that end before entry_time (candle.T < entry_time)
    # or start before entry_time
    before = [c for c in parsed if c["t"] < entry_time]
    # Take the last n_candles
    before = before[-n_candles:]
    return before


def _compute_price_velocity(candles: list) -> dict:
    """Compute price velocity from a sequence of candles.

    Velocity = (close_last - open_first) / open_first

    Returns:
        dict with direction ("up", "down", "flat"), magnitude (abs %),
        and raw values.
    """
    if not candles:
        return {"direction": "unknown", "magnitude": 0.0, "pct_change": 0.0}

    first_open = candles[0]["o"]
    last_close = candles[-1]["c"]

    if first_open <= 0:
        return {"direction": "unknown", "magnitude": 0.0, "pct_change": 0.0}

    pct_change = (last_close - first_open) / first_open
    magnitude = abs(pct_change)

    if magnitude < PRICE_VELOCITY_LOW:
        direction = "flat"
    elif pct_change > 0:
        direction = "up"
    else:
        direction = "down"

    return {
        "direction": direction,
        "magnitude": magnitude,
        "pct_change": pct_change,
    }


def _compute_volume_spike(candles: list) -> dict:
    """Detect volume spike in the last candle relative to prior candles.

    A volume spike is detected when the last candle's volume exceeds
    the mean + SD_THRESHOLD * std of prior candles.

    Returns:
        dict with detected (bool), ratio (float), and volume values.
    """
    if len(candles) < 2:
        return {"detected": False, "ratio": 1.0, "last_volume": 0.0, "mean_volume": 0.0}

    volumes = [c["v"] for c in candles]
    last_vol = volumes[-1]
    prior_vols = volumes[:-1]

    mean_vol = float(np.mean(prior_vols))
    std_vol = float(np.std(prior_vols))

    if mean_vol <= 0:
        return {
            "detected": False,
            "ratio": 1.0,
            "last_volume": last_vol,
            "mean_volume": mean_vol,
        }

    ratio = last_vol / mean_vol

    # Spike if last volume exceeds mean + threshold * std
    threshold = mean_vol + VOLUME_SPIKE_SD_THRESHOLD * std_vol
    detected = last_vol > threshold and ratio > 1.5

    return {
        "detected": detected,
        "ratio": ratio,
        "last_volume": last_vol,
        "mean_volume": mean_vol,
    }


def _compute_volatility(candles: list) -> dict:
    """Compute volatility from candle true ranges.

    Uses average (high - low) / close as a percentage.

    Returns:
        dict with level ("low", "medium", "high"), range_pct, and atr values.
    """
    if not candles:
        return {"level": "unknown", "range_pct": 0.0, "avg_range": 0.0}

    ranges = []
    for c in candles:
        close = c["c"]
        if close > 0:
            range_pct = (c["h"] - c["l"]) / close
            ranges.append(range_pct)

    if not ranges:
        return {"level": "unknown", "range_pct": 0.0, "avg_range": 0.0}

    avg_range = float(np.mean(ranges))

    if avg_range < VOLATILITY_LOW_THRESHOLD:
        level = "low"
    elif avg_range > VOLATILITY_HIGH_THRESHOLD:
        level = "high"
    else:
        level = "medium"

    return {
        "level": level,
        "range_pct": avg_range,
        "avg_range": avg_range,
    }


def _compute_consecutive_ticks(candles: list) -> dict:
    """Count consecutive candles closing in the same direction.

    Scans backward from the most recent candle to count how many
    consecutive candles closed up or down relative to their open.

    Returns:
        dict with count (int), direction ("up", "down", "mixed"),
        and all tick directions.
    """
    if not candles:
        return {"count": 0, "direction": "unknown", "tick_directions": []}

    # Determine direction of each candle (close vs open)
    tick_dirs = []
    for c in candles:
        if c["c"] > c["o"]:
            tick_dirs.append("up")
        elif c["c"] < c["o"]:
            tick_dirs.append("down")
        else:
            tick_dirs.append("flat")

    # Count consecutive from the end
    if not tick_dirs:
        return {"count": 0, "direction": "unknown", "tick_directions": tick_dirs}

    last_dir = tick_dirs[-1]
    count = 0
    for d in reversed(tick_dirs):
        if d == last_dir:
            count += 1
        else:
            break

    return {
        "count": count,
        "direction": last_dir if count >= MIN_CONSECUTIVE_TICKS else "mixed",
        "tick_directions": tick_dirs,
    }


def _compute_time_of_day(entry_time: float) -> dict:
    """Extract time-of-day information from entry timestamp.

    Args:
        entry_time: Timestamp in milliseconds.

    Returns:
        dict with hour_utc (0-23), day_of_week, and period labels.
    """
    if entry_time <= 0:
        return {"hour_utc": -1, "day_of_week": "unknown", "period": "unknown"}

    dt = datetime.fromtimestamp(entry_time / 1000.0, tz=timezone.utc)
    hour = dt.hour

    # Classify trading period
    if 0 <= hour < 6:
        period = "asian"
    elif 6 <= hour < 13:
        period = "european"
    elif 13 <= hour < 22:
        period = "us"
    else:
        period = "late_us"

    return {
        "hour_utc": hour,
        "day_of_week": dt.strftime("%A"),
        "period": period,
    }


def analyze_entry_conditions(
    candles: list,
    entry_time: float,
    n_candles: int = CANDLES_BEFORE_ENTRY,
) -> dict:
    """Analyze market conditions present at a position entry time.

    Examines the N candles immediately before entry_time to identify
    trigger conditions: price velocity, volume spike, volatility,
    consecutive ticks, and time-of-day.

    Args:
        candles: List of candle dicts (Hyperliquid candleSnapshot schema).
            Each candle has: t, T, o, c, h, l, v, n, s, i.
        entry_time: Position entry timestamp in milliseconds.
        n_candles: Number of candles before entry to analyze (default 6).

    Returns:
        Dict with structured condition descriptions:
            price_velocity: {direction, magnitude, pct_change}
            volume_spike: {detected, ratio, last_volume, mean_volume}
            volatility: {level, range_pct, avg_range}
            consecutive_ticks: {count, direction, tick_directions}
            time_of_day: {hour_utc, day_of_week, period}
    """
    # Select candles before entry
    before_entry = _get_candles_before_entry(candles, entry_time, n_candles)

    # Compute each condition
    price_velocity = _compute_price_velocity(before_entry)
    volume_spike = _compute_volume_spike(before_entry)
    volatility = _compute_volatility(before_entry)
    consecutive_ticks = _compute_consecutive_ticks(before_entry)
    time_of_day = _compute_time_of_day(entry_time)

    return {
        "price_velocity": price_velocity,
        "volume_spike": volume_spike,
        "volatility": volatility,
        "consecutive_ticks": consecutive_ticks,
        "time_of_day": time_of_day,
    }


def find_common_triggers(trigger_descriptions: list) -> dict:
    """Identify conditions shared across ≥60% of a wallet's entries.

    Given all trigger descriptions for a single wallet, scans each
    condition dimension to find patterns that appear in at least 60%
    of entries.

    Args:
        trigger_descriptions: List of trigger dicts, each with:
            position_index: int
            entry_time: float (ms)
            coin: str
            conditions: dict from analyze_entry_conditions()

    Returns:
        Dict with:
            trigger_signature: Human-readable string describing the
                common trigger pattern
            common_conditions: List of dicts, each with:
                name: condition name (e.g., "volume_spike_detected")
                value: the shared value
                pct_matching: fraction of entries matching this condition
                matching_count: absolute count of matching entries
            pct_matching: highest percentage match across common conditions
            total_entries: number of entries analyzed
    """
    if not trigger_descriptions:
        return {
            "trigger_signature": "no entries",
            "common_conditions": [],
            "pct_matching": 0.0,
            "total_entries": 0,
        }

    n_entries = len(trigger_descriptions)

    # Extract condition values across all entries
    condition_matches = {}

    # Price velocity direction
    directions = [
        t["conditions"]["price_velocity"]["direction"] for t in trigger_descriptions
    ]
    for d in ("up", "down", "flat"):
        count = sum(1 for x in directions if x == d)
        pct = count / n_entries
        if pct >= COMMON_TRIGGER_THRESHOLD:
            condition_matches[f"price_velocity_{d}"] = {
                "name": f"price_velocity_{d}",
                "value": d,
                "matching_count": count,
                "pct_matching": pct,
            }

    # Price velocity magnitude (strong movement)
    strong_moves = sum(
        1
        for t in trigger_descriptions
        if t["conditions"]["price_velocity"]["magnitude"] > PRICE_VELOCITY_HIGH
    )
    pct = strong_moves / n_entries
    if pct >= COMMON_TRIGGER_THRESHOLD:
        condition_matches["strong_price_movement"] = {
            "name": "strong_price_movement",
            "value": f">{PRICE_VELOCITY_HIGH * 100:.1f}%",
            "matching_count": strong_moves,
            "pct_matching": pct,
        }

    # Volume spike
    vol_spikes = sum(
        1 for t in trigger_descriptions if t["conditions"]["volume_spike"]["detected"]
    )
    pct = vol_spikes / n_entries
    if pct >= COMMON_TRIGGER_THRESHOLD:
        condition_matches["volume_spike_detected"] = {
            "name": "volume_spike_detected",
            "value": True,
            "matching_count": vol_spikes,
            "pct_matching": pct,
        }

    # Volatility level
    for level in ("low", "medium", "high"):
        count = sum(
            1
            for t in trigger_descriptions
            if t["conditions"]["volatility"]["level"] == level
        )
        pct = count / n_entries
        if pct >= COMMON_TRIGGER_THRESHOLD:
            condition_matches[f"volatility_{level}"] = {
                "name": f"volatility_{level}",
                "value": level,
                "matching_count": count,
                "pct_matching": pct,
            }

    # Consecutive ticks (direction + count ≥ MIN_CONSECUTIVE_TICKS)
    for tick_dir in ("up", "down"):
        count = sum(
            1
            for t in trigger_descriptions
            if (
                t["conditions"]["consecutive_ticks"]["direction"] == tick_dir
                and t["conditions"]["consecutive_ticks"]["count"]
                >= MIN_CONSECUTIVE_TICKS
            )
        )
        pct = count / n_entries
        if pct >= COMMON_TRIGGER_THRESHOLD:
            condition_matches[f"consecutive_{tick_dir}_ticks"] = {
                "name": f"consecutive_{tick_dir}_ticks",
                "value": f"≥{MIN_CONSECUTIVE_TICKS}",
                "matching_count": count,
                "pct_matching": pct,
            }

    # Time-of-day period
    periods = [
        t["conditions"].get("time_of_day", {}).get("period", "unknown")
        for t in trigger_descriptions
    ]
    for period in ("asian", "european", "us", "late_us"):
        count = sum(1 for p in periods if p == period)
        pct = count / n_entries
        if pct >= COMMON_TRIGGER_THRESHOLD:
            condition_matches[f"trading_period_{period}"] = {
                "name": f"trading_period_{period}",
                "value": period,
                "matching_count": count,
                "pct_matching": pct,
            }

    # Build results
    common_conditions = sorted(
        condition_matches.values(),
        key=lambda x: x["pct_matching"],
        reverse=True,
    )

    # Compute best pct_matching
    pct_matching = common_conditions[0]["pct_matching"] if common_conditions else 0.0

    # Generate human-readable trigger signature
    signature_parts = []
    for c in common_conditions:
        name = c["name"]
        pct_str = f"{c['pct_matching']:.0%}"
        if name.startswith("price_velocity_"):
            signature_parts.append(f"price moving {c['value']} ({pct_str})")
        elif name == "strong_price_movement":
            signature_parts.append(f"strong price move {c['value']} ({pct_str})")
        elif name == "volume_spike_detected":
            signature_parts.append(f"volume spike ({pct_str})")
        elif name.startswith("volatility_"):
            signature_parts.append(f"volatility {c['value']} ({pct_str})")
        elif name.startswith("consecutive_"):
            signature_parts.append(
                f"{c['value']}+ consecutive {c.get('direction', '')} ticks ({pct_str})"
            )
        elif name.startswith("trading_period_"):
            signature_parts.append(f"{c['value']} session ({pct_str})")

    trigger_signature = (
        " + ".join(signature_parts) if signature_parts else "no common pattern"
    )

    return {
        "trigger_signature": trigger_signature,
        "common_conditions": common_conditions,
        "pct_matching": pct_matching,
        "total_entries": n_entries,
    }


def _default_conditions() -> dict:
    """Return default/empty conditions when no candle data is available."""
    return {
        "price_velocity": {"direction": "unknown", "magnitude": 0.0, "pct_change": 0.0},
        "volume_spike": {
            "detected": False,
            "ratio": 1.0,
            "last_volume": 0.0,
            "mean_volume": 0.0,
        },
        "volatility": {"level": "unknown", "range_pct": 0.0, "avg_range": 0.0},
        "consecutive_ticks": {
            "count": 0,
            "direction": "unknown",
            "tick_directions": [],
        },
        "time_of_day": {"hour_utc": -1, "day_of_week": "unknown", "period": "unknown"},
    }


def reconstruct_triggers(
    clusters: list,
    candles_by_coin: dict,
) -> dict:
    """Reconstruct entry triggers for all position clusters.

    For each position cluster, fetches the relevant candle data and
    analyzes conditions at entry time. Then finds common triggers
    across all entries.

    Args:
        clusters: List of position cluster dicts, each with:
            coin, direction, entry_time, exit_time, entry_price,
            exit_price, total_size, realized_pnl, fees_paid, scale_in
        candles_by_coin: Dict mapping coin symbol to list of candle dicts.
            Candles should cover the time range of all clusters.

    Returns:
        Dict with:
            triggers: List of per-position trigger descriptions
            common_triggers: Common trigger analysis across all entries
    """
    if not clusters:
        return {"triggers": [], "common_triggers": {}}

    triggers = []
    for idx, cluster in enumerate(clusters):
        coin = cluster.get("coin", "UNKNOWN")
        entry_time = cluster.get("entry_time", 0)

        # Get candles for this coin
        candles = candles_by_coin.get(coin, [])

        if not candles or entry_time <= 0:
            # No candle data for this position
            conditions = _default_conditions()
            if entry_time > 0:
                conditions["time_of_day"] = _compute_time_of_day(entry_time)
        else:
            conditions = analyze_entry_conditions(candles, entry_time)

        triggers.append(
            {
                "position_index": idx,
                "entry_time": entry_time,
                "coin": coin,
                "direction": cluster.get("direction", "unknown"),
                "conditions": conditions,
            }
        )

    # Find common triggers across all entries
    common_triggers = find_common_triggers(triggers)

    return {
        "triggers": triggers,
        "common_triggers": common_triggers,
    }


def reconstruct_wallet_triggers(
    wallet_address: str,
    clusters: list,
    fills: list,
    candles_by_coin: dict,
) -> dict:
    """High-level wallet trigger reconstruction.

    Combines position cluster analysis with candle data to produce
    a comprehensive trigger profile for a wallet.

    Args:
        wallet_address: The wallet's address string.
        clusters: List of position cluster dicts from position_clustering.
        fills: List of raw fill dicts (used for coin discovery and metadata).
        candles_by_coin: Dict mapping coin symbol to list of candle dicts.

    Returns:
        Dict with:
            wallet: wallet address
            num_positions: count of position clusters analyzed
            triggers: list of per-position trigger descriptions
            common_triggers: common trigger analysis
            coins_analyzed: list of coins with candle data
    """
    result = reconstruct_triggers(clusters, candles_by_coin)

    # Discover which coins had candle data
    coins_with_candles = sorted(candles_by_coin.keys())
    coins_in_clusters = sorted(set(c.get("coin", "UNKNOWN") for c in clusters))

    return {
        "wallet": wallet_address,
        "num_positions": len(clusters),
        "triggers": result["triggers"],
        "common_triggers": result["common_triggers"],
        "coins_analyzed": coins_in_clusters,
        "coins_with_candle_data": coins_with_candles,
    }
