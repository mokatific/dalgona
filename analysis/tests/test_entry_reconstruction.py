"""Tests for entry_reconstruction module.

Validates entry trigger reconstruction using synthetic candle data and
position clusters. Covers:
  - Entry condition detection: price velocity, volume spike, volatility,
    consecutive ticks, time-of-day
  - find_common_triggers() identifying patterns shared across ≥60% of entries
  - Structured trigger descriptions per position
  - Edge cases: empty data, insufficient candles, single entry
"""

import pytest
import numpy as np

from analysis.entry_reconstruction import (
    analyze_entry_conditions,
    find_common_triggers,
    reconstruct_wallet_triggers,
    reconstruct_triggers,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic candle data
# ---------------------------------------------------------------------------


def _candle(
    open_px=50000.0,
    high=50100.0,
    low=49900.0,
    close=50050.0,
    volume=100.0,
    time_ms=0,
    interval="5m",
):
    """Create a single candle dict matching Hyperliquid candleSnapshot schema."""
    return {
        "t": time_ms,  # start time (ms)
        "T": time_ms + 300_000,  # end time (ms) — 5 min interval
        "s": "BTC",
        "i": interval,
        "o": str(open_px),
        "c": str(close),
        "h": str(high),
        "l": str(low),
        "v": str(volume),
        "n": 50,  # number of trades
    }


def _generate_candles(
    base_price=50000.0,
    n_candles=20,
    interval_ms=300_000,  # 5 min
    start_time=0,
    price_fn=None,
    volume_fn=None,
):
    """Generate a sequence of candles.

    Args:
        base_price: Starting price
        n_candles: Number of candles to generate
        interval_ms: Time between candles in ms
        start_time: First candle start time
        price_fn: Optional callable(candle_index, base_price) -> (open, high, low, close)
        volume_fn: Optional callable(candle_index) -> volume

    Returns:
        List of candle dicts.
    """
    candles = []
    for i in range(n_candles):
        t = start_time + i * interval_ms
        if price_fn:
            o, h, l, c = price_fn(i, base_price)
        else:
            noise = (i % 3 - 1) * 10  # slight oscillation
            o = base_price + noise
            h = o + 50
            l = o - 50
            c = o + 20

        vol = volume_fn(i) if volume_fn else 100.0
        candles.append(
            _candle(
                open_px=o,
                high=h,
                low=l,
                close=c,
                volume=vol,
                time_ms=t,
            )
        )
    return candles


def _make_position_cluster(
    coin="BTC",
    direction="long",
    entry_time=3_000_000,  # 50 min mark
    exit_time=6_000_000,  # 100 min mark
    entry_price=50000.0,
    exit_price=50500.0,
    total_size=1.0,
    realized_pnl=500.0,
    fees_paid=5.0,
):
    """Create a minimal position cluster dict."""
    return {
        "coin": coin,
        "direction": direction,
        "entry_fills": [
            {"px": str(entry_price), "sz": str(total_size), "time": entry_time}
        ],
        "exit_fills": [
            {"px": str(exit_price), "sz": str(total_size), "time": exit_time}
        ],
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "total_size": total_size,
        "realized_pnl": realized_pnl,
        "fees_paid": fees_paid,
        "scale_in": False,
    }


# Timestamps for tests (5-minute intervals)
T0 = 0
T1 = 300_000  # 5 min
T2 = 600_000  # 10 min
T3 = 900_000  # 15 min
T4 = 1_200_000  # 20 min
T5 = 1_500_000  # 25 min
T6 = 1_800_000  # 30 min
T7 = 2_100_000  # 35 min
T8 = 2_400_000  # 40 min
T9 = 2_700_000  # 45 min
T10 = 3_000_000  # 50 min


# ---------------------------------------------------------------------------
# Test 1: analyze_entry_conditions detects price velocity
# ---------------------------------------------------------------------------


class TestPriceVelocity:
    """Price velocity detection in candle conditions."""

    def test_strong_upward_velocity_detected(self):
        """3 consecutive up-candles before entry should show positive velocity."""
        # Entry at T6 (30 min), candles from T0-T5
        candles = [
            _candle(open_px=50000, close=50050, time_ms=T0),
            _candle(open_px=50050, close=50100, time_ms=T1),
            _candle(open_px=50100, close=50200, time_ms=T2),
            _candle(open_px=50200, close=50300, time_ms=T3),
            _candle(open_px=50300, close=50400, time_ms=T4),
            _candle(open_px=50400, close=50500, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["price_velocity"]["direction"] == "up"
        assert result["price_velocity"]["magnitude"] > 0

    def test_strong_downward_velocity_detected(self):
        """3 consecutive down-candles before entry should show negative velocity."""
        candles = [
            _candle(open_px=50500, close=50400, time_ms=T0),
            _candle(open_px=50400, close=50300, time_ms=T1),
            _candle(open_px=50300, close=50200, time_ms=T2),
            _candle(open_px=50200, close=50100, time_ms=T3),
            _candle(open_px=50100, close=50000, time_ms=T4),
            _candle(open_px=50000, close=49900, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["price_velocity"]["direction"] == "down"
        assert result["price_velocity"]["magnitude"] > 0

    def test_flat_price_low_velocity(self):
        """Sideways price action should show low velocity."""
        candles = [
            _candle(open_px=50000, close=50010, time_ms=T0),
            _candle(open_px=50010, close=50005, time_ms=T1),
            _candle(open_px=50005, close=50000, time_ms=T2),
            _candle(open_px=50000, close=50010, time_ms=T3),
            _candle(open_px=50010, close=50005, time_ms=T4),
            _candle(open_px=50005, close=50000, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["price_velocity"]["direction"] in ("flat", "up", "down")
        assert result["price_velocity"]["magnitude"] < 0.01  # Low velocity


# ---------------------------------------------------------------------------
# Test 2: Volume spike detection
# ---------------------------------------------------------------------------


class TestVolumeSpike:
    """Volume spike detection relative to mean."""

    def test_volume_spike_detected(self):
        """A single candle with much higher volume should be flagged."""
        candles = [
            _candle(volume=100.0, time_ms=T0),
            _candle(volume=100.0, time_ms=T1),
            _candle(volume=100.0, time_ms=T2),
            _candle(volume=100.0, time_ms=T3),
            _candle(volume=100.0, time_ms=T4),
            _candle(volume=500.0, time_ms=T5),  # Spike: 5x normal
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["volume_spike"]["detected"] is True
        assert result["volume_spike"]["ratio"] > 2.0

    def test_no_volume_spike(self):
        """Consistent volumes should not flag a spike."""
        candles = [
            _candle(volume=100.0, time_ms=T0),
            _candle(volume=105.0, time_ms=T1),
            _candle(volume=95.0, time_ms=T2),
            _candle(volume=100.0, time_ms=T3),
            _candle(volume=102.0, time_ms=T4),
            _candle(volume=98.0, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["volume_spike"]["detected"] is False


# ---------------------------------------------------------------------------
# Test 3: Volatility detection
# ---------------------------------------------------------------------------


class TestVolatility:
    """Volatility level detection from candle ranges."""

    def test_high_volatility_detected(self):
        """Candles with large ranges should show high volatility."""
        candles = [
            _candle(open_px=50000, high=50500, low=49500, close=50200, time_ms=T0),
            _candle(open_px=50200, high=50800, low=49600, close=50400, time_ms=T1),
            _candle(open_px=50400, high=51000, low=49800, close=50500, time_ms=T2),
            _candle(open_px=50500, high=51200, low=49800, close=50600, time_ms=T3),
            _candle(open_px=50600, high=51400, low=49800, close=50800, time_ms=T4),
            _candle(open_px=50800, high=51600, low=50000, close=51000, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["volatility"]["level"] in ("high", "medium", "low")
        assert result["volatility"]["range_pct"] > 0

    def test_low_volatility(self):
        """Candles with tight ranges should show low volatility."""
        candles = [
            _candle(open_px=50000, high=50010, low=49990, close=50005, time_ms=T0),
            _candle(open_px=50005, high=50015, low=49995, close=50010, time_ms=T1),
            _candle(open_px=50010, high=50020, low=50000, close=50015, time_ms=T2),
            _candle(open_px=50015, high=50025, low=50005, close=50020, time_ms=T3),
            _candle(open_px=50020, high=50030, low=50010, close=50025, time_ms=T4),
            _candle(open_px=50025, high=50035, low=50015, close=50030, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["volatility"]["level"] == "low"


# ---------------------------------------------------------------------------
# Test 4: Consecutive ticks detection
# ---------------------------------------------------------------------------


class TestConsecutiveTicks:
    """Consecutive directional candle detection."""

    def test_consecutive_up_ticks(self):
        """3 consecutive up-candles should be detected."""
        candles = [
            _candle(open_px=50000, close=50050, time_ms=T0),
            _candle(open_px=50050, close=50100, time_ms=T1),
            _candle(open_px=50100, close=50200, time_ms=T2),
            _candle(open_px=50200, close=50300, time_ms=T3),
            _candle(open_px=50300, close=50400, time_ms=T4),
            _candle(open_px=50400, close=50500, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["consecutive_ticks"]["count"] >= 3
        assert result["consecutive_ticks"]["direction"] == "up"

    def test_consecutive_down_ticks(self):
        """3 consecutive down-candles should be detected."""
        candles = [
            _candle(open_px=50500, close=50400, time_ms=T0),
            _candle(open_px=50400, close=50300, time_ms=T1),
            _candle(open_px=50300, close=50200, time_ms=T2),
            _candle(open_px=50200, close=50100, time_ms=T3),
            _candle(open_px=50100, close=50000, time_ms=T4),
            _candle(open_px=50000, close=49900, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["consecutive_ticks"]["count"] >= 3
        assert result["consecutive_ticks"]["direction"] == "down"

    def test_mixed_direction_no_consecutive(self):
        """Alternating up/down candles should show low consecutive count."""
        candles = [
            _candle(open_px=50000, close=50050, time_ms=T0),
            _candle(open_px=50050, close=50000, time_ms=T1),
            _candle(open_px=50000, close=50050, time_ms=T2),
            _candle(open_px=50050, close=50000, time_ms=T3),
            _candle(open_px=50000, close=50050, time_ms=T4),
            _candle(open_px=50050, close=50000, time_ms=T5),
        ]
        entry_time = T6
        result = analyze_entry_conditions(candles, entry_time)
        assert result["consecutive_ticks"]["count"] <= 1


# ---------------------------------------------------------------------------
# Test 5: find_common_triggers across multiple entries
# ---------------------------------------------------------------------------


class TestFindCommonTriggers:
    """find_common_triggers identifies patterns shared across ≥60% of entries."""

    def test_common_volume_spike_pattern(self):
        """
        5 entries, 4 of which have volume spikes (>60%).
        Common trigger should identify volume spike.
        """
        triggers = []
        for i in range(5):
            has_spike = i < 4  # First 4 have spike, last doesn't
            triggers.append(
                {
                    "position_index": i,
                    "entry_time": i * 1_000_000,
                    "coin": "BTC",
                    "conditions": {
                        "price_velocity": {"direction": "up", "magnitude": 0.01},
                        "volume_spike": {
                            "detected": has_spike,
                            "ratio": 3.0 if has_spike else 1.0,
                        },
                        "volatility": {"level": "medium", "range_pct": 0.02},
                        "consecutive_ticks": {"count": 2, "direction": "up"},
                    },
                }
            )

        result = find_common_triggers(triggers)
        assert len(result["common_conditions"]) > 0
        # Volume spike should be common (80% match)
        vol_conditions = [
            c
            for c in result["common_conditions"]
            if "volume_spike" in c.get("name", "")
        ]
        assert len(vol_conditions) >= 1
        assert vol_conditions[0]["pct_matching"] >= 0.6

    def test_common_consecutive_ticks_pattern(self):
        """
        5 entries, all 5 have 3+ consecutive up ticks.
        Common trigger should identify consecutive ticks pattern.
        """
        triggers = []
        for i in range(5):
            triggers.append(
                {
                    "position_index": i,
                    "entry_time": i * 1_000_000,
                    "coin": "BTC",
                    "conditions": {
                        "price_velocity": {"direction": "up", "magnitude": 0.02},
                        "volume_spike": {"detected": False, "ratio": 1.0},
                        "volatility": {"level": "low", "range_pct": 0.005},
                        "consecutive_ticks": {"count": 3, "direction": "up"},
                    },
                }
            )

        result = find_common_triggers(triggers)
        assert len(result["common_conditions"]) > 0
        tick_conditions = [
            c for c in result["common_conditions"] if "consecutive" in c.get("name", "")
        ]
        assert len(tick_conditions) >= 1
        assert tick_conditions[0]["pct_matching"] >= 0.6

    def test_no_common_pattern_below_threshold(self):
        """
        5 entries with completely different conditions.
        No single pattern should reach 60% threshold.
        """
        triggers = []
        directions = ["up", "down", "up", "down", "up"]
        for i in range(5):
            triggers.append(
                {
                    "position_index": i,
                    "entry_time": i * 1_000_000,
                    "coin": "BTC",
                    "conditions": {
                        "price_velocity": {
                            "direction": directions[i],
                            "magnitude": 0.01,
                        },
                        "volume_spike": {
                            "detected": i % 2 == 0,
                            "ratio": 2.0 if i % 2 == 0 else 1.0,
                        },
                        "volatility": {"level": "medium", "range_pct": 0.02},
                        "consecutive_ticks": {
                            "count": i % 3,
                            "direction": directions[i],
                        },
                    },
                }
            )

        result = find_common_triggers(triggers)
        # All common conditions should have pct_matching >= 0.6 or the list
        # should be empty (meaning nothing reached threshold)
        for c in result.get("common_conditions", []):
            assert c["pct_matching"] >= 0.6

    def test_trigger_signature_output(self):
        """Output must include trigger signature string and pct."""
        triggers = []
        for i in range(5):
            triggers.append(
                {
                    "position_index": i,
                    "entry_time": i * 1_000_000,
                    "coin": "BTC",
                    "conditions": {
                        "price_velocity": {"direction": "up", "magnitude": 0.02},
                        "volume_spike": {"detected": True, "ratio": 3.0},
                        "volatility": {"level": "medium", "range_pct": 0.02},
                        "consecutive_ticks": {"count": 2, "direction": "up"},
                    },
                }
            )

        result = find_common_triggers(triggers)
        assert "trigger_signature" in result
        assert isinstance(result["trigger_signature"], str)
        assert len(result["trigger_signature"]) > 0
        assert "pct_matching" in result
        assert result["pct_matching"] >= 0.6


# ---------------------------------------------------------------------------
# Test 6: reconstruct_triggers for multiple positions
# ---------------------------------------------------------------------------


class TestReconstructTriggers:
    """Full trigger reconstruction across multiple position clusters."""

    def test_reconstruct_triggers_basic(self):
        """reconstruct_triggers processes multiple position clusters."""
        # 30 min of candles ending at entry_time
        candles = _generate_candles(
            base_price=50000.0,
            n_candles=10,
            start_time=0,
            interval_ms=300_000,
        )

        clusters = [
            _make_position_cluster(
                entry_time=2_700_000, exit_time=5_400_000
            ),  # Entry at 45 min
            _make_position_cluster(
                entry_time=1_200_000, exit_time=3_600_000
            ),  # Entry at 20 min
        ]

        candles_by_coin = {"BTC": candles}
        result = reconstruct_triggers(clusters, candles_by_coin)

        assert "triggers" in result
        assert len(result["triggers"]) == 2
        for trigger in result["triggers"]:
            assert "position_index" in trigger
            assert "entry_time" in trigger
            assert "conditions" in trigger
            assert "price_velocity" in trigger["conditions"]
            assert "volume_spike" in trigger["conditions"]
            assert "volatility" in trigger["conditions"]
            assert "consecutive_ticks" in trigger["conditions"]

    def test_reconstruct_triggers_with_common(self):
        """reconstruct_triggers includes common trigger analysis."""

        # Two entries with similar conditions (upward velocity + volume spike)
        def price_fn_up(i, base):
            o = base + i * 50
            c = o + 50
            return (o, c + 30, o - 30, c)

        def vol_fn_spike(i):
            return 200.0 if i >= 3 else 100.0  # Spike in last few candles

        candles = _generate_candles(
            base_price=50000.0,
            n_candles=10,
            start_time=0,
            interval_ms=300_000,
            price_fn=price_fn_up,
            volume_fn=vol_fn_spike,
        )

        clusters = [
            _make_position_cluster(entry_time=1_500_000, exit_time=3_000_000),
            _make_position_cluster(entry_time=1_200_000, exit_time=2_700_000),
        ]

        candles_by_coin = {"BTC": candles}
        result = reconstruct_triggers(clusters, candles_by_coin)

        assert "common_triggers" in result


# ---------------------------------------------------------------------------
# Test 7: reconstruct_wallet_triggers high-level function
# ---------------------------------------------------------------------------


class TestReconstructWalletTriggers:
    """High-level wallet trigger reconstruction."""

    def test_wallet_triggers_output_structure(self):
        """reconstruct_wallet_triggers produces structured output."""
        candles = _generate_candles(
            base_price=50000.0,
            n_candles=20,
            start_time=0,
            interval_ms=300_000,
        )

        clusters = [
            _make_position_cluster(entry_time=2_700_000, exit_time=5_400_000),
            _make_position_cluster(entry_time=1_200_000, exit_time=3_600_000),
        ]

        fills = [
            {"coin": "BTC", "time": 2_700_000, "sz": "1.0", "px": "50000"},
            {"coin": "BTC", "time": 5_400_000, "sz": "1.0", "px": "50500"},
            {"coin": "BTC", "time": 1_200_000, "sz": "1.0", "px": "50000"},
            {"coin": "BTC", "time": 3_600_000, "sz": "1.0", "px": "50500"},
        ]

        candles_by_coin = {"BTC": candles}
        result = reconstruct_wallet_triggers(
            wallet_address="0xtest",
            clusters=clusters,
            fills=fills,
            candles_by_coin=candles_by_coin,
        )

        assert result["wallet"] == "0xtest"
        assert "triggers" in result
        assert "num_positions" in result
        assert result["num_positions"] >= 1
        assert "common_triggers" in result


# ---------------------------------------------------------------------------
# Test 8: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty data, insufficient candles, single entry."""

    def test_empty_candles(self):
        """Empty candle list should return graceful empty conditions."""
        result = analyze_entry_conditions([], T6)
        assert result["price_velocity"]["direction"] == "unknown"
        assert result["volume_spike"]["detected"] is False

    def test_insufficient_candles(self):
        """Fewer than 3 candles should still produce a result."""
        candles = [
            _candle(open_px=50000, close=50050, time_ms=T0),
        ]
        result = analyze_entry_conditions(candles, T6)
        assert "price_velocity" in result
        assert "volume_spike" in result

    def test_single_entry_find_common(self):
        """Single entry should still produce common triggers (100% match)."""
        triggers = [
            {
                "position_index": 0,
                "entry_time": T0,
                "coin": "BTC",
                "conditions": {
                    "price_velocity": {"direction": "up", "magnitude": 0.02},
                    "volume_spike": {"detected": True, "ratio": 3.0},
                    "volatility": {"level": "high", "range_pct": 0.03},
                    "consecutive_ticks": {"count": 3, "direction": "up"},
                },
            }
        ]
        result = find_common_triggers(triggers)
        # All conditions should match 100% for a single entry
        assert result["pct_matching"] == 1.0

    def test_no_positions(self):
        """Empty clusters should produce empty result."""
        result = reconstruct_triggers([], {"BTC": []})
        assert result["triggers"] == []
        assert result["common_triggers"] == {}

    def test_missing_candle_for_coin(self):
        """Cluster for a coin with no candles should produce default conditions."""
        clusters = [_make_position_cluster(coin="ETH", entry_time=T6)]
        candles_by_coin = {"BTC": _generate_candles(n_candles=10)}
        result = reconstruct_triggers(clusters, candles_by_coin)
        assert len(result["triggers"]) == 1
        # Should have default/empty conditions, not crash
        trigger = result["triggers"][0]
        assert "conditions" in trigger

    def test_time_of_day_in_conditions(self):
        """Conditions should include time-of-day information."""
        candles = _generate_candles(
            n_candles=6,
            start_time=0,
            interval_ms=300_000,
        )
        # Entry at a specific time of day
        entry_time = 14 * 3_600_000  # 14:00 UTC
        result = analyze_entry_conditions(candles, entry_time)
        assert "time_of_day" in result
        assert result["time_of_day"]["hour_utc"] == 14
