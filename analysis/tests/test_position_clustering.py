"""Tests for position_clustering module.

Validates fill→position cycle clustering using synthetic Hyperliquid fill data.
Covers: full open→close, partial closes, scale-ins, direction reversals,
multiple markets, edge cases (<10 trades, empty fills).
"""

import pytest
import numpy as np

from analysis.position_clustering import (
    cluster_fills,
    cluster_wallet_fills,
    PositionCluster,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic fills
# ---------------------------------------------------------------------------


def _fill(
    coin: str = "BTC",
    side: str = "B",
    px: str = "50000.0",
    sz: str = "1.0",
    fee: str = "0.50",
    closed_pnl: str = "0.0",
    time: int = 1_700_000_000_000,
    direction: str = "Open Long",
    start_position: str = "0.0",
    hash_: str = "0xhash",
) -> dict:
    """Build a single fill dict matching Hyperliquid schema."""
    return {
        "coin": coin,
        "side": side,
        "px": px,
        "sz": sz,
        "fee": fee,
        "closedPnl": closed_pnl,
        "time": time,
        "dir": direction,
        "hash": hash_,
        "startPosition": start_position,
    }


# Timestamps (ms) spaced 60s apart for readability
T0 = 1_700_000_000_000
T1 = T0 + 60_000
T2 = T1 + 60_000
T3 = T2 + 60_000
T4 = T3 + 60_000
T5 = T4 + 60_000
T6 = T5 + 60_000
T7 = T6 + 60_000
T8 = T7 + 60_000
T9 = T8 + 60_000


# ---------------------------------------------------------------------------
# Test 1: Full open→close position (simple long)
# ---------------------------------------------------------------------------


class TestFullOpenClose:
    """Single entry fill followed by single exit fill."""

    def test_simple_long_position(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="50000",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="5.0",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="51000",
                sz="1.0",
                start_position="1.0",
                closed_pnl="1000.0",
                time=T2,
                fee="5.1",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["direction"] == "long"
        assert c["coin"] == "BTC"
        assert len(c["entry_fills"]) == 1
        assert len(c["exit_fills"]) == 1
        assert c["entry_price"] == 50000.0
        assert c["exit_price"] == 51000.0
        assert c["realized_pnl"] == 1000.0
        assert c["fees_paid"] == pytest.approx(10.1)
        assert c["entry_time"] == T0
        assert c["exit_time"] == T2
        assert c["total_size"] == pytest.approx(1.0)

    def test_simple_short_position(self):
        fills = [
            _fill(
                direction="Open Short",
                side="S",
                px="50000",
                sz="2.0",
                start_position="0.0",
                time=T0,
                fee="10.0",
            ),
            _fill(
                direction="Close Short",
                side="B",
                px="48000",
                sz="2.0",
                start_position="-2.0",
                closed_pnl="4000.0",
                time=T3,
                fee="9.6",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["direction"] == "short"
        assert c["entry_price"] == 50000.0
        assert c["exit_price"] == 48000.0
        assert c["realized_pnl"] == 4000.0
        assert c["total_size"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 2: Partial closes (multiple exit fills)
# ---------------------------------------------------------------------------


class TestPartialClose:
    """Position closed over multiple exit fills."""

    def test_two_partial_closes(self):
        fills = [
            # Open long 1.0 BTC
            _fill(
                direction="Open Long",
                side="B",
                px="50000",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="5.0",
            ),
            # Partial close 0.4
            _fill(
                direction="Close Long",
                side="S",
                px="52000",
                sz="0.4",
                start_position="1.0",
                closed_pnl="800.0",
                time=T2,
                fee="2.08",
            ),
            # Partial close 0.6
            _fill(
                direction="Close Long",
                side="S",
                px="53000",
                sz="0.6",
                start_position="0.6",
                closed_pnl="1800.0",
                time=T3,
                fee="3.18",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["direction"] == "long"
        assert len(c["entry_fills"]) == 1
        assert len(c["exit_fills"]) == 2
        # VWAP exit: (0.4 * 52000 + 0.6 * 53000) / (0.4 + 0.6) = 52600
        assert c["exit_price"] == pytest.approx(52600.0)
        assert c["entry_price"] == pytest.approx(50000.0)
        assert c["realized_pnl"] == pytest.approx(2600.0)  # 800 + 1800
        assert c["fees_paid"] == pytest.approx(10.26)  # 5.0 + 2.08 + 3.18

    def test_three_partial_closes(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="3.0",
                start_position="0.0",
                time=T0,
                fee="0.3",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="110",
                sz="1.0",
                start_position="3.0",
                closed_pnl="10.0",
                time=T1,
                fee="0.11",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="105",
                sz="1.0",
                start_position="2.0",
                closed_pnl="5.0",
                time=T2,
                fee="0.105",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="115",
                sz="1.0",
                start_position="1.0",
                closed_pnl="15.0",
                time=T3,
                fee="0.115",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert len(c["exit_fills"]) == 3
        assert c["realized_pnl"] == pytest.approx(30.0)
        # VWAP exit: (1*110 + 1*105 + 1*115) / 3 = 110
        assert c["exit_price"] == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# Test 3: Scale-ins (multiple entry fills within 5 min)
# ---------------------------------------------------------------------------


class TestScaleIn:
    """Multiple open fills within 5 minutes grouped as one entry."""

    def test_two_scale_ins_within_5min(self):
        # Two entry fills 2 minutes apart (within 5-min window)
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="50000",
                sz="0.5",
                start_position="0.0",
                time=T0,
                fee="2.5",
            ),
            _fill(
                direction="Open Long",
                side="B",
                px="50500",
                sz="0.5",
                start_position="0.5",
                time=T0 + 120_000,
                fee="2.525",
            ),  # +2 min
            _fill(
                direction="Close Long",
                side="S",
                px="52000",
                sz="1.0",
                start_position="1.0",
                closed_pnl="2000.0",
                time=T4,
                fee="5.2",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert len(c["entry_fills"]) == 2
        assert len(c["exit_fills"]) == 1
        # VWAP entry: (0.5*50000 + 0.5*50500) / 1.0 = 50250
        assert c["entry_price"] == pytest.approx(50250.0)
        assert c["exit_price"] == pytest.approx(52000.0)
        assert c["total_size"] == pytest.approx(1.0)

    def test_scale_in_not_grouped_after_5min(self):
        # Two entry fills 6 minutes apart — scale_in flag should be False
        # but fills are still grouped in same cluster
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="50000",
                sz="0.5",
                start_position="0.0",
                time=T0,
                fee="2.5",
            ),
            _fill(
                direction="Open Long",
                side="B",
                px="50500",
                sz="0.5",
                start_position="0.5",
                time=T0 + 360_000,
                fee="2.525",
            ),  # +6 min
            # Close must come AFTER the second open (T0+360000 = T6)
            _fill(
                direction="Close Long",
                side="S",
                px="52000",
                sz="1.0",
                start_position="1.0",
                closed_pnl="2000.0",
                time=T7,
                fee="5.2",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        # Both open fills are still entry fills (scale-in status is about metadata)
        assert len(c["entry_fills"]) == 2
        assert c["entry_price"] == pytest.approx(50250.0)
        # scale_in should be False because >5 min between entry fills
        assert c["scale_in"] is False

    def test_short_with_scale_in(self):
        fills = [
            _fill(
                direction="Open Short",
                side="S",
                px="60000",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="6.0",
            ),
            _fill(
                direction="Open Short",
                side="S",
                px="60500",
                sz="1.0",
                start_position="-1.0",
                time=T0 + 180_000,
                fee="6.05",
            ),  # +3 min
            _fill(
                direction="Close Short",
                side="B",
                px="59000",
                sz="2.0",
                start_position="-2.0",
                closed_pnl="22000.0",
                time=T4,
                fee="11.8",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["direction"] == "short"
        assert len(c["entry_fills"]) == 2
        # VWAP entry: (1*60000 + 1*60500) / 2.0 = 60250
        assert c["entry_price"] == pytest.approx(60250.0)


# ---------------------------------------------------------------------------
# Test 4: Direction reversals
# ---------------------------------------------------------------------------


class TestDirectionReversal:
    """Close long then immediately open short creates two positions."""

    def test_long_then_short_reversal(self):
        fills = [
            # Long position
            _fill(
                direction="Open Long",
                side="B",
                px="50000",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="5.0",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="49000",
                sz="1.0",
                start_position="1.0",
                closed_pnl="-1000.0",
                time=T2,
                fee="4.9",
            ),
            # Immediate reversal to short
            _fill(
                direction="Open Short",
                side="S",
                px="49000",
                sz="1.0",
                start_position="0.0",
                time=T2 + 10_000,
                fee="4.9",
            ),
            _fill(
                direction="Close Short",
                side="B",
                px="48000",
                sz="1.0",
                start_position="-1.0",
                closed_pnl="1000.0",
                time=T5,
                fee="4.8",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 2

        c0 = clusters[0]
        assert c0["direction"] == "long"
        assert c0["realized_pnl"] == -1000.0
        assert c0["entry_time"] == T0
        assert c0["exit_time"] == T2

        c1 = clusters[1]
        assert c1["direction"] == "short"
        assert c1["realized_pnl"] == 1000.0
        assert c1["entry_time"] == T2 + 10_000
        assert c1["exit_time"] == T5

    def test_short_then_long_reversal(self):
        fills = [
            _fill(
                direction="Open Short",
                side="S",
                px="30000",
                sz="0.5",
                start_position="0.0",
                time=T0,
                fee="1.5",
            ),
            _fill(
                direction="Close Short",
                side="B",
                px="31000",
                sz="0.5",
                start_position="-0.5",
                closed_pnl="-500.0",
                time=T1,
                fee="1.55",
            ),
            _fill(
                direction="Open Long",
                side="B",
                px="31000",
                sz="0.5",
                start_position="0.0",
                time=T1 + 5_000,
                fee="1.55",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="32000",
                sz="0.5",
                start_position="0.5",
                closed_pnl="500.0",
                time=T3,
                fee="1.6",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 2
        assert clusters[0]["direction"] == "short"
        assert clusters[0]["realized_pnl"] == -500.0
        assert clusters[1]["direction"] == "long"
        assert clusters[1]["realized_pnl"] == 500.0


# ---------------------------------------------------------------------------
# Test 5: Multiple positions across different markets
# ---------------------------------------------------------------------------


class TestMultipleMarkets:
    """Fills for different coins produce independent clusters."""

    def test_btc_and_eth_positions(self):
        fills = [
            # BTC long
            _fill(
                coin="BTC",
                direction="Open Long",
                side="B",
                px="50000",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="5.0",
            ),
            _fill(
                coin="BTC",
                direction="Close Long",
                side="S",
                px="51000",
                sz="1.0",
                start_position="1.0",
                closed_pnl="1000.0",
                time=T2,
                fee="5.1",
            ),
            # ETH short
            _fill(
                coin="ETH",
                direction="Open Short",
                side="S",
                px="3000",
                sz="2.0",
                start_position="0.0",
                time=T0,
                fee="0.6",
            ),
            _fill(
                coin="ETH",
                direction="Close Short",
                side="B",
                px="2900",
                sz="2.0",
                start_position="-2.0",
                closed_pnl="200.0",
                time=T3,
                fee="0.58",
            ),
        ]
        clusters = cluster_fills(fills)

        btc = [c for c in clusters if c["coin"] == "BTC"]
        eth = [c for c in clusters if c["coin"] == "ETH"]
        assert len(btc) == 1
        assert len(eth) == 1
        assert btc[0]["direction"] == "long"
        assert eth[0]["direction"] == "short"


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Graceful handling of small wallets and empty data."""

    def test_empty_fills(self):
        clusters = cluster_fills([])
        assert clusters == []

    def test_single_fill(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="0.1",
            ),
        ]
        clusters = cluster_fills(fills)
        # Open position with no close — should produce an open cluster
        assert len(clusters) == 1
        c = clusters[0]
        assert c["direction"] == "long"
        assert c["exit_fills"] == []
        assert c["exit_time"] is None
        assert c["exit_price"] is None
        assert c["realized_pnl"] == 0.0

    def test_fewer_than_10_trades(self):
        """Wallets with <10 trades handled gracefully (VAL-ANALYSIS-009)."""
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="0.1",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="105",
                sz="1.0",
                start_position="1.0",
                closed_pnl="5.0",
                time=T1,
                fee="0.105",
            ),
            _fill(
                direction="Open Short",
                side="S",
                px="200",
                sz="0.5",
                start_position="0.0",
                time=T2,
                fee="0.1",
            ),
            _fill(
                direction="Close Short",
                side="B",
                px="195",
                sz="0.5",
                start_position="-0.5",
                closed_pnl="2.5",
                time=T3,
                fee="0.0975",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 2
        # Should not crash or produce unreliable results
        for c in clusters:
            assert "direction" in c
            assert "entry_price" in c
            assert "realized_pnl" in c

    def test_cluster_wallet_fills_with_metadata(self):
        """cluster_wallet_fills adds wallet metadata to results."""
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="0.1",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="110",
                sz="1.0",
                start_position="1.0",
                closed_pnl="10.0",
                time=T1,
                fee="0.11",
            ),
        ]
        result = cluster_wallet_fills("0xtestwallet", fills)
        assert result["wallet"] == "0xtestwallet"
        assert result["total_fills"] == 2
        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["realized_pnl"] == 10.0
        assert result["num_clusters"] == 1

    def test_cluster_wallet_fills_insufficient_data(self):
        """Wallets with very few fills get flagged as insufficient_data."""
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="0.1",
            ),
        ]
        result = cluster_wallet_fills("0xsmall", fills)
        assert result["insufficient_data"] is True
        assert result["clusters"][0]["exit_price"] is None


# ---------------------------------------------------------------------------
# Test 7: VWAP computation
# ---------------------------------------------------------------------------


class TestVWAPComputation:
    """Verify volume-weighted average price correctness."""

    def test_vwap_entry_with_different_sizes(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="2.0",
                start_position="0.0",
                time=T0,
                fee="0.2",
            ),
            _fill(
                direction="Open Long",
                side="B",
                px="110",
                sz="3.0",
                start_position="2.0",
                time=T0 + 60_000,
                fee="0.33",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="120",
                sz="5.0",
                start_position="5.0",
                closed_pnl="100.0",
                time=T3,
                fee="0.6",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        # VWAP entry: (2*100 + 3*110) / 5 = (200 + 330) / 5 = 106
        assert c["entry_price"] == pytest.approx(106.0)
        assert c["exit_price"] == pytest.approx(120.0)
        assert c["total_size"] == pytest.approx(5.0)

    def test_vwap_exit_with_different_sizes(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="200",
                sz="4.0",
                start_position="0.0",
                time=T0,
                fee="0.8",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="210",
                sz="1.0",
                start_position="4.0",
                closed_pnl="10.0",
                time=T1,
                fee="0.21",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="220",
                sz="3.0",
                start_position="3.0",
                closed_pnl="60.0",
                time=T2,
                fee="0.66",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["entry_price"] == pytest.approx(200.0)
        # VWAP exit: (1*210 + 3*220) / 4 = (210 + 660) / 4 = 217.5
        assert c["exit_price"] == pytest.approx(217.5)


# ---------------------------------------------------------------------------
# Test 8: Scale-in detection metadata
# ---------------------------------------------------------------------------


class TestScaleInDetection:
    """Verify scale_in flag is set on positions with multiple entry fills."""

    def test_scale_in_flagged(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="0.1",
            ),
            _fill(
                direction="Open Long",
                side="B",
                px="102",
                sz="1.0",
                start_position="1.0",
                time=T0 + 120_000,
                fee="0.102",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="110",
                sz="2.0",
                start_position="2.0",
                closed_pnl="20.0",
                time=T3,
                fee="0.22",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        assert clusters[0]["scale_in"] is True

    def test_no_scale_in_flagged(self):
        fills = [
            _fill(
                direction="Open Long",
                side="B",
                px="100",
                sz="1.0",
                start_position="0.0",
                time=T0,
                fee="0.1",
            ),
            _fill(
                direction="Close Long",
                side="S",
                px="110",
                sz="1.0",
                start_position="1.0",
                closed_pnl="10.0",
                time=T2,
                fee="0.11",
            ),
        ]
        clusters = cluster_fills(fills)
        assert len(clusters) == 1
        assert clusters[0]["scale_in"] is False


# ---------------------------------------------------------------------------
# Test 9: Orphaned fills handling
# ---------------------------------------------------------------------------


class TestOrphanedFills:
    """Fills that don't fit normal patterns are handled gracefully."""

    def test_close_without_open(self):
        """A Close fill without a preceding Open should not crash."""
        fills = [
            _fill(
                direction="Close Long",
                side="S",
                px="110",
                sz="1.0",
                start_position="1.0",
                closed_pnl="10.0",
                time=T0,
                fee="0.11",
            ),
        ]
        # Should not crash; the orphaned close is handled gracefully
        clusters = cluster_fills(fills)
        # Orphaned close should still create a cluster
        assert len(clusters) == 1
        assert clusters[0]["direction"] == "long"
        assert len(clusters[0]["exit_fills"]) == 1
        assert clusters[0]["entry_fills"] == []
