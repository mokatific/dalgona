"""Tests for wallet_metrics.py

Uses synthetic fill and position cluster data to verify all Bulk.Trade
metrics. Tests cover:
  - Basic metrics (total_trades, win_rate, avg_hold_time, etc.)
  - clip_size_consistency (% of fills within ±10% of median)
  - fill_interval_stats (median gap, pct_sub_30s)
  - scale_in_count (multi-fill entries)
  - active_hours / coverage_pct (UTC hour coverage)
  - fee_adjusted_pnl and fee_adjusted_win_rate
  - sharpe_ratio, max_drawdown, profit_factor
  - PnL distribution (mean, median, max winner, max loser, skewness)
  - Edge cases: <10 trades, empty fills, single market
"""

import math

import pytest

from analysis.position_clustering import cluster_fills
from analysis.wallet_metrics import compute_wallet_metrics


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------


def _fill(
    coin="BTC",
    side="B",
    px=50000.0,
    sz=1.0,
    fee=0.50,
    closedPnl=0.0,
    time_ms=0,
    dir_str="Open Long",
    start_position=0.0,
    hash_val="0xabc",
):
    """Create a single fill dict matching Hyperliquid schema."""
    return {
        "coin": coin,
        "side": side,
        "px": str(px),
        "sz": str(sz),
        "fee": str(fee),
        "closedPnl": str(closedPnl),
        "time": time_ms,
        "dir": dir_str,
        "hash": hash_val,
        "startPosition": str(start_position),
    }


def _make_long_position(
    coin="BTC",
    entry_px=50000.0,
    exit_px=51000.0,
    sz=1.0,
    fee=0.50,
    entry_time=0,
    exit_time=3600_000,  # 1 hour
    closedPnl=1000.0,
    start_pos=0.0,
):
    """Create fills for a complete long position (open → close)."""
    fills = [
        _fill(
            coin=coin,
            side="B",
            px=entry_px,
            sz=sz,
            fee=fee,
            closedPnl=0.0,
            time_ms=entry_time,
            dir_str="Open Long",
            start_position=start_pos,
        ),
        _fill(
            coin=coin,
            side="S",
            px=exit_px,
            sz=sz,
            fee=fee,
            closedPnl=closedPnl,
            time_ms=exit_time,
            dir_str="Close Long",
            start_position=sz,
        ),
    ]
    return fills


def _make_short_position(
    coin="ETH",
    entry_px=3000.0,
    exit_px=2900.0,
    sz=5.0,
    fee=0.30,
    entry_time=0,
    exit_time=7200_000,  # 2 hours
    closedPnl=500.0,
    start_pos=0.0,
):
    """Create fills for a complete short position."""
    fills = [
        _fill(
            coin=coin,
            side="S",
            px=entry_px,
            sz=sz,
            fee=fee,
            closedPnl=0.0,
            time_ms=entry_time,
            dir_str="Open Short",
            start_position=start_pos,
        ),
        _fill(
            coin=coin,
            side="B",
            px=exit_px,
            sz=sz,
            fee=fee,
            closedPnl=closedPnl,
            time_ms=exit_time,
            dir_str="Close Short",
            start_position=-sz,
        ),
    ]
    return fills


# ---------------------------------------------------------------------------
# Test 1: Basic metrics — total_trades, win_rate, avg_hold_time, markets, direction
# ---------------------------------------------------------------------------


class TestBasicMetrics:
    """Test core metrics from position cluster data."""

    def test_total_trades_and_win_rate(self):
        """6 closed positions: 4 winners, 2 losers → win_rate = 4/6."""
        fills = []
        base_time = 0
        # 4 winning longs on BTC
        for i in range(4):
            fills.extend(
                _make_long_position(
                    coin="BTC",
                    entry_px=50000.0,
                    exit_px=51000.0,
                    sz=1.0,
                    fee=1.0,
                    entry_time=base_time,
                    exit_time=base_time + 3600_000,
                    closedPnl=500.0,
                    start_pos=0.0,
                )
            )
            base_time += 7200_000

        # 2 losing shorts on ETH
        for i in range(2):
            fills.extend(
                _make_short_position(
                    coin="ETH",
                    entry_px=3000.0,
                    exit_px=3100.0,
                    sz=5.0,
                    fee=1.0,
                    entry_time=base_time,
                    exit_time=base_time + 3600_000,
                    closedPnl=-500.0,
                    start_pos=0.0,
                )
            )
            base_time += 7200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert metrics["total_trades"] == 6
        assert abs(metrics["win_rate"] - 4.0 / 6.0) < 1e-6
        assert metrics["markets_traded"] == ["BTC", "ETH"]
        assert metrics["primary_market"] == "BTC"  # 4 trades on BTC
        assert metrics["preferred_direction"] == "long"  # 4 longs vs 2 shorts

    def test_avg_hold_time(self):
        """Three positions: 1h, 2h, 3h → avg = 2h."""
        fills = []
        # Position 1: 1 hour
        fills.extend(
            _make_long_position(
                entry_time=0,
                exit_time=3600_000,
                closedPnl=100.0,
            )
        )
        # Position 2: 2 hours
        fills.extend(
            _make_long_position(
                entry_time=7_200_000,
                exit_time=7_200_000 + 7_200_000,
                closedPnl=200.0,
            )
        )
        # Position 3: 3 hours
        fills.extend(
            _make_long_position(
                entry_time=15_000_000,
                exit_time=15_000_000 + 10_800_000,
                closedPnl=300.0,
            )
        )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # avg_hold_time in hours: (1 + 2 + 3) / 3 = 2.0
        assert abs(metrics["avg_hold_time_hours"] - 2.0) < 1e-4

    def test_avg_pnl_per_trade(self):
        """PnLs: +100, -50, +200 → avg = 250/3 ≈ 83.33."""
        fills = []
        base = 0
        for pnl in [100.0, -50.0, 200.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        expected_avg = 250.0 / 3.0
        assert abs(metrics["avg_pnl_per_trade"] - expected_avg) < 1e-4


# ---------------------------------------------------------------------------
# Test 2: clip_size_consistency
# ---------------------------------------------------------------------------


class TestClipSizeConsistency:
    """Test % of fills within ±10% of median size."""

    def test_fixed_clip_high_consistency(self):
        """All fills are exactly 1.0 size → consistency should be 100%."""
        fills = []
        base = 0
        for _ in range(10):
            fills.extend(
                _make_long_position(
                    sz=1.0,
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=50.0,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # All fills are size 1.0 → 100% within ±10% of median
        assert metrics["clip_size_consistency"] == 1.0

    def test_varied_clip_lower_consistency(self):
        """Mix of sizes: 1.0 (×8) and 5.0 (×2) → median is 1.0, 5.0 is far."""
        fills = []
        base = 0
        # 4 positions with sz=1.0 → 8 fills of size 1.0
        for _ in range(4):
            fills.extend(
                _make_long_position(
                    sz=1.0,
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=50.0,
                )
            )
            base += 7_200_000
        # 1 position with sz=5.0 → 2 fills of size 5.0
        fills.extend(
            _make_long_position(
                sz=5.0,
                entry_time=base,
                exit_time=base + 3600_000,
                closedPnl=250.0,
            )
        )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # Median fill size = 1.0; fills of 5.0 are outside ±10%
        # 8 out of 10 fills are size 1.0 → consistency = 0.8
        assert abs(metrics["clip_size_consistency"] - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Test 3: fill_interval_stats
# ---------------------------------------------------------------------------


class TestFillIntervalStats:
    """Test median gap and pct_sub_30s for bot detection."""

    def test_regular_intervals_above_30s(self):
        """Fills 60s apart → median gap 60s, pct_sub_30s = 0%."""
        fills = []
        for i in range(10):
            fills.append(
                _fill(
                    sz=0.1,
                    time_ms=i * 60_000,  # 60-second gaps
                    dir_str="Open Long",
                    start_position=0.0 if i == 0 else float(i) * 0.1,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        stats = metrics["fill_interval_stats"]
        assert stats["median_gap_seconds"] == 60.0
        assert stats["pct_sub_30s"] == 0.0

    def test_sub_30s_intervals_bot_like(self):
        """Fills 10s apart → median gap 10s, pct_sub_30s = 100%."""
        fills = []
        for i in range(10):
            fills.append(
                _fill(
                    sz=0.1,
                    time_ms=i * 10_000,  # 10-second gaps
                    dir_str="Open Long",
                    start_position=0.0 if i == 0 else float(i) * 0.1,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        stats = metrics["fill_interval_stats"]
        assert stats["median_gap_seconds"] == 10.0
        assert stats["pct_sub_30s"] == 1.0

    def test_mixed_intervals(self):
        """Mix of 10s and 120s gaps."""
        fills = []
        times = [0, 10_000, 130_000, 140_000, 260_000]
        for i, t in enumerate(times):
            fills.append(
                _fill(
                    sz=0.1,
                    time_ms=t,
                    dir_str="Open Long",
                    start_position=0.0 if i == 0 else float(i) * 0.1,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        stats = metrics["fill_interval_stats"]
        # Intervals: 10s, 120s, 10s, 120s → sorted: [10, 10, 120, 120] → median 65.0
        assert stats["median_gap_seconds"] == 65.0
        # 2 out of 4 intervals are < 30s → 50%
        assert abs(stats["pct_sub_30s"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: scale_in_count
# ---------------------------------------------------------------------------


class TestScaleInCount:
    """Test counting multi-fill entries (scale-ins)."""

    def test_single_and_multi_fill_entries(self):
        """2 single-fill entries + 1 scale-in (3 entry fills within 5 min)."""
        fills = []
        base = 0

        # Position 1: single fill entry → close
        fills.extend(
            _make_long_position(
                entry_time=base,
                exit_time=base + 3600_000,
                closedPnl=100.0,
            )
        )
        base += 7_200_000

        # Position 2: single fill entry → close
        fills.extend(
            _make_long_position(
                entry_time=base,
                exit_time=base + 3600_000,
                closedPnl=-50.0,
            )
        )
        base += 7_200_000

        # Position 3: scale-in with 3 entry fills within 5 min → close
        fills.append(
            _fill(
                px=50000.0,
                sz=0.5,
                time_ms=base,
                dir_str="Open Long",
                start_position=0.0,
            )
        )
        fills.append(
            _fill(
                px=50100.0,
                sz=0.3,
                time_ms=base + 60_000,  # 1 min later
                dir_str="Open Long",
                start_position=0.5,
            )
        )
        fills.append(
            _fill(
                px=50200.0,
                sz=0.2,
                time_ms=base + 120_000,  # 2 min later
                dir_str="Open Long",
                start_position=0.8,
            )
        )
        # Close the scaled-in position
        fills.append(
            _fill(
                px=51000.0,
                sz=1.0,
                time_ms=base + 3600_000,
                dir_str="Close Long",
                start_position=1.0,
                closedPnl=500.0,
            )
        )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # Only position 3 has scale_in=True
        assert metrics["scale_in_count"] == 1
        assert metrics["total_trades"] == 3


# ---------------------------------------------------------------------------
# Test 5: active_hours / coverage_pct
# ---------------------------------------------------------------------------


class TestActiveHours:
    """Test UTC hour coverage and bot detection."""

    def test_full_coverage_24h(self):
        """Fills in all 24 UTC hours → coverage 100%."""
        fills = []
        base = 0
        for h in range(24):
            # Each fill at the start of a different UTC hour
            fills.append(
                _fill(
                    time_ms=base + h * 3600_000,
                    dir_str="Open Long",
                    start_position=float(h) * 0.1,
                    sz=0.1,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert metrics["active_hours"] == list(range(24))
        assert abs(metrics["coverage_pct"] - 1.0) < 1e-6

    def test_partial_coverage(self):
        """Fills in 6 UTC hours → coverage = 6/24 = 25%."""
        fills = []
        active_hours = [2, 8, 10, 14, 18, 22]
        for i, h in enumerate(active_hours):
            fills.append(
                _fill(
                    time_ms=h * 3600_000,
                    dir_str="Open Long",
                    start_position=float(i) * 0.1,
                    sz=0.1,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert sorted(metrics["active_hours"]) == sorted(active_hours)
        assert abs(metrics["coverage_pct"] - 6.0 / 24.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 6: fee_adjusted_pnl and fee_adjusted_win_rate
# ---------------------------------------------------------------------------


class TestFeeAdjustedMetrics:
    """Test PnL after fee deduction."""

    def test_fee_adjusted_pnl(self):
        """
        Three trades (each position has entry + exit fill, so 2 × fee per trade):
          Trade 1: closedPnl=+100, fees=10+10=20
          Trade 2: closedPnl=-50,  fees=10+10=20
          Trade 3: closedPnl=+30,  fees=10+10=20
        fee_adjusted_pnl = (100 - 50 + 30) - (20 + 20 + 20) = 80 - 60 = 20
        """
        fills = []
        base = 0
        for pnl, fee in [(100.0, 10.0), (-50.0, 10.0), (30.0, 10.0)]:
            fills.extend(
                _make_long_position(
                    fee=fee,
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # fee_adjusted_pnl = sum(realized_pnl) - sum(fees_paid per cluster)
        # Each cluster has 2 fills with fee=10 → fees_paid = 20
        expected_fap = (100.0 - 50.0 + 30.0) - 60.0
        assert abs(metrics["fee_adjusted_pnl"] - expected_fap) < 1e-4

    def test_fee_adjusted_win_rate(self):
        """
        Trade 1: closedPnl=+100, fee=10 → net +90 (win)
        Trade 2: closedPnl=-50,  fee=10 → net -60 (loss)
        Trade 3: closedPnl=+5,   fee=10 → net -5  (loss after fees!)
        fee_adjusted_win_rate = 1/3
        """
        fills = []
        base = 0
        for pnl, fee in [(100.0, 10.0), (-50.0, 10.0), (5.0, 10.0)]:
            fills.extend(
                _make_long_position(
                    fee=fee,
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # Only trade 1 has fee-adjusted PnL > 0
        assert abs(metrics["fee_adjusted_win_rate"] - 1.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 7: sharpe_ratio, max_drawdown, profit_factor
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    """Test Sharpe ratio, max drawdown, and profit factor."""

    def test_profit_factor(self):
        """
        Wins: +200, +300 → gross_wins = 500
        Losses: -100, -50 → gross_losses = 150
        profit_factor = 500 / 150 ≈ 3.333
        """
        fills = []
        base = 0
        for pnl in [200.0, -100.0, 300.0, -50.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        expected_pf = 500.0 / 150.0
        assert abs(metrics["profit_factor"] - expected_pf) < 1e-4

    def test_max_drawdown(self):
        """
        Cumulative PnL: +100, +200, +100, +250, +50
        Drawdowns from peak:
          After trade 1: peak=100, drawdown=0
          After trade 2: peak=200, drawdown=0
          After trade 3: peak=200, drawdown=100
          After trade 4: peak=250, drawdown=0
          After trade 5: peak=250, drawdown=200
        Max drawdown = 200
        """
        fills = []
        base = 0
        for pnl in [100.0, 100.0, -100.0, 150.0, -200.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # Cumulative: 100, 200, 100, 250, 50
        # Peak:       100, 200, 200, 250, 250
        # DD:           0,   0, 100,   0, 200
        assert abs(metrics["max_drawdown"] - 200.0) < 1e-4

    def test_sharpe_ratio_positive(self):
        """All winning trades → positive Sharpe ratio."""
        fills = []
        base = 0
        for pnl in [100.0, 150.0, 120.0, 80.0, 200.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # Mean = 130, Std ≈ 43.0 → Sharpe ≈ 3.02
        assert metrics["sharpe_ratio"] > 0
        assert metrics["sharpe_ratio"] > 2.0  # Strong positive edge


# ---------------------------------------------------------------------------
# Test 8: PnL distribution
# ---------------------------------------------------------------------------


class TestPnlDistribution:
    """Test PnL distribution statistics."""

    def test_pnl_distribution_stats(self):
        """
        PnLs: -50, 10, 20, 30, 500
        mean = 102
        median = 20
        max_winner = 500
        max_loser = -50
        skewness: positive (right-skewed, heavy right tail from 500)
        """
        fills = []
        base = 0
        for pnl in [-50.0, 10.0, 20.0, 30.0, 500.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        dist = metrics["pnl_distribution"]
        assert abs(dist["mean"] - 102.0) < 1e-4
        assert abs(dist["median"] - 20.0) < 1e-4
        assert abs(dist["max_winner"] - 500.0) < 1e-4
        assert abs(dist["max_loser"] - (-50.0)) < 1e-4
        # Positive skewness: right tail is longer (500 vs -50)
        assert dist["skewness"] > 0


# ---------------------------------------------------------------------------
# Test 9: Edge case — few trades
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_few_trades(self):
        """Wallet with only 2 trades should still compute metrics."""
        fills = []
        fills.extend(
            _make_long_position(
                entry_time=0,
                exit_time=3600_000,
                closedPnl=100.0,
            )
        )
        fills.extend(
            _make_long_position(
                entry_time=7_200_000,
                exit_time=10_800_000,
                closedPnl=-50.0,
            )
        )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert metrics["total_trades"] == 2
        assert metrics["win_rate"] == 0.5
        assert metrics["sharpe_ratio"] is not None

    def test_empty_fills(self):
        """Empty fills should return safe defaults."""
        metrics = compute_wallet_metrics([], [])

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["clip_size_consistency"] == 0.0
        assert metrics["coverage_pct"] == 0.0

    def test_all_losses(self):
        """All losing trades: win_rate=0, profit_factor=0."""
        fills = []
        base = 0
        for pnl in [-100.0, -200.0, -50.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_all_wins(self):
        """All winning trades: win_rate=1, max_drawdown=0."""
        fills = []
        base = 0
        for pnl in [100.0, 200.0, 50.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=pnl,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert metrics["win_rate"] == 1.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["profit_factor"] is None  # No losses → undefined PF

    def test_avg_leverage(self):
        """avg_leverage computed from position notional sizes."""
        fills = []
        base = 0
        # Position 1: 1.0 BTC @ 50000 = $50,000 notional
        fills.extend(
            _make_long_position(
                sz=1.0,
                entry_px=50000.0,
                entry_time=base,
                exit_time=base + 3600_000,
                closedPnl=100.0,
            )
        )
        base += 7_200_000
        # Position 2: 2.0 BTC @ 50000 = $100,000 notional
        fills.extend(
            _make_long_position(
                sz=2.0,
                entry_px=50000.0,
                entry_time=base,
                exit_time=base + 3600_000,
                closedPnl=200.0,
            )
        )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        # avg notional = (50000 + 100000) / 2 = 75000
        assert abs(metrics["avg_notional"] - 75000.0) < 1e-4

    def test_mixed_direction_preferred(self):
        """3 longs, 3 shorts → preferred_direction = 'mixed'."""
        fills = []
        base = 0
        for _ in range(3):
            fills.extend(
                _make_long_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=50.0,
                )
            )
            base += 7_200_000
            fills.extend(
                _make_short_position(
                    entry_time=base,
                    exit_time=base + 3600_000,
                    closedPnl=50.0,
                )
            )
            base += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)

        assert metrics["preferred_direction"] == "mixed"
