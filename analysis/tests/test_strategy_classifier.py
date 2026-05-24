"""Tests for strategy_classifier.py

Uses synthetic wallet metrics and position cluster data to verify strategy
classification. Tests cover:
  - Momentum scalper classification (short hold, high win rate, directional)
  - Trend follower classification (long hold, high profit factor, skewed PnL)
  - Mean reversion classification (mixed direction, scale-ins, moderate hold)
  - LP consumer classification (very short hold, very high win rate, 24/7 bot)
  - Grid classification (consistent clips, 24/7, multiple markets, mixed dir)
  - Unknown / insufficient data classification
  - Confidence scores and evidence reporting
  - Edge cases: empty data, low confidence fallback
"""

import pytest

from analysis.position_clustering import cluster_fills
from analysis.wallet_metrics import compute_wallet_metrics
from analysis.strategy_classifier import classify_wallet, classify_strategies


# ---------------------------------------------------------------------------
# Helpers to build synthetic fills
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
    exit_time=3600_000,
    closedPnl=1000.0,
    start_pos=0.0,
):
    """Create fills for a complete long position."""
    return [
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
            start_position=start_pos + sz,
        ),
    ]


def _make_short_position(
    coin="ETH",
    entry_px=3000.0,
    exit_px=2900.0,
    sz=5.0,
    fee=0.30,
    entry_time=0,
    exit_time=7200_000,
    closedPnl=500.0,
    start_pos=0.0,
):
    """Create fills for a complete short position."""
    return [
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
            start_position=start_pos - sz,
        ),
    ]


# ---------------------------------------------------------------------------
# Test 1: Momentum Scalper classification
# ---------------------------------------------------------------------------


class TestMomentumScalper:
    """Wallet with short holds, high win rate, directional trades."""

    def test_momentum_scalper_classified(self):
        """
        Momentum scalper profile:
        - Short hold times (< 2 hours avg)
        - High win rate (> 60%)
        - Many trades (> 20)
        - Directional preference (mostly longs)
        - Moderate clip consistency
        """
        fills = []
        base_time = 0
        # 15 winning long scalps (45 min hold)
        for _ in range(15):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=50200.0,
                    sz=1.0,
                    fee=1.0,
                    entry_time=base_time,
                    exit_time=base_time + 2_700_000,  # 45 min
                    closedPnl=200.0,
                )
            )
            base_time += 5_400_000  # 1.5h between entries

        # 3 losing long scalps
        for _ in range(3):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=49900.0,
                    sz=1.0,
                    fee=1.0,
                    entry_time=base_time,
                    exit_time=base_time + 2_700_000,
                    closedPnl=-100.0,
                )
            )
            base_time += 5_400_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] == "momentum_scalper"
        assert result["confidence"] > 0.5
        assert len(result["evidence"]) > 0
        # Evidence should mention hold time and win rate
        evidence_str = " ".join(result["evidence"])
        assert "hold_time" in evidence_str or "win_rate" in evidence_str

    def test_momentum_scalper_high_confidence(self):
        """Very clear scalper profile should get high confidence (> 0.7)."""
        fills = []
        base_time = 0
        # 25 winning scalps with 30 min hold — in scalper sweet spot (15m-2h)
        for _ in range(25):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=50100.0,
                    sz=0.5,
                    fee=0.5,
                    entry_time=base_time,
                    exit_time=base_time + 1_800_000,  # 30 min
                    closedPnl=50.0,
                )
            )
            base_time += 3_600_000

        # 5 small losers
        for _ in range(5):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=49950.0,
                    sz=0.5,
                    fee=0.5,
                    entry_time=base_time,
                    exit_time=base_time + 1_200_000,  # 20 min
                    closedPnl=-25.0,
                )
            )
            base_time += 3_600_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] == "momentum_scalper"
        assert result["confidence"] >= 0.7


# ---------------------------------------------------------------------------
# Test 2: Trend Follower classification
# ---------------------------------------------------------------------------


class TestTrendFollower:
    """Wallet with long holds, directional, big winners and small losers."""

    def test_trend_follower_classified(self):
        """
        Trend follower profile:
        - Long hold times (> 4 hours avg)
        - Moderate win rate (40-55%)
        - High profit factor (big winners offset small losers)
        - Directional preference
        - Positive PnL skewness (occasional big wins)
        """
        fills = []
        base_time = 0

        # 4 big winners with long holds (12 hours)
        for _ in range(4):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=53000.0,
                    sz=1.0,
                    fee=2.0,
                    entry_time=base_time,
                    exit_time=base_time + 43_200_000,  # 12 hours
                    closedPnl=3000.0,
                )
            )
            base_time += 86_400_000  # 24h gap

        # 6 small losers with long holds (8 hours)
        for _ in range(6):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=49800.0,
                    sz=1.0,
                    fee=2.0,
                    entry_time=base_time,
                    exit_time=base_time + 28_800_000,  # 8 hours
                    closedPnl=-200.0,
                )
            )
            base_time += 86_400_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] == "trend_follower"
        assert result["confidence"] > 0.4
        assert len(result["evidence"]) > 0
        evidence_str = " ".join(result["evidence"])
        assert "hold_time" in evidence_str or "profit_factor" in evidence_str


# ---------------------------------------------------------------------------
# Test 3: Mean Reversion classification
# ---------------------------------------------------------------------------


class TestMeanReversion:
    """Wallet with moderate holds, mixed direction, scale-ins."""

    def test_mean_reversion_classified(self):
        """
        Mean reversion profile:
        - Moderate hold times (1-4 hours)
        - Decent win rate (> 50%)
        - Mixed direction (buy dips, sell rips) — equal longs and shorts
        - Scale-in entries (averaging into positions)
        - >= 10 trades total
        """
        fills = []
        base_time = 0

        # 8 winning trades: strictly alternating long and short (4L + 4S), 2h hold
        for i in range(8):
            if i % 2 == 0:
                fills.extend(
                    _make_long_position(
                        coin="BTC",
                        entry_px=50000.0,
                        exit_px=50300.0,
                        sz=1.0,
                        fee=1.0,
                        entry_time=base_time,
                        exit_time=base_time + 7_200_000,  # 2 hours
                        closedPnl=300.0,
                    )
                )
            else:
                fills.extend(
                    _make_short_position(
                        coin="BTC",
                        entry_px=50300.0,
                        exit_px=50000.0,
                        sz=1.0,
                        fee=1.0,
                        entry_time=base_time,
                        exit_time=base_time + 7_200_000,  # 2 hours
                        closedPnl=300.0,
                    )
                )
            base_time += 14_400_000

        # 2 losing shorts (to keep direction balanced)
        for _ in range(2):
            fills.extend(
                _make_short_position(
                    coin="BTC",
                    entry_px=50000.0,
                    exit_px=50200.0,
                    sz=1.0,
                    fee=1.0,
                    entry_time=base_time,
                    exit_time=base_time + 7_200_000,
                    closedPnl=-200.0,
                )
            )
            base_time += 14_400_000

        # Add scale-in fills: 2 entry fills within 5 min for one position
        fills.append(
            _fill(
                px=49000.0,
                sz=0.5,
                time_ms=base_time,
                dir_str="Open Long",
                start_position=0.0,
            )
        )
        fills.append(
            _fill(
                px=48900.0,
                sz=0.5,
                time_ms=base_time + 120_000,  # 2 min later
                dir_str="Open Long",
                start_position=0.5,
            )
        )
        fills.append(
            _fill(
                px=49200.0,
                sz=1.0,
                time_ms=base_time + 7_200_000,
                dir_str="Close Long",
                start_position=1.0,
                closedPnl=200.0,
            )
        )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] == "mean_reversion"
        assert result["confidence"] > 0.3
        assert len(result["evidence"]) > 0


# ---------------------------------------------------------------------------
# Test 4: LP Consumer classification
# ---------------------------------------------------------------------------


class TestLPConsumer:
    """Wallet with very short holds, very high win rate, bot-like activity."""

    def test_lp_consumer_classified(self):
        """
        LP Consumer profile:
        - Very short hold times (< 30 min)
        - Very high win rate (> 70%)
        - High clip size consistency
        - Spread across many hours (24/7 bot-like)
        - Many trades
        """
        fills = []
        # 25 winning ultra-short trades spread across 24h with consistent sizes
        for h in range(25):
            entry_time = h * 3_456_000  # spread evenly across ~24h
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=50030.0,
                    sz=1.0,
                    fee=0.5,
                    entry_time=entry_time,
                    exit_time=entry_time + 600_000,  # 10 min
                    closedPnl=30.0,
                )
            )

        # 3 tiny losers
        for i in range(3):
            entry_time = (25 + i) * 3_456_000
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=49995.0,
                    sz=1.0,
                    fee=0.5,
                    entry_time=entry_time,
                    exit_time=entry_time + 300_000,  # 5 min
                    closedPnl=-5.0,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] == "lp_consumer"
        assert result["confidence"] > 0.5
        evidence_str = " ".join(result["evidence"])
        assert "win_rate" in evidence_str or "hold_time" in evidence_str


# ---------------------------------------------------------------------------
# Test 5: Grid classification
# ---------------------------------------------------------------------------


class TestGrid:
    """Wallet with consistent clips, multiple markets, mixed direction, 24/7."""

    def test_grid_classified(self):
        """
        Grid bot profile:
        - Consistent clip sizes
        - Multiple markets (3)
        - Mixed direction (exactly equal longs and shorts)
        - High coverage (trades across many hours)
        - Many trades
        - Moderate win rate
        """
        fills = []
        base_time = 0

        # Trades across 3 markets with consistent sizes
        # Strictly alternating direction to ensure "mixed"
        markets = ["BTC", "ETH", "SOL"]
        # 15 winning longs
        for i in range(15):
            coin = markets[i % 3]
            fills.extend(
                _make_long_position(
                    coin=coin,
                    entry_px=100.0,
                    exit_px=101.0,
                    sz=0.5,
                    fee=0.1,
                    entry_time=base_time,
                    exit_time=base_time + 1_800_000,  # 30 min
                    closedPnl=0.5,
                )
            )
            base_time += 3_600_000

        # 15 winning shorts
        for i in range(15):
            coin = markets[i % 3]
            fills.extend(
                _make_short_position(
                    coin=coin,
                    entry_px=100.0,
                    exit_px=99.0,
                    sz=0.5,
                    fee=0.1,
                    entry_time=base_time,
                    exit_time=base_time + 1_800_000,
                    closedPnl=0.5,
                )
            )
            base_time += 3_600_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] == "grid"
        assert result["confidence"] > 0.4
        evidence_str = " ".join(result["evidence"])
        assert (
            "clip_consistency" in evidence_str
            or "coverage" in evidence_str
            or "direction" in evidence_str
        )


# ---------------------------------------------------------------------------
# Test 6: Unknown / Insufficient data
# ---------------------------------------------------------------------------


class TestUnknownAndInsufficientData:
    """Wallets with few trades or no clear pattern should be 'unknown'."""

    def test_insufficient_data_classified_as_unknown(self):
        """Wallet with < 10 trades should be 'unknown' with low confidence."""
        fills = []
        # Only 3 trades
        for i in range(3):
            fills.extend(
                _make_long_position(
                    entry_time=i * 7_200_000,
                    exit_time=i * 7_200_000 + 3_600_000,
                    closedPnl=50.0,
                )
            )

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert result["strategy"] in ("unknown", "insufficient_data")
        assert result["confidence"] < 0.5

    def test_no_clear_pattern_classified_as_unknown(self):
        """
        Wallet with mixed signals — no strong pattern match — should
        be 'unknown' or low confidence.
        """
        # Only 5 trades with no clear pattern: mixed hold times,
        # inconsistent PnL, mixed direction
        fills = []
        base_time = 0
        pnls = [100.0, -200.0, 50.0, -300.0, 150.0]
        hold_times = [30_000, 28_800_000, 600_000, 43_200_000, 1_800_000]

        for pnl, hold in zip(pnls, hold_times):
            fills.extend(
                _make_long_position(
                    entry_time=base_time,
                    exit_time=base_time + hold,
                    closedPnl=pnl,
                    sz=float(len(fills) % 3 + 1) * 0.5,  # varied sizes
                )
            )
            base_time += hold + 3_600_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        # Should be unknown or a low-confidence classification
        assert result["confidence"] < 0.6

    def test_empty_data_classified_as_unknown(self):
        """Empty fills should return unknown with 0 confidence."""
        result = classify_wallet({}, [])

        assert result["strategy"] in ("unknown", "insufficient_data")
        assert result["confidence"] == 0.0

    def test_empty_metrics_no_clusters(self):
        """Empty metrics and clusters dict should return unknown."""
        result = classify_wallet({"total_trades": 0}, [])

        assert result["strategy"] in ("unknown", "insufficient_data")
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Test 7: Evidence and confidence reporting
# ---------------------------------------------------------------------------


class TestEvidenceAndConfidence:
    """Verify evidence lists and confidence scores are informative."""

    def test_evidence_contains_metric_names(self):
        """Each evidence item should reference specific metric names."""
        fills = []
        base_time = 0
        for _ in range(15):
            fills.extend(
                _make_long_position(
                    entry_px=50000.0,
                    exit_px=50200.0,
                    sz=1.0,
                    fee=1.0,
                    entry_time=base_time,
                    exit_time=base_time + 2_700_000,
                    closedPnl=200.0,
                )
            )
            base_time += 5_400_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        # Evidence should be a non-empty list of strings
        assert isinstance(result["evidence"], list)
        assert len(result["evidence"]) > 0
        for e in result["evidence"]:
            assert isinstance(e, str)
            assert len(e) > 5  # Each evidence should be descriptive

    def test_confidence_between_0_and_1(self):
        """Confidence should always be in [0, 1]."""
        fills = []
        base_time = 0
        for pnl in [100.0, -50.0, 200.0, 75.0, -25.0, 150.0, 80.0, -10.0]:
            fills.extend(
                _make_long_position(
                    entry_time=base_time,
                    exit_time=base_time + 3_600_000,
                    closedPnl=pnl,
                )
            )
            base_time += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_result_has_required_fields(self):
        """Every classification result must have strategy, confidence, evidence."""
        fills = []
        base_time = 0
        for _ in range(5):
            fills.extend(
                _make_long_position(
                    entry_time=base_time,
                    exit_time=base_time + 3_600_000,
                    closedPnl=50.0,
                )
            )
            base_time += 7_200_000

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        result = classify_wallet(metrics, clusters)

        assert "strategy" in result
        assert "confidence" in result
        assert "evidence" in result
        assert isinstance(result["strategy"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["evidence"], list)


# ---------------------------------------------------------------------------
# Test 8: classify_strategies batch processing
# ---------------------------------------------------------------------------


class TestClassifyStrategies:
    """Test batch classification of multiple wallets."""

    def test_classify_strategies_multiple_wallets(self):
        """classify_strategies should classify a list of wallets."""
        wallet1_fills = []
        base_time = 0
        # Scalper-like wallet
        for _ in range(15):
            wallet1_fills.extend(
                _make_long_position(
                    entry_time=base_time,
                    exit_time=base_time + 900_000,  # 15 min
                    closedPnl=50.0,
                )
            )
            base_time += 3_600_000

        wallet2_fills = []
        base_time = 0
        # Trend follower-like wallet
        for _ in range(4):
            wallet2_fills.extend(
                _make_long_position(
                    entry_time=base_time,
                    exit_time=base_time + 43_200_000,  # 12 hours
                    closedPnl=3000.0,
                )
            )
            base_time += 86_400_000
        for _ in range(6):
            wallet2_fills.extend(
                _make_long_position(
                    entry_time=base_time,
                    exit_time=base_time + 28_800_000,  # 8 hours
                    closedPnl=-200.0,
                )
            )
            base_time += 86_400_000

        wallets = [
            {"address": "0xscalper", "fills": wallet1_fills},
            {"address": "0xtrend", "fills": wallet2_fills},
        ]

        results = classify_strategies(wallets)

        assert len(results) == 2
        for r in results:
            assert "wallet" in r
            assert "strategy" in r
            assert "confidence" in r
            assert "evidence" in r

        # Verify wallets are classified differently
        strategies = [r["strategy"] for r in results]
        assert "momentum_scalper" in strategies
        assert "trend_follower" in strategies
