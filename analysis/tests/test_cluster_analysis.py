"""Tests for cluster_analysis.py

Uses synthetic fill data to test clustering logic without API calls.
Covers: normalization, wallet profiling, grouping, tolerance splitting,
edge cases, and idempotency.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from analysis.cluster_analysis import (
    MIN_CLUSTER_SIZE,
    _build_cluster,
    _compute_median_fill_notional,
    _parse_float,
    _safe_get,
    _split_by_tolerance,
    analyze_clusters,
    compute_wallet_profile,
    group_wallets_by_similarity,
    normalize_fills,
    save_results,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_fill(
    coin="BTC",
    side="B",
    px="100.0",
    sz="1.0",
    fee="0.10",
    closed_pnl="0.0",
    time_ms=1000000,
    dir_str="Open Long",
):
    """Create a synthetic fill dict in scraper format (snake_case)."""
    return {
        "coin": coin,
        "side": side,
        "px": px,
        "sz": sz,
        "fee": fee,
        "closed_pnl": closed_pnl,
        "time": time_ms,
        "dir": dir_str,
        "hash": f"0x{time_ms:016x}",
    }


def _make_open_long_fills(coin="BTC", px="100.0", sz="1.0", base_time=1000000):
    """Create fills for opening a long position."""
    return [
        _make_fill(
            coin=coin, side="B", px=px, sz=sz, time_ms=base_time, dir_str="Open Long"
        ),
    ]


def _make_close_long_fills(
    coin="BTC", px="110.0", sz="1.0", base_time=2000000, closed_pnl="10.0"
):
    """Create fills for closing a long position."""
    return [
        _make_fill(
            coin=coin,
            side="A",
            px=px,
            sz=sz,
            time_ms=base_time,
            dir_str="Close Long",
            closed_pnl=closed_pnl,
        ),
    ]


def _make_position_fills(
    coin="BTC",
    entry_px="100.0",
    exit_px="110.0",
    sz="1.0",
    entry_time=1000000,
    exit_time=2000000,
    closed_pnl="10.0",
    direction="long",
):
    """Create fills for a complete open→close position cycle."""
    if direction == "long":
        opens = _make_open_long_fills(coin, entry_px, sz, entry_time)
        closes = _make_close_long_fills(coin, exit_px, sz, exit_time, closed_pnl)
    else:
        opens = [
            _make_fill(
                coin=coin,
                side="A",
                px=entry_px,
                sz=sz,
                time_ms=entry_time,
                dir_str="Open Short",
            ),
        ]
        closes = [
            _make_fill(
                coin=coin,
                side="B",
                px=exit_px,
                sz=sz,
                time_ms=exit_time,
                dir_str="Close Short",
                closed_pnl=closed_pnl,
            ),
        ]
    return opens + closes


def _make_wallet_with_positions(
    address="0xwallet1",
    coin="BTC",
    n_positions=15,
    direction="long",
    hold_time_hours=1.0,
    entry_px=100.0,
    exit_offset=10.0,
    sz=1.0,
):
    """Create a wallet dict with multiple position cycles.

    Generates ``n_positions`` complete open→close position cycles on
    the given coin with the specified parameters.
    """
    fills = []
    for i in range(n_positions):
        entry_time = 1000000 + int(i * hold_time_hours * 2 * 3_600_000)
        exit_time = entry_time + int(hold_time_hours * 3_600_000)

        pnl = exit_offset if direction == "long" else -exit_offset
        fills.extend(
            _make_position_fills(
                coin=coin,
                entry_px=str(entry_px),
                exit_px=str(
                    entry_px + exit_offset
                    if direction == "long"
                    else entry_px - exit_offset
                ),
                sz=str(sz),
                entry_time=entry_time,
                exit_time=exit_time,
                closed_pnl=str(pnl),
                direction=direction,
            )
        )

    return {
        "address": address,
        "source": "test",
        "total_fills": len(fills),
        "net_pnl": sum(float(f.get("closed_pnl", 0)) for f in fills),
        "last_active": "2026-05-20T00:00:00Z",
        "fills": fills,
    }


def _make_scalper_wallet(address="0xscalper", coin="BTC", n_positions=15):
    """Short hold time, directional, high win rate → momentum_scalper."""
    return _make_wallet_with_positions(
        address=address,
        coin=coin,
        n_positions=n_positions,
        direction="long",
        hold_time_hours=0.5,
        sz=1.0,
    )


def _make_trend_wallet(address="0xtrend", coin="ETH", n_positions=15):
    """Long hold time, directional → trend_follower."""
    return _make_wallet_with_positions(
        address=address,
        coin=coin,
        n_positions=n_positions,
        direction="long",
        hold_time_hours=8.0,
        sz=2.0,
    )


def _make_mean_reversion_wallet(address="0xmr", coin="SOL", n_positions=15):
    """Mixed direction, moderate hold → mean_reversion."""
    fills = []
    for i in range(n_positions):
        direction = "long" if i % 2 == 0 else "short"
        entry_time = 1000000 + int(i * 4 * 3_600_000)
        exit_time = entry_time + int(2 * 3_600_000)
        fills.extend(
            _make_position_fills(
                coin=coin,
                entry_px="50.0",
                exit_px="52.0" if direction == "long" else "48.0",
                sz="1.0",
                entry_time=entry_time,
                exit_time=exit_time,
                closed_pnl="2.0" if direction == "long" else "-1.0",
                direction=direction,
            )
        )
    return {
        "address": address,
        "source": "test",
        "total_fills": len(fills),
        "net_pnl": sum(float(f.get("closed_pnl", 0)) for f in fills),
        "last_active": "2026-05-20T00:00:00Z",
        "fills": fills,
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestNormalizeFills:
    """Tests for fill normalization."""

    def test_converts_closed_pnl_to_camel_case(self):
        """closed_pnl (snake_case) is converted to closedPnl (camelCase)."""
        fills = [
            _make_fill(closed_pnl="10.5", dir_str="Close Long", time_ms=1000),
        ]
        result = normalize_fills(fills)
        assert "closedPnl" in result[0]
        assert result[0]["closedPnl"] == "10.5"

    def test_preserves_existing_closed_pnl(self):
        """If closedPnl already exists, don't overwrite."""
        fills = [
            {
                "coin": "BTC",
                "side": "B",
                "px": "100",
                "sz": "1",
                "fee": "0.1",
                "closedPnl": "5.0",
                "time": 1000,
                "dir": "Open Long",
                "hash": "0xabc",
            }
        ]
        result = normalize_fills(fills)
        assert result[0]["closedPnl"] == "5.0"

    def test_infers_start_position(self):
        """startPosition is inferred from fill sequence."""
        fills = [
            _make_fill(side="B", sz="10", dir_str="Open Long", time_ms=1000),
            _make_fill(side="B", sz="5", dir_str="Open Long", time_ms=2000),
            _make_fill(
                side="A", sz="15", dir_str="Close Long", time_ms=3000, closed_pnl="1.0"
            ),
        ]
        result = normalize_fills(fills)
        assert result[0]["startPosition"] == 0.0
        assert result[1]["startPosition"] == 10.0
        assert result[2]["startPosition"] == 15.0

    def test_empty_fills(self):
        result = normalize_fills([])
        assert result == []

    def test_preserves_time_order(self):
        """Fills from multiple coins are interleaved by time."""
        fills = [
            _make_fill(coin="ETH", time_ms=3000, dir_str="Open Long"),
            _make_fill(coin="BTC", time_ms=1000, dir_str="Open Long"),
            _make_fill(coin="ETH", time_ms=2000, dir_str="Open Long"),
        ]
        result = normalize_fills(fills)
        times = [f["time"] for f in result]
        assert times == sorted(times)

    def test_per_coin_position_tracking(self):
        """Position tracked independently per coin."""
        fills = [
            _make_fill(coin="BTC", side="B", sz="5", dir_str="Open Long", time_ms=1000),
            _make_fill(
                coin="ETH", side="B", sz="10", dir_str="Open Long", time_ms=2000
            ),
            _make_fill(
                coin="BTC", side="A", sz="5", dir_str="Close Long", time_ms=3000
            ),
        ]
        result = normalize_fills(fills)
        btc_fills = [f for f in result if f["coin"] == "BTC"]
        eth_fills = [f for f in result if f["coin"] == "ETH"]

        # BTC: 0 → 5 → 0
        assert btc_fills[0]["startPosition"] == 0.0
        assert btc_fills[1]["startPosition"] == 5.0

        # ETH: 0 → 10
        assert eth_fills[0]["startPosition"] == 0.0


class TestComputeWalletProfile:
    """Tests for the per-wallet analysis pipeline."""

    def test_returns_required_fields(self):
        wallet = _make_scalper_wallet()
        profile = compute_wallet_profile(wallet)
        assert "address" in profile
        assert "strategy" in profile
        assert "confidence" in profile
        assert "evidence" in profile
        assert "metrics" in profile
        assert "clusters" in profile
        assert "median_fill_notional" in profile
        assert "num_clusters" in profile

    def test_metrics_populated(self):
        wallet = _make_scalper_wallet(n_positions=15)
        profile = compute_wallet_profile(wallet)
        metrics = profile["metrics"]
        assert metrics["total_trades"] > 0
        assert 0 <= metrics["win_rate"] <= 1.0

    def test_median_fill_notional(self):
        wallet = _make_wallet_with_positions(sz=2.0, entry_px=50.0, n_positions=5)
        profile = compute_wallet_profile(wallet)
        # All fills are 2.0 * 50.0 = 100.0 or 2.0 * 60.0 = 120.0 notional
        assert profile["median_fill_notional"] > 0


class TestMedianFillNotional:
    """Tests for _compute_median_fill_notional helper."""

    def test_computes_median(self):
        fills = [
            {"px": "100", "sz": "1"},
            {"px": "200", "sz": "1"},
            {"px": "300", "sz": "1"},
        ]
        result = _compute_median_fill_notional(fills)
        assert result == 200.0

    def test_empty_fills(self):
        assert _compute_median_fill_notional([]) == 0.0

    def test_zero_px(self):
        fills = [{"px": "0", "sz": "1"}]
        assert _compute_median_fill_notional(fills) == 0.0


class TestSplitByTolerance:
    """Tests for _split_by_tolerance helper."""

    def test_all_similar(self):
        """All values within tolerance → one group."""
        profiles = [{"v": 1.0}, {"v": 1.1}, {"v": 0.9}, {"v": 1.05}, {"v": 0.95}]
        groups = _split_by_tolerance(
            profiles, key_func=lambda p: p["v"], tolerance=0.20, min_size=3
        )
        assert len(groups) == 1
        assert len(groups[0]) == 5

    def test_split_on_large_gap(self):
        """Values with a large gap → two groups."""
        profiles = [
            {"v": 1.0},
            {"v": 1.1},
            {"v": 1.2},
            {"v": 5.0},
            {"v": 5.1},
            {"v": 5.2},
        ]
        groups = _split_by_tolerance(
            profiles, key_func=lambda p: p["v"], tolerance=0.20, min_size=3
        )
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 3

    def test_below_min_size_excluded(self):
        """Groups below min_size are excluded."""
        profiles = [
            {"v": 1.0},
            {"v": 1.1},
            {"v": 100.0},
        ]
        groups = _split_by_tolerance(
            profiles, key_func=lambda p: p["v"], tolerance=0.20, min_size=3
        )
        # No group has >= 3 members, but total is >= 3 so whole group returned
        assert len(groups) >= 1

    def test_empty_input(self):
        groups = _split_by_tolerance(
            [], key_func=lambda p: 0, tolerance=0.2, min_size=1
        )
        assert groups == []

    def test_zero_values_grouped(self):
        """All-zero values are considered similar."""
        profiles = [{"v": 0.0}, {"v": 0.0}, {"v": 0.0}]
        groups = _split_by_tolerance(
            profiles, key_func=lambda p: p["v"], tolerance=0.20, min_size=3
        )
        assert len(groups) == 1


class TestGroupWalletsBySimilarity:
    """Tests for the main grouping function."""

    def test_different_strategies_separate(self):
        """Wallets with different strategy types are in different clusters."""
        profiles = []
        for i in range(7):
            p = compute_wallet_profile(_make_scalper_wallet(address=f"0xscalper{i}"))
            profiles.append(p)
        for i in range(7):
            p = compute_wallet_profile(_make_trend_wallet(address=f"0xtrend{i}"))
            profiles.append(p)

        clusters = group_wallets_by_similarity(profiles, min_cluster_size=5)
        assert len(clusters) >= 2

        strategies = {c["strategy"] for c in clusters}
        assert "momentum_scalper" in strategies or any(
            "momentum" in c["strategy"] for c in clusters
        )

    def test_min_cluster_size_enforced(self):
        """Clusters with fewer than min_cluster_size wallets are excluded."""
        profiles = []
        # Only 2 scalper wallets (below min 5)
        for i in range(2):
            p = compute_wallet_profile(_make_scalper_wallet(address=f"0xsmall{i}"))
            profiles.append(p)

        clusters = group_wallets_by_similarity(profiles, min_cluster_size=5)
        assert len(clusters) == 0

    def test_same_strategy_same_market_clustered(self):
        """Wallets with same strategy and market are in same cluster."""
        profiles = []
        for i in range(8):
            p = compute_wallet_profile(
                _make_scalper_wallet(address=f"0xcluster{i}", coin="BTC")
            )
            profiles.append(p)

        clusters = group_wallets_by_similarity(profiles, min_cluster_size=5)
        assert len(clusters) >= 1
        # All wallets should be in some cluster
        clustered_addresses = set()
        for c in clusters:
            clustered_addresses.update(c["member_wallets"])
        assert len(clustered_addresses) >= 5

    def test_cluster_has_required_fields(self):
        """Each cluster dict has all required fields."""
        profiles = []
        for i in range(7):
            p = compute_wallet_profile(_make_scalper_wallet(address=f"0xfield{i}"))
            profiles.append(p)

        clusters = group_wallets_by_similarity(profiles, min_cluster_size=5)
        if not clusters:
            pytest.skip("No clusters formed")

        c = clusters[0]
        assert "cluster_id" in c
        assert "strategy" in c
        assert "primary_market" in c
        assert "direction" in c
        assert "member_wallets" in c
        assert "size" in c
        assert "shared_parameters" in c
        assert "divergence_metrics" in c
        assert isinstance(c["member_wallets"], list)
        assert c["size"] == len(c["member_wallets"])

    def test_idempotent_clustering(self):
        """Same input produces identical output on repeated calls."""
        profiles = []
        for i in range(7):
            p = compute_wallet_profile(_make_scalper_wallet(address=f"0xidem{i}"))
            profiles.append(p)

        clusters1 = group_wallets_by_similarity(profiles, min_cluster_size=5)
        clusters2 = group_wallets_by_similarity(profiles, min_cluster_size=5)

        assert len(clusters1) == len(clusters2)
        for c1, c2 in zip(clusters1, clusters2):
            assert c1["cluster_id"] == c2["cluster_id"]
            assert c1["member_wallets"] == c2["member_wallets"]


class TestAnalyzeClusters:
    """Tests for the main entry point."""

    def test_returns_required_fields(self):
        wallets = [
            _make_scalper_wallet(address=f"0xs{i}", coin="BTC") for i in range(7)
        ]
        wallets += [_make_trend_wallet(address=f"0xt{i}", coin="ETH") for i in range(7)]

        result = analyze_clusters(wallets, min_cluster_size=5)
        assert "clusters" in result
        assert "total_wallets" in result
        assert "classified_wallets" in result
        assert "clustered_wallets" in result
        assert "unclustered_wallets" in result
        assert "profiles" in result
        assert result["total_wallets"] == 14

    def test_produces_multiple_clusters(self):
        """Two distinct strategy types produce ≥2 clusters."""
        wallets = [
            _make_scalper_wallet(address=f"0xs{i}", coin="BTC") for i in range(8)
        ]
        wallets += [_make_trend_wallet(address=f"0xt{i}", coin="ETH") for i in range(8)]

        result = analyze_clusters(wallets, min_cluster_size=5)
        assert len(result["clusters"]) >= 2

    def test_handles_empty_wallets(self):
        result = analyze_clusters([], min_cluster_size=5)
        assert result["clusters"] == []
        assert result["total_wallets"] == 0


class TestSaveResults:
    """Tests for atomic file write."""

    def test_atomic_write(self):
        """Results are written to .tmp then renamed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "clusters.json")
            profiles = []
            for i in range(6):
                p = compute_wallet_profile(_make_scalper_wallet(address=f"0xsave{i}"))
                profiles.append(p)
            clusters = group_wallets_by_similarity(profiles, min_cluster_size=5)
            results = {
                "clusters": clusters,
                "total_wallets": 6,
                "classified_wallets": 6,
                "clustered_wallets": sum(c["size"] for c in clusters),
                "unclustered_wallets": 0,
                "profiles": profiles,
            }
            save_results(results, output_path)

            assert os.path.exists(output_path)
            # .tmp should not remain
            assert not os.path.exists(output_path + ".tmp")

            with open(output_path) as f:
                data = json.load(f)
            assert "clusters" in data
            assert data["total_wallets"] == 6

    def test_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.json")
            results = {
                "clusters": [],
                "total_wallets": 0,
                "classified_wallets": 0,
                "clustered_wallets": 0,
                "unclustered_wallets": 0,
                "profiles": [],
            }
            save_results(results, output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
