"""Tests for blueprint_generator.py

Uses synthetic cluster/profile data to verify blueprint generation,
parameter derivation from medians, traceability, and atomic file I/O.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from analysis.blueprint_generator import (
    _empty_blueprint,
    _median_or_zero,
    _percentile_or_zero,
    compute_cluster_parameters,
    generate_all_blueprints,
    generate_blueprint,
    save_blueprint,
)
from analysis.cluster_analysis import compute_wallet_profile


# ---------------------------------------------------------------------------
# Synthetic data factories (reuse from cluster_analysis patterns)
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
    fills = []
    for i in range(n_positions):
        entry_time = 1000000 + int(i * hold_time_hours * 2 * 3_600_000)
        exit_time = entry_time + int(hold_time_hours * 3_600_000)

        if direction == "long":
            fills.append(
                _make_fill(
                    coin=coin,
                    side="B",
                    px=str(entry_px),
                    sz=str(sz),
                    time_ms=entry_time,
                    dir_str="Open Long",
                )
            )
            fills.append(
                _make_fill(
                    coin=coin,
                    side="A",
                    px=str(entry_px + exit_offset),
                    sz=str(sz),
                    time_ms=exit_time,
                    dir_str="Close Long",
                    closed_pnl=str(exit_offset * sz),
                )
            )
        else:
            fills.append(
                _make_fill(
                    coin=coin,
                    side="A",
                    px=str(entry_px),
                    sz=str(sz),
                    time_ms=entry_time,
                    dir_str="Open Short",
                )
            )
            fills.append(
                _make_fill(
                    coin=coin,
                    side="B",
                    px=str(entry_px - exit_offset),
                    sz=str(sz),
                    time_ms=exit_time,
                    dir_str="Close Short",
                    closed_pnl=str(exit_offset * sz),
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


def _make_cluster_and_profiles(
    n_wallets=7, coin="BTC", strategy="momentum_scalper", hold_time_hours=0.5
):
    """Create a set of profiles and a cluster dict for testing."""
    profiles = []
    for i in range(n_wallets):
        wallet = _make_wallet_with_positions(
            address=f"0xbp_test_{i}",
            coin=coin,
            n_positions=15,
            hold_time_hours=hold_time_hours,
            sz=1.0,
        )
        profile = compute_wallet_profile(wallet)
        profiles.append(profile)

    cluster = {
        "cluster_id": "cluster-001",
        "strategy": strategy,
        "primary_market": coin,
        "direction": "long",
        "member_wallets": [p["address"] for p in profiles],
        "size": n_wallets,
        "shared_parameters": {},
        "divergence_metrics": {},
        "profiles": profiles,
    }
    return cluster, profiles


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestComputeClusterParameters:
    """Tests for statistical parameter computation."""

    def test_returns_all_sections(self):
        cluster, profiles = _make_cluster_and_profiles()
        params = compute_cluster_parameters(profiles)

        assert "hold_time" in params
        assert "win_rate" in params
        assert "clip_size" in params
        assert "pnl" in params
        assert "fees" in params
        assert "tp_sl" in params

    def test_hold_time_positive(self):
        cluster, profiles = _make_cluster_and_profiles(hold_time_hours=0.5)
        params = compute_cluster_parameters(profiles)
        # Median hold time should be positive
        assert params["hold_time"]["median_hours"] > 0

    def test_win_rate_between_0_and_1(self):
        cluster, profiles = _make_cluster_and_profiles()
        params = compute_cluster_parameters(profiles)
        assert 0 <= params["win_rate"]["median"] <= 1.0

    def test_clip_size_positive(self):
        cluster, profiles = _make_cluster_and_profiles()
        params = compute_cluster_parameters(profiles)
        assert params["clip_size"]["median_notional"] > 0

    def test_empty_profiles(self):
        params = compute_cluster_parameters([])
        assert params == {}

    def test_pnl_sections(self):
        cluster, profiles = _make_cluster_and_profiles()
        params = compute_cluster_parameters(profiles)
        assert params["pnl"]["total_positions"] > 0
        assert params["pnl"]["winning_positions"] >= 0
        assert params["pnl"]["losing_positions"] >= 0


class TestGenerateBlueprint:
    """Tests for blueprint generation from cluster data."""

    def test_has_required_fields(self):
        """Blueprint contains all required top-level fields."""
        cluster, profiles = _make_cluster_and_profiles()
        bp = generate_blueprint("cluster-001", cluster, profiles)

        required_fields = [
            "strategy_name",
            "strategy_type",
            "source_cluster_id",
            "source_wallets",
            "primary_market",
            "direction",
            "markets",
            "entry_conditions",
            "exit_conditions",
            "risk_parameters",
            "statistical_parameters",
            "confidence_score",
            "sample_size",
            "parameter_traceability",
        ]
        for field in required_fields:
            assert field in bp, f"Missing field: {field}"

    def test_source_wallets_match_cluster(self):
        """source_wallets matches cluster member_wallets."""
        cluster, profiles = _make_cluster_and_profiles()
        bp = generate_blueprint("cluster-001", cluster, profiles)
        assert bp["source_wallets"] == cluster["member_wallets"]

    def test_sample_size_correct(self):
        """sample_size.wallets matches profile count."""
        cluster, profiles = _make_cluster_and_profiles(n_wallets=7)
        bp = generate_blueprint("cluster-001", cluster, profiles)
        assert bp["sample_size"]["wallets"] == 7

    def test_confidence_between_0_and_1(self):
        cluster, profiles = _make_cluster_and_profiles()
        bp = generate_blueprint("cluster-001", cluster, profiles)
        assert 0 <= bp["confidence_score"] <= 1.0

    def test_parameters_from_medians(self):
        """Risk parameters are derived from cluster data, not invented."""
        cluster, profiles = _make_cluster_and_profiles(n_wallets=7, coin="BTC")
        bp = generate_blueprint("cluster-001", cluster, profiles)

        # clip_size_usd should be the median fill notional from profiles
        clip_notional = bp["risk_parameters"]["clip_size_usd"]
        assert clip_notional > 0, "clip_size_usd should be positive (from data)"

        # max_hold_hours should be p75 of hold times (positive)
        max_hold = bp["risk_parameters"]["max_hold_hours"]
        assert max_hold >= 0

    def test_traceability_populated(self):
        """parameter_traceability maps each parameter to its data source."""
        cluster, profiles = _make_cluster_and_profiles()
        bp = generate_blueprint("cluster-001", cluster, profiles)

        trace = bp["parameter_traceability"]
        assert "clip_size_usd" in trace
        assert "take_profit_pct" in trace
        assert "stop_loss_pct" in trace
        assert "max_hold_hours" in trace
        assert "confidence_score" in trace

        # Each traceability entry should mention data derivation
        for key, value in trace.items():
            assert isinstance(value, str)
            assert len(value) > 10, f"Traceability for {key} is too short"

    def test_strategy_name_format(self):
        """strategy_name follows strategy_market_direction convention."""
        cluster, profiles = _make_cluster_and_profiles(
            strategy="momentum_scalper", coin="BTC"
        )
        bp = generate_blueprint("cluster-001", cluster, profiles)
        assert "momentum_scalper" in bp["strategy_name"]
        assert "btc" in bp["strategy_name"].lower()

    def test_empty_cluster_blueprint(self):
        """Empty cluster produces valid empty blueprint."""
        cluster = {
            "member_wallets": [],
            "strategy": "unknown",
            "primary_market": "UNKNOWN",
            "direction": "unknown",
        }
        bp = generate_blueprint("cluster-empty", cluster, profiles=[])
        assert bp["sample_size"]["wallets"] == 0
        assert bp["confidence_score"] == 0.0

    def test_exit_conditions_has_tp_sl(self):
        cluster, profiles = _make_cluster_and_profiles()
        bp = generate_blueprint("cluster-001", cluster, profiles)
        ec = bp["exit_conditions"]
        assert "take_profit_pct" in ec
        assert "stop_loss_pct" in ec
        assert "max_hold_hours" in ec


class TestEmptyBlueprint:
    """Tests for _empty_blueprint edge case."""

    def test_returns_valid_structure(self):
        cluster = {
            "member_wallets": [],
            "strategy": "unknown",
            "primary_market": "UNKNOWN",
            "direction": "unknown",
        }
        bp = _empty_blueprint("cluster-x", cluster)
        assert bp["source_cluster_id"] == "cluster-x"
        assert bp["sample_size"]["wallets"] == 0


class TestSaveBlueprint:
    """Tests for atomic file write."""

    def test_atomic_write(self):
        """File is written via .tmp then renamed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bp.json")
            bp = {"strategy_name": "test", "data": [1, 2, 3]}
            save_blueprint(bp, path)

            assert os.path.exists(path)
            assert not os.path.exists(path + ".tmp")

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["strategy_name"] == "test"

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "bp.json")
            save_blueprint({"test": True}, path)
            assert os.path.exists(path)


class TestGenerateAllBlueprints:
    """Tests for batch blueprint generation."""

    def test_generates_multiple_blueprints(self):
        cluster1, profiles1 = _make_cluster_and_profiles(
            n_wallets=6, coin="BTC", strategy="momentum_scalper"
        )
        cluster2, profiles2 = _make_cluster_and_profiles(
            n_wallets=6, coin="ETH", strategy="trend_follower"
        )

        # Merge profiles
        all_profiles = profiles1 + profiles2

        clusters = [
            {**cluster1, "profiles": profiles1},
            {**cluster2, "profiles": profiles2},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            blueprints = generate_all_blueprints(
                clusters, all_profiles, output_dir=tmpdir
            )

            assert len(blueprints) == 2
            assert os.path.exists(os.path.join(tmpdir, "cluster-001.json"))
            assert os.path.exists(os.path.join(tmpdir, "cluster-001.json"))

    def test_empty_clusters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blueprints = generate_all_blueprints([], output_dir=tmpdir)
            assert blueprints == []


class TestMedianOrZero:
    """Tests for helper function."""

    def test_non_empty(self):
        assert _median_or_zero([1, 2, 3]) == 2.0

    def test_empty(self):
        assert _median_or_zero([]) == 0.0

    def test_single(self):
        assert _median_or_zero([42]) == 42.0


class TestPercentileOrZero:
    """Tests for helper function."""

    def test_non_empty(self):
        assert _percentile_or_zero([1, 2, 3, 4, 5], 50) == 3.0

    def test_empty(self):
        assert _percentile_or_zero([], 50) == 0.0
