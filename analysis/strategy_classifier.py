"""Strategy Classifier Module

Classifies wallets into strategy types based on wallet metrics and position
cluster data. Uses a rule-based scoring system inspired by the Bulk.Trade
methodology for strategy identification.

Strategy taxonomy:
  momentum_scalper  — Short holds (< 2h), high win rate, directional
  mean_reversion    — Moderate holds, mixed direction, scale-in entries
  trend_follower    — Long holds (> 4h), high profit factor, skewed PnL
  lp_consumer       — Very short holds (< 30m), very high win rate, bot-like
  grid              — Consistent clips, 24/7, multi-market, mixed direction
  unknown           — Insufficient data or no clear pattern

Each classification includes:
  - strategy: string label from the taxonomy above
  - confidence: float in [0, 1] indicating classification certainty
  - evidence: list of strings describing which metrics supported the decision

Input:
  metrics  — dict from wallet_metrics.compute_wallet_metrics()
  clusters — list of position cluster dicts from position_clustering.cluster_fills()
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimum number of closed positions for reliable classification
MIN_TRADES_FOR_CLASSIFICATION = 10

# Strategy type constants
STRATEGY_MOMENTUM_SCALPER = "momentum_scalper"
STRATEGY_MEAN_REVERSION = "mean_reversion"
STRATEGY_TREND_FOLLOWER = "trend_follower"
STRATEGY_LP_CONSUMER = "lp_consumer"
STRATEGY_GRID = "grid"
STRATEGY_UNKNOWN = "unknown"
STRATEGY_INSUFFICIENT_DATA = "insufficient_data"

ALL_STRATEGIES = [
    STRATEGY_MOMENTUM_SCALPER,
    STRATEGY_MEAN_REVERSION,
    STRATEGY_TREND_FOLLOWER,
    STRATEGY_LP_CONSUMER,
    STRATEGY_GRID,
]


def _safe_get(metrics: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from the metrics dict."""
    return metrics.get(key, default)


def _safe_float(metrics: dict, key: str, default: float = 0.0) -> float:
    """Safely get a float value from the metrics dict."""
    val = _safe_get(metrics, key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_nested_float(metrics: dict, *keys, default: float = 0.0) -> float:
    """Safely get a nested float value from the metrics dict."""
    current = metrics
    for key in keys[:-1]:
        if not isinstance(current, dict):
            return default
        current = current.get(key, {})
    if not isinstance(current, dict):
        return default
    val = current.get(keys[-1], default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Strategy scoring functions
# ---------------------------------------------------------------------------


def _score_momentum_scalper(metrics: dict) -> tuple[float, list[str]]:
    """Score wallet for momentum scalper strategy.

    Indicators:
      - avg_hold_time 15min-2h (short but not ultra-short like LP consumer)
      - win_rate > 0.55
      - total_trades >= 15
      - preferred_direction is 'long' or 'short' (not mixed)
      - NOT ultra-short hold (LP territory) and NOT mixed direction (grid territory)
    """
    evidence = []
    score = 0.0
    max_score = 0.0

    # Hold time: short holds are key indicator (15min to 2h sweet spot)
    max_score += 1.0
    hold_time = _safe_float(metrics, "avg_hold_time_hours", 0.0)
    if 0.25 <= hold_time <= 2.0:  # 15 min to 2 hours — scalper sweet spot
        score += 1.0
        evidence.append(f"hold_time: short ({hold_time:.2f}h)")
    elif 2.0 < hold_time <= 4.0:
        score += 0.5
        evidence.append(f"hold_time: moderate ({hold_time:.2f}h)")
    elif 0 < hold_time < 0.25:  # < 15 min — LP consumer territory, not scalper
        score += 0.3  # Reduced: too short for typical scalper
        evidence.append(f"hold_time: very short ({hold_time:.2f}h)")

    # Win rate: high win rate
    max_score += 1.0
    win_rate = _safe_float(metrics, "win_rate", 0.0)
    if win_rate > 0.7:
        score += 1.0
        evidence.append(f"win_rate: very high ({win_rate:.1%})")
    elif win_rate > 0.55:
        score += 0.7
        evidence.append(f"win_rate: high ({win_rate:.1%})")
    elif win_rate > 0.45:
        score += 0.2

    # Total trades: many trades needed for statistical reliability
    max_score += 0.5
    total_trades = int(_safe_float(metrics, "total_trades", 0))
    if total_trades >= 20:
        score += 0.5
        evidence.append(f"trade_count: high ({total_trades})")
    elif total_trades >= 10:
        score += 0.3

    # Direction: directional (not mixed) — critical differentiator from grid
    max_score += 0.7
    direction = _safe_get(metrics, "preferred_direction", "unknown")
    if direction in ("long", "short"):
        score += 0.7
        evidence.append(f"direction: directional ({direction})")
    # Mixed direction strongly suggests NOT a scalper

    # Clip consistency: moderate (not necessarily fixed like grid)
    max_score += 0.3
    clip_cons = _safe_float(metrics, "clip_size_consistency", 0.0)
    if clip_cons > 0.7:
        score += 0.3
        evidence.append(f"clip_consistency: moderate ({clip_cons:.0%})")
    elif clip_cons > 0.5:
        score += 0.15

    confidence = score / max_score if max_score > 0 else 0.0
    return confidence, evidence


def _score_trend_follower(metrics: dict) -> tuple[float, list[str]]:
    """Score wallet for trend follower strategy.

    Indicators:
      - avg_hold_time > 4 hours
      - profit_factor > 1.5 (big winners offset small losers)
      - positive PnL skewness (right-tailed distribution)
      - preferred_direction is 'long' or 'short'
      - moderate win rate (30-55%)
    """
    evidence = []
    score = 0.0
    max_score = 0.0

    # Hold time: long holds are key indicator
    max_score += 1.0
    hold_time = _safe_float(metrics, "avg_hold_time_hours", 0.0)
    if hold_time > 12.0:
        score += 1.0
        evidence.append(f"hold_time: very long ({hold_time:.2f}h)")
    elif hold_time > 4.0:
        score += 0.8
        evidence.append(f"hold_time: long ({hold_time:.2f}h)")
    elif hold_time > 2.0:
        score += 0.3

    # Profit factor: high (big winners vs small losers)
    max_score += 1.0
    profit_factor = _safe_float(metrics, "profit_factor", 0.0)
    if profit_factor == 0.0:
        # profit_factor is None when no losses → infinite, which is trend-like
        raw_pf = _safe_get(metrics, "profit_factor")
        if raw_pf is None:
            score += 0.5  # All wins — could be trend follower or just lucky
            evidence.append("profit_factor: infinite (no losses)")
        else:
            max_score -= 1.0  # No meaningful data
    elif profit_factor > 3.0:
        score += 1.0
        evidence.append(f"profit_factor: very high ({profit_factor:.2f})")
    elif profit_factor > 1.5:
        score += 0.7
        evidence.append(f"profit_factor: high ({profit_factor:.2f})")
    elif profit_factor > 1.0:
        score += 0.3
        evidence.append(f"profit_factor: moderate ({profit_factor:.2f})")

    # PnL skewness: positive skew (big winners, small losers)
    max_score += 0.8
    skewness = _safe_nested_float(metrics, "pnl_distribution", "skewness", default=0.0)
    if skewness > 1.0:
        score += 0.8
        evidence.append(f"pnl_skewness: strongly positive ({skewness:.2f})")
    elif skewness > 0.3:
        score += 0.5
        evidence.append(f"pnl_skewness: positive ({skewness:.2f})")
    elif skewness > 0:
        score += 0.2

    # Direction: directional (not mixed)
    max_score += 0.5
    direction = _safe_get(metrics, "preferred_direction", "unknown")
    if direction in ("long", "short"):
        score += 0.5
        evidence.append(f"direction: directional ({direction})")
    elif direction == "mixed":
        score += 0.1

    # Win rate: moderate (trend followers accept frequent small losses)
    max_score += 0.4
    win_rate = _safe_float(metrics, "win_rate", 0.0)
    if 0.3 <= win_rate <= 0.55:
        score += 0.4
        evidence.append(f"win_rate: moderate ({win_rate:.1%})")
    elif 0.55 < win_rate <= 0.65:
        score += 0.2

    confidence = score / max_score if max_score > 0 else 0.0
    return confidence, evidence


def _score_mean_reversion(metrics: dict, clusters: list) -> tuple[float, list[str]]:
    """Score wallet for mean reversion strategy.

    Mean reversion traders buy dips and sell rips, resulting in mixed
    direction and often scaling into positions.

    Indicators:
      - preferred_direction is 'mixed' — PRIMARY differentiator
      - avg_hold_time 1-4 hours (moderate)
      - scale_in_count > 0 (averaging into positions)
      - win_rate > 0.5 (decent)
      - single or few markets (focused)
    """
    evidence = []
    score = 0.0
    max_score = 0.0

    # Direction: mixed is PRIMARY indicator — differentiates from scalper/trend
    max_score += 1.5
    direction = _safe_get(metrics, "preferred_direction", "unknown")
    if direction == "mixed":
        score += 1.5
        evidence.append("direction: mixed (buy/sell both sides)")
    else:
        score += 0.1

    # Hold time: moderate (1-4 hours) — differentiates from scalper (short) and trend (long)
    max_score += 1.0
    hold_time = _safe_float(metrics, "avg_hold_time_hours", 0.0)
    if 1.0 <= hold_time <= 4.0:
        score += 1.0
        evidence.append(f"hold_time: moderate ({hold_time:.2f}h)")
    elif 0.5 <= hold_time < 1.0 or 4.0 < hold_time <= 8.0:
        score += 0.5
        evidence.append(f"hold_time: near-moderate ({hold_time:.2f}h)")

    # Scale-in count: averaging into positions (characteristic MR behavior)
    max_score += 0.8
    scale_in_count = int(_safe_float(metrics, "scale_in_count", 0))
    total_trades = int(_safe_float(metrics, "total_trades", 0))
    if total_trades > 0:
        scale_in_ratio = scale_in_count / total_trades
        if scale_in_ratio > 0.3:
            score += 0.8
            evidence.append(
                f"scale_in_ratio: high ({scale_in_count}/{total_trades} = {scale_in_ratio:.0%})"
            )
        elif scale_in_ratio > 0.1:
            score += 0.4
            evidence.append(
                f"scale_in_ratio: moderate ({scale_in_count}/{total_trades})"
            )

    # Win rate: decent (> 50%)
    max_score += 0.7
    win_rate = _safe_float(metrics, "win_rate", 0.0)
    if win_rate > 0.6:
        score += 0.7
        evidence.append(f"win_rate: good ({win_rate:.1%})")
    elif win_rate > 0.5:
        score += 0.5
        evidence.append(f"win_rate: decent ({win_rate:.1%})")
    elif win_rate > 0.4:
        score += 0.2

    # Single/few market focus (differentiator from grid which is multi-market)
    max_score += 0.3
    markets = _safe_get(metrics, "markets_traded", [])
    if isinstance(markets, list) and len(markets) <= 3:
        score += 0.3
        evidence.append(f"markets: focused ({len(markets)} markets)")

    confidence = score / max_score if max_score > 0 else 0.0
    return confidence, evidence


def _score_lp_consumer(metrics: dict) -> tuple[float, list[str]]:
    """Score wallet for LP consumption strategy.

    LP consumers exploit LP pools with ultra-short holds and very high win
    rates. They are bot-like: 24/7 activity, consistent clip sizes, high
    frequency fills.

    Indicators:
      - avg_hold_time < 30 minutes (very short) — PRIMARY differentiator
      - win_rate > 0.7 (very high)
      - clip_size_consistency > 0.8 (consistent)
      - coverage_pct > 0.7 (near-24/7 activity)
      - fill_interval_stats.pct_sub_30s (high frequency)
      - total_trades >= 15 (many trades)
    """
    evidence = []
    score = 0.0
    max_score = 0.0

    # Hold time: ultra-short is the PRIMARY differentiator from scalper
    max_score += 1.5
    hold_time = _safe_float(metrics, "avg_hold_time_hours", 0.0)
    if hold_time > 0 and hold_time <= 0.25:  # <= 15 min
        score += 1.5
        evidence.append(f"hold_time: ultra-short ({hold_time * 60:.1f}min)")
    elif 0.25 < hold_time <= 0.5:  # <= 30 min
        score += 1.2
        evidence.append(f"hold_time: very short ({hold_time * 60:.1f}min)")
    elif 0.5 < hold_time <= 1.0:
        score += 0.4

    # Win rate: very high
    max_score += 1.0
    win_rate = _safe_float(metrics, "win_rate", 0.0)
    if win_rate > 0.85:
        score += 1.0
        evidence.append(f"win_rate: very high ({win_rate:.1%})")
    elif win_rate > 0.7:
        score += 0.8
        evidence.append(f"win_rate: high ({win_rate:.1%})")
    elif win_rate > 0.6:
        score += 0.4

    # Clip size consistency: consistent (bot-like)
    max_score += 0.8
    clip_cons = _safe_float(metrics, "clip_size_consistency", 0.0)
    if clip_cons > 0.9:
        score += 0.8
        evidence.append(f"clip_consistency: very high ({clip_cons:.0%})")
    elif clip_cons > 0.8:
        score += 0.6
        evidence.append(f"clip_consistency: high ({clip_cons:.0%})")
    elif clip_cons > 0.6:
        score += 0.3

    # Coverage: 24/7 bot activity
    max_score += 0.7
    coverage = _safe_float(metrics, "coverage_pct", 0.0)
    if coverage > 0.9:
        score += 0.7
        evidence.append(f"coverage: 24/7 ({coverage:.0%})")
    elif coverage > 0.7:
        score += 0.5
        evidence.append(f"coverage: high ({coverage:.0%})")
    elif coverage > 0.5:
        score += 0.2

    # Fill frequency: high
    max_score += 0.5
    pct_sub_30 = _safe_nested_float(
        metrics, "fill_interval_stats", "pct_sub_30s", default=0.0
    )
    if pct_sub_30 > 0.5:
        score += 0.5
        evidence.append(f"fill_frequency: very high ({pct_sub_30:.0%} < 30s)")
    elif pct_sub_30 > 0.3:
        score += 0.3

    # Total trades: many
    max_score += 0.3
    total_trades = int(_safe_float(metrics, "total_trades", 0))
    if total_trades >= 30:
        score += 0.3
        evidence.append(f"trade_count: very high ({total_trades})")
    elif total_trades >= 15:
        score += 0.15

    confidence = score / max_score if max_score > 0 else 0.0
    return confidence, evidence


def _score_grid(metrics: dict) -> tuple[float, list[str]]:
    """Score wallet for grid bot strategy.

    Grid bots are characterized by systematic buying and selling on both
    sides with consistent position sizes. They are bot-like and typically
    operate across multiple markets.

    Indicators:
      - preferred_direction is 'mixed' — PRIMARY differentiator
      - clip_size_consistency > 0.8 (very consistent clips)
      - coverage_pct > 0.75 (near-24/7 activity)
      - multiple markets (>= 2) — key differentiator from scalper
      - moderate win rate (45-65%)
      - high total trades
    """
    evidence = []
    score = 0.0
    max_score = 0.0

    # Direction: mixed is the PRIMARY differentiator from scalper
    max_score += 1.5
    direction = _safe_get(metrics, "preferred_direction", "unknown")
    if direction == "mixed":
        score += 1.5
        evidence.append("direction: mixed (grid-like)")
    else:
        score += 0.0  # Not mixed = not grid

    # Clip consistency: very consistent (bot-like)
    max_score += 1.0
    clip_cons = _safe_float(metrics, "clip_size_consistency", 0.0)
    if clip_cons > 0.9:
        score += 1.0
        evidence.append(f"clip_consistency: very high ({clip_cons:.0%})")
    elif clip_cons > 0.8:
        score += 0.7
        evidence.append(f"clip_consistency: high ({clip_cons:.0%})")
    elif clip_cons > 0.6:
        score += 0.3

    # Multiple markets — key differentiator from single-market strategies
    max_score += 1.0
    markets = _safe_get(metrics, "markets_traded", [])
    if isinstance(markets, list):
        n_markets = len(markets)
        if n_markets >= 3:
            score += 1.0
            evidence.append(f"markets: multi-market ({n_markets})")
        elif n_markets == 2:
            score += 0.5
            evidence.append(f"markets: dual-market ({n_markets})")

    # Coverage: near 24/7 (bot-like)
    max_score += 0.8
    coverage = _safe_float(metrics, "coverage_pct", 0.0)
    if coverage > 0.9:
        score += 0.8
        evidence.append(f"coverage: 24/7 ({coverage:.0%})")
    elif coverage > 0.75:
        score += 0.6
        evidence.append(f"coverage: high ({coverage:.0%})")
    elif coverage > 0.5:
        score += 0.3

    # Win rate: moderate (grid bots win small amounts frequently)
    max_score += 0.5
    win_rate = _safe_float(metrics, "win_rate", 0.0)
    if 0.5 <= win_rate <= 0.65:
        score += 0.5
        evidence.append(f"win_rate: moderate ({win_rate:.1%})")
    elif 0.4 <= win_rate < 0.5 or 0.65 < win_rate <= 0.80:
        score += 0.25

    # Total trades: high
    max_score += 0.4
    total_trades = int(_safe_float(metrics, "total_trades", 0))
    if total_trades >= 30:
        score += 0.4
        evidence.append(f"trade_count: high ({total_trades})")
    elif total_trades >= 15:
        score += 0.2

    confidence = score / max_score if max_score > 0 else 0.0
    return confidence, evidence


# ---------------------------------------------------------------------------
# Main classification logic
# ---------------------------------------------------------------------------


def classify_wallet(metrics: dict, clusters: list) -> dict:
    """Classify a single wallet's trading strategy.

    Args:
        metrics: Dict of wallet metrics from wallet_metrics.compute_wallet_metrics().
        clusters: List of position cluster dicts from
            position_clustering.cluster_fills().

    Returns:
        Dict with:
            strategy: string label (momentum_scalper, mean_reversion,
                trend_follower, lp_consumer, grid, unknown, insufficient_data)
            confidence: float in [0, 1]
            evidence: list of strings describing supporting metrics
    """
    if not metrics and not clusters:
        logger.debug("No metrics or clusters provided")
        return {
            "strategy": STRATEGY_INSUFFICIENT_DATA,
            "confidence": 0.0,
            "evidence": ["no data available"],
        }

    # Check for insufficient data
    total_trades = int(_safe_float(metrics, "total_trades", 0))
    if total_trades < MIN_TRADES_FOR_CLASSIFICATION:
        logger.debug(
            "Insufficient data: %d trades (need %d)",
            total_trades,
            MIN_TRADES_FOR_CLASSIFICATION,
        )
        evidence = [
            f"insufficient_data: only {total_trades} trades (need {MIN_TRADES_FOR_CLASSIFICATION})"
        ]
        # Still try to classify but cap confidence
        if total_trades == 0:
            return {
                "strategy": STRATEGY_INSUFFICIENT_DATA,
                "confidence": 0.0,
                "evidence": evidence,
            }

        # For wallets with some trades but < threshold, try classification
        # but cap confidence low
        scores = _compute_all_scores(metrics, clusters)
        best_strategy, best_conf, best_evidence = _select_best(scores)

        # Cap confidence for insufficient data
        capped_conf = min(best_conf, 0.3)
        return {
            "strategy": STRATEGY_INSUFFICIENT_DATA,
            "confidence": capped_conf,
            "evidence": evidence
            + [f"best_guess: {best_strategy} (conf={best_conf:.2f} but capped)"],
        }

    # Score all strategies
    scores = _compute_all_scores(metrics, clusters)
    best_strategy, best_conf, best_evidence = _select_best(scores)

    # If best confidence is too low, classify as unknown
    if best_conf < 0.25:
        logger.debug(
            "Low confidence classification: %s (%.2f)", best_strategy, best_conf
        )
        return {
            "strategy": STRATEGY_UNKNOWN,
            "confidence": best_conf,
            "evidence": best_evidence
            + [f"low_confidence: best match was {best_strategy} at {best_conf:.0%}"],
        }

    logger.info("Classified wallet as %s (confidence=%.2f)", best_strategy, best_conf)
    return {
        "strategy": best_strategy,
        "confidence": round(best_conf, 4),
        "evidence": best_evidence,
    }


def _compute_all_scores(
    metrics: dict, clusters: list
) -> list[tuple[str, float, list[str]]]:
    """Compute scores for all strategy types.

    Returns list of (strategy_name, confidence, evidence) tuples.
    """
    scores = []

    conf, ev = _score_momentum_scalper(metrics)
    scores.append((STRATEGY_MOMENTUM_SCALPER, conf, ev))

    conf, ev = _score_trend_follower(metrics)
    scores.append((STRATEGY_TREND_FOLLOWER, conf, ev))

    conf, ev = _score_mean_reversion(metrics, clusters)
    scores.append((STRATEGY_MEAN_REVERSION, conf, ev))

    conf, ev = _score_lp_consumer(metrics)
    scores.append((STRATEGY_LP_CONSUMER, conf, ev))

    conf, ev = _score_grid(metrics)
    scores.append((STRATEGY_GRID, conf, ev))

    return scores


def _select_best(
    scores: list[tuple[str, float, list[str]]],
) -> tuple[str, float, list[str]]:
    """Select the highest-scoring strategy.

    Args:
        scores: List of (strategy, confidence, evidence) tuples.

    Returns:
        Tuple of (best_strategy, best_confidence, best_evidence).
    """
    if not scores:
        return STRATEGY_UNKNOWN, 0.0, ["no scores computed"]

    best = max(scores, key=lambda x: x[1])
    return best


def classify_strategies(wallets: list[dict]) -> list[dict]:
    """Classify multiple wallets in batch.

    Args:
        wallets: List of dicts, each with:
            address: wallet address string
            fills: list of fill dicts (Hyperliquid schema)

    Returns:
        List of classification result dicts, each with:
            wallet: wallet address string
            strategy: classified strategy type
            confidence: float in [0, 1]
            evidence: list of supporting metric strings
            metrics: the computed wallet metrics dict
    """
    from analysis.position_clustering import cluster_fills
    from analysis.wallet_metrics import compute_wallet_metrics

    results = []
    for wallet_data in wallets:
        address = wallet_data.get("address", "unknown")
        fills = wallet_data.get("fills", [])

        logger.info("Classifying wallet %s (%d fills)", address, len(fills))

        clusters = cluster_fills(fills)
        metrics = compute_wallet_metrics(clusters, fills)
        classification = classify_wallet(metrics, clusters)

        results.append(
            {
                "wallet": address,
                "strategy": classification["strategy"],
                "confidence": classification["confidence"],
                "evidence": classification["evidence"],
                "metrics": metrics,
            }
        )

    return results
