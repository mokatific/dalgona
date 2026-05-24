//! Strategy trait and implementations for the Dalgona trading system.
//!
//! This module defines the `Strategy` trait that all trading strategies must implement.
//! It also provides the `MomentumScalperStrategy` (extracted from the original `MomentumDetector`),
//! `LpConsumptionStrategy` (LP depth consumption detector from M1 blueprints),
//! and a centralized factory function for strategy instantiation.

use crate::signal::{MomentumDetector, MomentumSnapshot, PoolSnapshot, Signal};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Parameters for a trading strategy, used for display and validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    pub direction_bias: String,
    pub momentum_threshold_pct: f64,
    pub lookback_count: usize,
    pub scale_in_clips: u32,
    pub clip_size_usd: f64,
    pub max_hold_secs: u64,
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub trailing_stop_pct: f64,
    pub trailing_activation_pct: f64,
    pub cooldown_after_loss_secs: u64,
    pub use_native_tp_sl: bool,
}

impl StrategyParams {
    /// Validate that all parameter values are within acceptable ranges.
    /// Returns an error message for the first invalid parameter found.
    pub fn validate(&self) -> Result<(), String> {
        if self.momentum_threshold_pct <= 0.0 {
            return Err(format!(
                "momentum_threshold_pct must be > 0, got {}",
                self.momentum_threshold_pct
            ));
        }
        if self.lookback_count == 0 {
            return Err("lookback_count must be > 0".to_string());
        }
        if self.clip_size_usd <= 0.0 {
            return Err(format!(
                "clip_size_usd must be > 0, got {}",
                self.clip_size_usd
            ));
        }
        if self.take_profit_pct <= 0.0 {
            return Err(format!(
                "take_profit_pct must be > 0, got {}",
                self.take_profit_pct
            ));
        }
        if self.stop_loss_pct <= 0.0 {
            return Err(format!(
                "stop_loss_pct must be > 0, got {}",
                self.stop_loss_pct
            ));
        }
        if self.trailing_stop_pct < 0.0 {
            return Err(format!(
                "trailing_stop_pct must be >= 0, got {}",
                self.trailing_stop_pct
            ));
        }
        if self.trailing_activation_pct < 0.0 {
            return Err(format!(
                "trailing_activation_pct must be >= 0, got {}",
                self.trailing_activation_pct
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LP Consumption Strategy Parameters
// ---------------------------------------------------------------------------

/// Parameters specific to the LP Consumption Detector strategy.
///
/// All parameter values originate from the M1 blueprint:
/// `data/strategy-blueprints/lp-consumer.json`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpConsumptionParams {
    // --- Entry parameters ---
    // From blueprint: data/strategy-blueprints/lp-consumer.json#parameters.entry
    /// Minimum utilization velocity (change per tick) to trigger an entry signal.
    /// From blueprint: entry.threshold_pct = 0.5 (interpreted as velocity threshold)
    pub consumption_velocity_threshold: f64,

    /// Minimum LP concentration ratio (0.0-1.0) to consider the pool concentrated enough.
    /// From blueprint: entry.confirmation = "consumption_directional >= 70%"
    /// i.e., at least 70% of consumption must be in one direction.
    pub lp_concentration_min: f64,

    /// Number of consecutive ticks with directional consumption above threshold
    /// required to confirm entry. Provides signal stability filtering.
    pub confirmation_ticks: usize,

    /// Maximum utilization ratio (0.0-1.0) — don't enter if pool is already maxed out.
    pub max_utilization: f64,

    // --- Standard trading parameters ---
    // From blueprint: data/strategy-blueprints/lp-consumer.json#parameters.exit + risk
    /// Direction bias: "long", "short", or "neutral".
    #[serde(default = "default_direction_bias")]
    pub direction_bias: String,
    /// Position clip size in USD.
    /// From blueprint: clip_size_usd = 20904.74 (scaled down for paper trading)
    #[serde(default = "default_clip_size_usd")]
    pub clip_size_usd: f64,
    /// Maximum hold duration in seconds.
    /// From blueprint: exit.max_hold_secs = 3600
    #[serde(default = "default_max_hold_secs")]
    pub max_hold_secs: u64,
    /// Take-profit threshold percentage.
    /// From blueprint: exit.take_profit_pct = 2.0
    #[serde(default = "default_take_profit_pct")]
    pub take_profit_pct: f64,
    /// Stop-loss threshold percentage.
    /// From blueprint: exit.stop_loss_pct = 1.0
    #[serde(default = "default_stop_loss_pct")]
    pub stop_loss_pct: f64,
    /// Trailing stop percentage.
    /// From blueprint: exit.trailing_stop_pct = 0.8
    #[serde(default = "default_trailing_stop_pct")]
    pub trailing_stop_pct: f64,
    /// Trailing stop activation percentage.
    /// From blueprint: exit.trailing_activation_pct = 1.2
    #[serde(default = "default_trailing_activation_pct")]
    pub trailing_activation_pct: f64,
    /// Cooldown after a losing trade, in seconds.
    /// From blueprint: risk.cooldown_after_loss_secs = 300
    #[serde(default = "default_cooldown_after_loss_secs")]
    pub cooldown_after_loss_secs: u64,
    /// Whether to use native on-chain TP/SL trigger orders.
    #[serde(default = "default_use_native_tp_sl")]
    pub use_native_tp_sl: bool,
    /// Leverage for positions.
    /// From blueprint: leverage = 2.5
    #[serde(default = "default_leverage")]
    pub leverage: f64,
    /// Number of scale-in clips (for multi-clip entry).
    #[serde(default = "default_scale_in_clips")]
    pub scale_in_clips: u32,
}

fn default_direction_bias() -> String {
    "neutral".to_string()
}
fn default_clip_size_usd() -> f64 {
    100.0
}
fn default_max_hold_secs() -> u64 {
    3600
}
fn default_take_profit_pct() -> f64 {
    2.0
}
fn default_stop_loss_pct() -> f64 {
    1.0
}
fn default_trailing_stop_pct() -> f64 {
    0.8
}
fn default_trailing_activation_pct() -> f64 {
    1.2
}
fn default_cooldown_after_loss_secs() -> u64 {
    300
}
fn default_use_native_tp_sl() -> bool {
    true
}
fn default_leverage() -> f64 {
    2.5
}
fn default_scale_in_clips() -> u32 {
    1
}

impl LpConsumptionParams {
    /// Validate LP consumption parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.consumption_velocity_threshold <= 0.0 {
            return Err(format!(
                "consumption_velocity_threshold must be > 0, got {}",
                self.consumption_velocity_threshold
            ));
        }
        if self.lp_concentration_min <= 0.0 || self.lp_concentration_min > 1.0 {
            return Err(format!(
                "lp_concentration_min must be in (0, 1], got {}",
                self.lp_concentration_min
            ));
        }
        if self.confirmation_ticks == 0 {
            return Err("confirmation_ticks must be > 0".to_string());
        }
        if self.clip_size_usd <= 0.0 {
            return Err(format!(
                "clip_size_usd must be > 0, got {}",
                self.clip_size_usd
            ));
        }
        if self.take_profit_pct <= 0.0 {
            return Err(format!(
                "take_profit_pct must be > 0, got {}",
                self.take_profit_pct
            ));
        }
        if self.stop_loss_pct <= 0.0 {
            return Err(format!(
                "stop_loss_pct must be > 0, got {}",
                self.stop_loss_pct
            ));
        }
        Ok(())
    }

    /// Convert to the generic StrategyParams for use by engine/risk modules
    /// that need a uniform parameter interface.
    pub fn to_strategy_params(&self) -> StrategyParams {
        StrategyParams {
            direction_bias: self.direction_bias.clone(),
            momentum_threshold_pct: self.consumption_velocity_threshold,
            lookback_count: self.confirmation_ticks * 5, // rough equivalent
            scale_in_clips: self.scale_in_clips,
            clip_size_usd: self.clip_size_usd,
            max_hold_secs: self.max_hold_secs,
            take_profit_pct: self.take_profit_pct,
            stop_loss_pct: self.stop_loss_pct,
            trailing_stop_pct: self.trailing_stop_pct,
            trailing_activation_pct: self.trailing_activation_pct,
            cooldown_after_loss_secs: self.cooldown_after_loss_secs,
            use_native_tp_sl: self.use_native_tp_sl,
        }
    }
}

// ---------------------------------------------------------------------------
// Mean Reversion Strategy Parameters
// ---------------------------------------------------------------------------

/// Parameters specific to the Mean Reversion Scalper strategy.
///
/// Exit parameters derived from: data/strategy-blueprints/swing-trader.json#parameters.exit
/// (scaled down for scalper timeframes — mean reversion targets smaller moves).
/// Entry logic based on M1 strategy classification analysis:
/// oscillating long/short direction, short hold times, tight PnL range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionParams {
    // --- Entry parameters ---
    // Mean reversion concept: fade momentum spikes after reversal signals.
    // Entry fires when price deviates from SMA by threshold, then reversal tick confirmed.
    /// Number of price points to compute the simple moving average (SMA).
    /// From M1 analysis: mean-reversion wallets showed short hold times,
    /// indicating quick entries/exits around the mean.
    pub mean_lookback: usize,

    /// Minimum price deviation from SMA (%) to detect a "spike".
    /// A spike above SMA → potential SHORT entry.
    /// A spike below SMA → potential LONG entry.
    /// From M1 analysis: tight PnL range suggests moderate threshold.
    pub deviation_threshold_pct: f64,

    /// Number of consecutive ticks moving back toward the mean required
    /// to confirm a reversal after a spike is detected.
    /// From M1 analysis: moderate-to-high trade frequency supports quick confirmation.
    pub reversal_confirmation_ticks: usize,

    // --- Exit parameters ---
    // From blueprint: data/strategy-blueprints/swing-trader.json#parameters.exit
    // Scaled for mean reversion: tighter TP, no trailing, shorter holds.
    /// How close to the SMA the price must return (%) for a "mean return" exit.
    /// This is the primary exit condition for mean reversion.
    pub mean_tolerance_pct: f64,

    /// Direction bias: "long", "short", or "neutral".
    #[serde(default = "default_direction_bias")]
    pub direction_bias: String,
    /// Position clip size in USD.
    /// From M1 analysis: mean-reversion wallets used moderate clip sizes.
    #[serde(default = "default_mr_clip_size_usd")]
    pub clip_size_usd: f64,
    /// Maximum hold duration in seconds.
    /// From blueprint: swing-trader exit.max_hold_secs = 28800 (scaled to 1800 for scalper).
    #[serde(default = "default_mr_max_hold_secs")]
    pub max_hold_secs: u64,
    /// Take-profit threshold percentage.
    /// From blueprint: swing-trader exit.take_profit_pct = 4.0 (scaled to 1.0 for scalper).
    #[serde(default = "default_mr_take_profit_pct")]
    pub take_profit_pct: f64,
    /// Stop-loss threshold percentage.
    /// From blueprint: swing-trader exit.stop_loss_pct = 2.0 (scaled to 1.5 for scalper).
    #[serde(default = "default_mr_stop_loss_pct")]
    pub stop_loss_pct: f64,
    /// Trailing stop percentage (0.0 = disabled for mean reversion).
    /// Mean reversion targets a fixed level (the mean), not momentum continuation.
    #[serde(default)]
    pub trailing_stop_pct: f64,
    /// Trailing stop activation percentage (0.0 = disabled).
    #[serde(default)]
    pub trailing_activation_pct: f64,
    /// Cooldown after a losing trade, in seconds.
    /// From blueprint: swing-trader risk.cooldown_after_loss_secs = 900 (scaled to 300).
    #[serde(default = "default_cooldown_after_loss_secs")]
    pub cooldown_after_loss_secs: u64,
    /// Whether to use native on-chain TP/SL trigger orders.
    #[serde(default = "default_use_native_tp_sl")]
    pub use_native_tp_sl: bool,
    /// Leverage for positions.
    /// From M1 analysis: mean-reversion wallets used moderate leverage.
    #[serde(default = "default_mr_leverage")]
    pub leverage: f64,
    /// Number of scale-in clips (for multi-clip entry).
    #[serde(default = "default_scale_in_clips")]
    pub scale_in_clips: u32,
}

fn default_mr_clip_size_usd() -> f64 {
    100.0
}
fn default_mr_max_hold_secs() -> u64 {
    1800
}
fn default_mr_take_profit_pct() -> f64 {
    1.0
}
fn default_mr_stop_loss_pct() -> f64 {
    1.5
}
fn default_mr_leverage() -> f64 {
    3.0
}

impl MeanReversionParams {
    /// Validate mean reversion parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.mean_lookback == 0 {
            return Err("mean_lookback must be > 0".to_string());
        }
        if self.deviation_threshold_pct <= 0.0 {
            return Err(format!(
                "deviation_threshold_pct must be > 0, got {}",
                self.deviation_threshold_pct
            ));
        }
        if self.reversal_confirmation_ticks == 0 {
            return Err("reversal_confirmation_ticks must be > 0".to_string());
        }
        if self.mean_tolerance_pct < 0.0 {
            return Err(format!(
                "mean_tolerance_pct must be >= 0, got {}",
                self.mean_tolerance_pct
            ));
        }
        if self.clip_size_usd <= 0.0 {
            return Err(format!(
                "clip_size_usd must be > 0, got {}",
                self.clip_size_usd
            ));
        }
        if self.take_profit_pct <= 0.0 {
            return Err(format!(
                "take_profit_pct must be > 0, got {}",
                self.take_profit_pct
            ));
        }
        if self.stop_loss_pct <= 0.0 {
            return Err(format!(
                "stop_loss_pct must be > 0, got {}",
                self.stop_loss_pct
            ));
        }
        Ok(())
    }

    /// Convert to the generic StrategyParams for use by engine/risk modules
    /// that need a uniform parameter interface.
    pub fn to_strategy_params(&self) -> StrategyParams {
        StrategyParams {
            direction_bias: self.direction_bias.clone(),
            momentum_threshold_pct: self.deviation_threshold_pct,
            lookback_count: self.mean_lookback,
            scale_in_clips: self.scale_in_clips,
            clip_size_usd: self.clip_size_usd,
            max_hold_secs: self.max_hold_secs,
            take_profit_pct: self.take_profit_pct,
            stop_loss_pct: self.stop_loss_pct,
            trailing_stop_pct: self.trailing_stop_pct,
            trailing_activation_pct: self.trailing_activation_pct,
            cooldown_after_loss_secs: self.cooldown_after_loss_secs,
            use_native_tp_sl: self.use_native_tp_sl,
        }
    }
}

// ---------------------------------------------------------------------------
// Trend Follower Strategy Parameters
// ---------------------------------------------------------------------------

/// Parameters specific to the Trend Follower strategy.
///
/// This strategy enters on confirmed momentum breakouts with wider stops,
/// trailing exits, and longer holds than scalper strategies. It accepts a
/// lower win rate in exchange for larger average winners.
///
/// Entry logic: price velocity must exceed a *higher* threshold than the
/// momentum scalper ( breakout confirmation ), AND the trend must be
/// confirmed over multiple consecutive ticks.
///
/// Exit logic: wider trailing stop, generous time stop. No mean-return exit.
///
/// Sensible defaults are used since M1 blueprints did not identify a
/// dedicated trend-following pattern. Parameters are tuned for Flash Trade
/// perp markets with 5-second polling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendFollowerParams {
    // --- Entry parameters ---
    /// Velocity threshold (%) for a "breakout" — higher than the momentum scalper's
    /// threshold. A true breakout should show sustained, strong directional momentum.
    /// Default 0.25 %/tick (vs 0.15 for the scalper).
    pub breakout_threshold_pct: f64,

    /// Number of consecutive ticks above breakout threshold required to confirm entry.
    /// This filters out brief spikes that don't become sustained trends.
    /// Default 4 ticks (20 seconds at 5s poll interval).
    pub confirmation_ticks: usize,

    /// Minimum number of price points required before any entry signal is generated.
    /// The strategy needs enough history to compute a reliable trend.
    /// Default 30.
    pub min_price_count: usize,

    // --- Exit parameters ---
    /// Direction bias: "long", "short", or "neutral".
    #[serde(default = "default_direction_bias")]
    pub direction_bias: String,
    /// Position clip size in USD.
    #[serde(default = "default_tf_clip_size_usd")]
    pub clip_size_usd: f64,
    /// Maximum hold duration in seconds — longer than scalper (7200 vs 1800).
    #[serde(default = "default_tf_max_hold_secs")]
    pub max_hold_secs: u64,
    /// Take-profit threshold percentage — wider than scalper (5.0 vs 2.5).
    #[serde(default = "default_tf_take_profit_pct")]
    pub take_profit_pct: f64,
    /// Stop-loss threshold percentage — wider than scalper (2.0 vs 1.0).
    #[serde(default = "default_tf_stop_loss_pct")]
    pub stop_loss_pct: f64,
    /// Trailing stop percentage — wider than scalper (1.5 vs 0.8).
    #[serde(default = "default_tf_trailing_stop_pct")]
    pub trailing_stop_pct: f64,
    /// Trailing stop activation percentage — when to start trailing (2.5 vs 1.5).
    #[serde(default = "default_tf_trailing_activation_pct")]
    pub trailing_activation_pct: f64,
    /// Cooldown after a losing trade, in seconds.
    #[serde(default = "default_cooldown_after_loss_secs")]
    pub cooldown_after_loss_secs: u64,
    /// Whether to use native on-chain TP/SL trigger orders.
    #[serde(default = "default_use_native_tp_sl")]
    pub use_native_tp_sl: bool,
    /// Leverage for positions.
    #[serde(default = "default_tf_leverage")]
    pub leverage: f64,
    /// Number of scale-in clips (for multi-clip entry).
    #[serde(default = "default_scale_in_clips")]
    pub scale_in_clips: u32,
}

fn default_tf_clip_size_usd() -> f64 {
    100.0
}
fn default_tf_max_hold_secs() -> u64 {
    7200
}
fn default_tf_take_profit_pct() -> f64 {
    5.0
}
fn default_tf_stop_loss_pct() -> f64 {
    2.0
}
fn default_tf_trailing_stop_pct() -> f64 {
    1.5
}
fn default_tf_trailing_activation_pct() -> f64 {
    2.5
}
fn default_tf_leverage() -> f64 {
    5.0
}

impl TrendFollowerParams {
    /// Validate trend follower parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.breakout_threshold_pct <= 0.0 {
            return Err(format!(
                "breakout_threshold_pct must be > 0, got {}",
                self.breakout_threshold_pct
            ));
        }
        if self.confirmation_ticks == 0 {
            return Err("confirmation_ticks must be > 0".to_string());
        }
        if self.min_price_count == 0 {
            return Err("min_price_count must be > 0".to_string());
        }
        if self.clip_size_usd <= 0.0 {
            return Err(format!(
                "clip_size_usd must be > 0, got {}",
                self.clip_size_usd
            ));
        }
        if self.take_profit_pct <= 0.0 {
            return Err(format!(
                "take_profit_pct must be > 0, got {}",
                self.take_profit_pct
            ));
        }
        if self.stop_loss_pct <= 0.0 {
            return Err(format!(
                "stop_loss_pct must be > 0, got {}",
                self.stop_loss_pct
            ));
        }
        if self.trailing_stop_pct < 0.0 {
            return Err(format!(
                "trailing_stop_pct must be >= 0, got {}",
                self.trailing_stop_pct
            ));
        }
        Ok(())
    }

    /// Convert to the generic StrategyParams for use by engine/risk modules
    /// that need a uniform parameter interface.
    pub fn to_strategy_params(&self) -> StrategyParams {
        StrategyParams {
            direction_bias: self.direction_bias.clone(),
            momentum_threshold_pct: self.breakout_threshold_pct,
            lookback_count: self.min_price_count,
            scale_in_clips: self.scale_in_clips,
            clip_size_usd: self.clip_size_usd,
            max_hold_secs: self.max_hold_secs,
            take_profit_pct: self.take_profit_pct,
            stop_loss_pct: self.stop_loss_pct,
            trailing_stop_pct: self.trailing_stop_pct,
            trailing_activation_pct: self.trailing_activation_pct,
            cooldown_after_loss_secs: self.cooldown_after_loss_secs,
            use_native_tp_sl: self.use_native_tp_sl,
        }
    }
}

// ---------------------------------------------------------------------------
// PositionContext
// ---------------------------------------------------------------------------

/// Context passed to `detect_exit` bundling all position and exit-threshold parameters.
/// This avoids passing 8+ individual parameters to the exit method.
#[derive(Debug, Clone)]
pub struct PositionContext {
    /// Whether this is a long position.
    pub is_long: bool,
    /// Entry price of the position.
    pub entry_price: f64,
    /// Current price of the position.
    pub current_price: f64,
    /// Peak price seen since entry (highest for longs, lowest for shorts).
    pub peak_price: f64,
    /// How long the position has been held, in seconds.
    pub hold_secs: u64,
    /// Maximum hold duration in seconds before time-stop triggers.
    pub max_hold_secs: u64,
    /// Take-profit threshold as a percentage.
    pub take_profit_pct: f64,
    /// Stop-loss threshold as a percentage.
    pub stop_loss_pct: f64,
    /// Trailing stop percentage.
    pub trailing_stop_pct: f64,
    /// Trailing stop activation percentage (peak profit must exceed this).
    pub trailing_activation_pct: f64,
}

// ---------------------------------------------------------------------------
// Strategy Trait
// ---------------------------------------------------------------------------

/// The core Strategy trait that all trading strategies must implement.
///
/// This trait is object-safe and `Send + Sync` so it can be used as `Box<dyn Strategy>`
/// inside async contexts.
pub trait Strategy: Send + Sync {
    /// Return the canonical name of this strategy (e.g., "momentum-scalper").
    fn name(&self) -> &str;

    /// Detect an entry signal given the latest market snapshot.
    ///
    /// The strategy is allowed mutable access because it maintains internal state
    /// (e.g., price buffers for the momentum detector).
    fn detect_entry(&mut self, snapshot: &MomentumSnapshot) -> Signal;

    /// Detect whether an exit should be triggered for an open position.
    ///
    /// Returns `Some(Signal::ExitLong/ExitShort)` if an exit condition is met,
    /// or `None` to continue holding.
    fn detect_exit(&self, snapshot: &MomentumSnapshot, context: &PositionContext)
    -> Option<Signal>;

    /// Return the strategy's parameter set (for display, logging, validation).
    fn parameters(&self) -> &StrategyParams;

    /// Push a new price point into the strategy's internal buffer.
    /// This is called by the engine on each tick before `detect_entry`/`detect_exit`.
    fn push_price(&mut self, price: f64, timestamp_ms: i64);

    /// Produce a `MomentumSnapshot` from the current internal state.
    /// Used for logging and debugging.
    fn snapshot(&self) -> MomentumSnapshot;
}

// ---------------------------------------------------------------------------
// MomentumScalperStrategy
// ---------------------------------------------------------------------------

/// The original momentum scalper strategy, extracted from `MomentumDetector`.
///
/// This wraps a `MomentumDetector` and implements the `Strategy` trait.
/// The logic is identical to the pre-refactor `MomentumDetector::detect_signal`
/// and `MomentumDetector::detect_exit` methods.
pub struct MomentumScalperStrategy {
    detector: MomentumDetector,
    params: StrategyParams,
}

impl MomentumScalperStrategy {
    pub fn new(params: StrategyParams) -> Self {
        let detector = MomentumDetector::new(params.momentum_threshold_pct, params.lookback_count);
        Self { detector, params }
    }
}

impl Strategy for MomentumScalperStrategy {
    fn name(&self) -> &str {
        "momentum-scalper"
    }

    fn detect_entry(&mut self, snapshot: &MomentumSnapshot) -> Signal {
        self.detector.detect_signal(snapshot)
    }

    fn detect_exit(&self, snapshot: &MomentumSnapshot, ctx: &PositionContext) -> Option<Signal> {
        self.detector.detect_exit(
            snapshot,
            ctx.is_long,
            ctx.entry_price,
            ctx.current_price,
            ctx.peak_price,
            ctx.hold_secs,
            ctx.max_hold_secs,
            ctx.take_profit_pct,
            ctx.stop_loss_pct,
            ctx.trailing_stop_pct,
            ctx.trailing_activation_pct,
        )
    }

    fn parameters(&self) -> &StrategyParams {
        &self.params
    }

    fn push_price(&mut self, price: f64, timestamp_ms: i64) {
        self.detector.push_price(price, timestamp_ms);
    }

    fn snapshot(&self) -> MomentumSnapshot {
        self.detector.analyze()
    }
}

// ---------------------------------------------------------------------------
// LpConsumptionStrategy
// ---------------------------------------------------------------------------

/// LP Consumption Detector strategy.
///
/// Detects when a single LP's depth is being consumed in one direction via pool
/// utilization data from the Flash Trade API. This is a market-structure arbitrage
/// strategy: find illiquid markets where a dominant LP provides most of the depth,
/// detect when that LP is being consumed, and ride the momentum.
///
/// **Entry conditions:**
/// 1. Pool data is available (no entry without pool data)
/// 2. One-sided utilization velocity exceeds `consumption_velocity_threshold`
/// 3. Directional concentration ratio exceeds `lp_concentration_min`
/// 4. Confirmed over `confirmation_ticks` consecutive ticks
/// 5. Utilization is below `max_utilization` (don't enter a fully consumed pool)
/// 6. Entry direction matches consumption direction
///
/// **Exit conditions (priority order):**
/// 1. Stop-loss (absolute loss from entry)
/// 2. Take-profit (absolute gain from entry)
/// 3. Trailing stop (after activation threshold reached)
/// 4. Time stop (max hold duration exceeded)
/// 5. Consumption stall (velocity drops below threshold while in position)
/// 6. Reversal (opposite-side consumption detected)
///
/// Blueprint source: `data/strategy-blueprints/lp-consumer.json`
pub struct LpConsumptionStrategy {
    params: LpConsumptionParams,
    generic_params: StrategyParams,
    /// Internal momentum detector for price-based exit signals.
    detector: MomentumDetector,
    /// Rolling window of utilization velocity readings for stall detection.
    velocity_history: VecDeque<f64>,
    /// Number of consecutive ticks with consumption exceeding threshold in the
    /// same direction. Positive = long consumption, negative = short consumption.
    consecutive_consumption: i32,
    /// Latest pool snapshot for snapshot() reporting.
    last_pool: Option<PoolSnapshot>,
}

impl LpConsumptionStrategy {
    pub fn new(params: LpConsumptionParams) -> Self {
        let generic = params.to_strategy_params();
        let detector = MomentumDetector::new(
            params.consumption_velocity_threshold,
            params.confirmation_ticks * 2,
        );
        Self {
            generic_params: generic,
            detector,
            velocity_history: VecDeque::with_capacity(60),
            consecutive_consumption: 0,
            last_pool: None,
            params,
        }
    }

    /// Compute the directional concentration ratio.
    /// Returns (long_ratio, short_ratio) where each is 0.0-1.0 and they sum to 1.0
    /// (or close to it if both are near zero).
    ///
    /// "Directional" means: of the total consumption happening, how much is on
    /// each side? If long_utilization_velocity = 0.8 and short = 0.2, then
    /// long_ratio = 0.8 / (0.8 + 0.2) = 0.8 = 80% concentrated on the long side.
    fn directional_concentration(pool: &PoolSnapshot) -> (f64, f64) {
        let long_v = pool.long_utilization_velocity.abs();
        let short_v = pool.short_utilization_velocity.abs();
        let total = long_v + short_v;
        if total < 1e-10 {
            return (0.5, 0.5);
        }
        (long_v / total, short_v / total)
    }
}

impl Strategy for LpConsumptionStrategy {
    fn name(&self) -> &str {
        "lp-consumption"
    }

    fn detect_entry(&mut self, snapshot: &MomentumSnapshot) -> Signal {
        let pool = match &snapshot.pool_data {
            Some(p) => p,
            None => {
                // No pool data available — cannot detect LP consumption.
                // Log at debug level to avoid spamming on every tick.
                debug!("[lp-consumption] No pool data in snapshot, skipping entry detection");
                return Signal::NoSignal;
            }
        };

        self.last_pool = Some(pool.clone());

        // Compute directional concentration
        let (long_conc, short_conc) = Self::directional_concentration(pool);

        // Determine which side has higher velocity
        let long_velocity = pool.long_utilization_velocity;
        let short_velocity = pool.short_utilization_velocity;
        let max_velocity = long_velocity.abs().max(short_velocity.abs());

        // Track consecutive consumption ticks
        // Positive = long-side being consumed (enter LONG), negative = short-side (enter SHORT)
        let direction = if long_velocity > short_velocity.abs()
            && long_conc >= self.params.lp_concentration_min
        {
            // Long utilization growing fast = bid side being consumed = price likely up = LONG
            1
        } else if short_velocity > long_velocity.abs()
            && short_conc >= self.params.lp_concentration_min
        {
            // Short utilization growing fast = ask side being consumed = price likely down = SHORT
            -1
        } else {
            // No clear directional consumption above concentration threshold
            if self.consecutive_consumption != 0 {
                debug!(
                    "[lp-consumption] Consecutive reset: long_conc={:.2} short_conc={:.2} \
                     long_v={:.4} short_v={:.4}",
                    long_conc, short_conc, long_velocity, short_velocity
                );
            }
            self.consecutive_consumption = 0;
            return Signal::NoSignal;
        };

        // Check velocity threshold
        if max_velocity < self.params.consumption_velocity_threshold {
            if self.consecutive_consumption != 0 {
                debug!(
                    "[lp-consumption] Velocity below threshold: {:.4} < {:.4}",
                    max_velocity, self.params.consumption_velocity_threshold
                );
            }
            self.consecutive_consumption = 0;
            return Signal::NoSignal;
        }

        // Check max utilization — don't enter a fully consumed pool
        let current_utilization = pool.long_utilization.max(pool.short_utilization);
        if current_utilization >= self.params.max_utilization {
            debug!(
                "[lp-consumption] Utilization too high: {:.2} >= {:.2}",
                current_utilization, self.params.max_utilization
            );
            self.consecutive_consumption = 0;
            return Signal::NoSignal;
        }

        // Track consecutive direction
        let prev_dir = self.consecutive_consumption.signum() as i32;
        if prev_dir == direction {
            self.consecutive_consumption += direction;
        } else {
            self.consecutive_consumption = direction;
        }

        // Record velocity for stall detection
        self.velocity_history.push_back(max_velocity);
        if self.velocity_history.len() > 60 {
            self.velocity_history.pop_front();
        }

        let consec_count = self.consecutive_consumption.abs() as usize;
        debug!(
            "[lp-consumption] Consecutive ticks: {} (need {}), velocity={:.4}, \
             long_conc={:.2}, short_conc={:.2}, utilization={:.2}",
            consec_count,
            self.params.confirmation_ticks,
            max_velocity,
            long_conc,
            short_conc,
            current_utilization,
        );

        // Check confirmation — need N consecutive ticks of directional consumption
        if consec_count < self.params.confirmation_ticks {
            return Signal::NoSignal;
        }

        // ENTRY SIGNAL
        let strength = (max_velocity / self.params.consumption_velocity_threshold * 50.0)
            .min(100.0)
            .max(50.0);

        if direction > 0 {
            info!(
                "[lp-consumption] LONG signal: velocity={:.4} (threshold={:.4}), \
                 concentration={:.0}%, consecutive={}, utilization={:.2}",
                max_velocity,
                self.params.consumption_velocity_threshold,
                long_conc * 100.0,
                consec_count,
                current_utilization,
            );
            Signal::MomentumLong {
                strength,
                velocity_pct: max_velocity,
            }
        } else {
            info!(
                "[lp-consumption] SHORT signal: velocity={:.4} (threshold={:.4}), \
                 concentration={:.0}%, consecutive={}, utilization={:.2}",
                max_velocity,
                self.params.consumption_velocity_threshold,
                short_conc * 100.0,
                consec_count,
                current_utilization,
            );
            Signal::MomentumShort {
                strength,
                velocity_pct: max_velocity,
            }
        }
    }

    fn detect_exit(&self, snapshot: &MomentumSnapshot, ctx: &PositionContext) -> Option<Signal> {
        // Standard SL/TP/trailing/time exits (same priority as momentum scalper)
        let pnl_pct = if ctx.is_long {
            (ctx.current_price - ctx.entry_price) / ctx.entry_price * 100.0
        } else {
            (ctx.entry_price - ctx.current_price) / ctx.entry_price * 100.0
        };

        let peak_profit_pct = if ctx.entry_price > 0.0 {
            if ctx.is_long {
                (ctx.peak_price - ctx.entry_price) / ctx.entry_price * 100.0
            } else {
                (ctx.entry_price - ctx.peak_price) / ctx.entry_price * 100.0
            }
        } else {
            0.0
        };

        let retracement_pct = if ctx.peak_price > 0.0 && peak_profit_pct > 0.0 {
            if ctx.is_long {
                (ctx.peak_price - ctx.current_price) / ctx.peak_price * 100.0
            } else {
                (ctx.current_price - ctx.peak_price) / ctx.peak_price * 100.0
            }
        } else {
            0.0
        };

        // 1. Stop loss (highest priority)
        if pnl_pct <= -ctx.stop_loss_pct {
            warn!(
                "[lp-consumption] STOP LOSS: pnl={:.2}%, threshold=-{:.2}%",
                pnl_pct, ctx.stop_loss_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::StopLoss,
            ));
        }

        // 2. Take profit
        if pnl_pct >= ctx.take_profit_pct {
            info!(
                "[lp-consumption] TAKE PROFIT: pnl={:.2}%, threshold={:.2}%",
                pnl_pct, ctx.take_profit_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TakeProfit,
            ));
        }

        // 3. Trailing stop
        if peak_profit_pct >= ctx.trailing_activation_pct
            && retracement_pct >= ctx.trailing_stop_pct
        {
            warn!(
                "[lp-consumption] TRAILING STOP: retracement={:.2}%, trail={:.2}%, peak_profit={:.2}%",
                retracement_pct, ctx.trailing_stop_pct, peak_profit_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TrailingStop,
            ));
        }

        // 4. Time stop
        if ctx.hold_secs >= ctx.max_hold_secs {
            warn!(
                "[lp-consumption] TIME STOP: held {}s, max={}s",
                ctx.hold_secs, ctx.max_hold_secs
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TimeStop,
            ));
        }

        // 5. LP-specific exit: consumption stall
        // If pool data shows velocity dropping below threshold, the LP consumption
        // momentum that triggered entry has stalled → exit
        if let Some(ref pool) = snapshot.pool_data {
            let relevant_velocity = if ctx.is_long {
                pool.long_utilization_velocity
            } else {
                pool.short_utilization_velocity
            };

            if relevant_velocity < self.params.consumption_velocity_threshold * 0.3 {
                debug!(
                    "[lp-consumption] Consumption stall: velocity={:.4} < {:.4}",
                    relevant_velocity,
                    self.params.consumption_velocity_threshold * 0.3
                );
                // Only exit if we're in profit or held long enough (avoid stalling at a loss)
                if pnl_pct > 0.0 || ctx.hold_secs > 300 {
                    return Some(exit_signal(
                        ctx.is_long,
                        crate::signal::ExitReason::MomentumLost,
                    ));
                }
            }
        }

        // 6. Reversal detection: opposite-side consumption is accelerating
        if let Some(ref pool) = snapshot.pool_data {
            let (long_conc, short_conc) = Self::directional_concentration(pool);
            let reversal_detected = if ctx.is_long {
                short_conc > self.params.lp_concentration_min
                    && pool.short_utilization_velocity > self.params.consumption_velocity_threshold
            } else {
                long_conc > self.params.lp_concentration_min
                    && pool.long_utilization_velocity > self.params.consumption_velocity_threshold
            };

            if reversal_detected {
                warn!(
                    "[lp-consumption] REVERSAL detected: {} side consumption accelerating",
                    if ctx.is_long { "short" } else { "long" }
                );
                return Some(exit_signal(
                    ctx.is_long,
                    crate::signal::ExitReason::ReversalDetected,
                ));
            }
        }

        None
    }

    fn parameters(&self) -> &StrategyParams {
        &self.generic_params
    }

    fn push_price(&mut self, price: f64, timestamp_ms: i64) {
        self.detector.push_price(price, timestamp_ms);
    }

    fn snapshot(&self) -> MomentumSnapshot {
        let mut snap = self.detector.analyze();
        // Layer in the pool data if we have it
        if self.last_pool.is_some() && snap.pool_data.is_none() {
            snap.pool_data = self.last_pool.clone();
        }
        snap
    }
}

/// Helper to create an exit signal.
fn exit_signal(is_long: bool, reason: crate::signal::ExitReason) -> Signal {
    if is_long {
        Signal::ExitLong { reason }
    } else {
        Signal::ExitShort { reason }
    }
}

// ---------------------------------------------------------------------------
// MeanReversionStrategy
// ---------------------------------------------------------------------------

/// Direction of a detected spike relative to the SMA.
#[derive(Debug, Clone, PartialEq)]
enum SpikeDirection {
    /// Price spiked above SMA → looking for SHORT entry (fade the spike).
    Above,
    /// Price spiked below SMA → looking for LONG entry (fade the spike).
    Below,
}

/// Mean Reversion Scalper strategy.
///
/// Fades momentum spikes by entering in the opposite direction after a sharp
/// price move away from the mean (SMA) followed by a reversal tick. This is
/// a contrarian strategy: it bets that extreme deviations from the mean will
/// revert.
///
/// **Entry conditions:**
/// 1. Sufficient price history (>= mean_lookback prices)
/// 2. Price deviates from SMA by more than deviation_threshold_pct (spike)
/// 3. Reversal confirmed: N consecutive ticks moving back toward the mean
/// 4. Entry direction is opposite to the spike (spike up → SHORT, spike down → LONG)
///
/// **Exit conditions (priority order):**
/// 1. Stop-loss (absolute loss from entry)
/// 2. Mean return: price returns to within mean_tolerance_pct of the computed SMA
/// 3. Take-profit (absolute gain from entry)
/// 4. Time stop (max hold duration exceeded)
///
/// Note: No trailing stop — mean reversion targets a fixed level (the mean),
/// not momentum continuation.
///
/// Blueprint source: derived from M1 analysis (mean-reversion wallets) and
/// data/strategy-blueprints/swing-trader.json for exit/risk parameter ranges.
pub struct MeanReversionStrategy {
    params: MeanReversionParams,
    generic_params: StrategyParams,
    /// Rolling price buffer for SMA calculation.
    prices: VecDeque<crate::signal::PricePoint>,
    /// Current spike state, if any.
    spike_state: Option<SpikeDirection>,
    /// Number of consecutive reversal ticks counted after a spike.
    reversal_ticks: usize,
}

impl MeanReversionStrategy {
    pub fn new(params: MeanReversionParams) -> Self {
        let generic = params.to_strategy_params();
        let capacity = params.mean_lookback * 2;
        Self {
            generic_params: generic,
            prices: VecDeque::with_capacity(capacity),
            spike_state: None,
            reversal_ticks: 0,
            params,
        }
    }

    /// Compute the simple moving average over the last `lookback` prices.
    /// Returns None if there are fewer than `lookback` prices in the buffer.
    fn compute_sma(&self, lookback: usize) -> Option<f64> {
        if self.prices.len() < lookback {
            return None;
        }
        let sum: f64 = self
            .prices
            .iter()
            .rev()
            .take(lookback)
            .map(|p| p.price)
            .sum();
        Some(sum / lookback as f64)
    }

    /// Get the previous price (second-to-last in buffer).
    fn prev_price(&self) -> Option<f64> {
        if self.prices.len() < 2 {
            return None;
        }
        // prices[len-2] is the second-to-last
        self.prices.iter().rev().nth(1).map(|p| p.price)
    }

    /// Get the current (most recent) price.
    fn current_price(&self) -> Option<f64> {
        self.prices.back().map(|p| p.price)
    }
}

impl Strategy for MeanReversionStrategy {
    fn name(&self) -> &str {
        "mean-reversion"
    }

    fn detect_entry(&mut self, _snapshot: &MomentumSnapshot) -> Signal {
        let current_price = match self.current_price() {
            Some(p) => p,
            None => return Signal::NoSignal,
        };

        let sma = match self.compute_sma(self.params.mean_lookback) {
            Some(s) => s,
            None => {
                debug!(
                    "[mean-reversion] Insufficient price history: {}/{} prices",
                    self.prices.len(),
                    self.params.mean_lookback
                );
                return Signal::NoSignal;
            }
        };

        if sma <= 0.0 {
            return Signal::NoSignal;
        }

        // Compute deviation from SMA as a percentage
        let deviation_pct = (current_price - sma) / sma * 100.0;

        // Check for spike detection
        if deviation_pct > self.params.deviation_threshold_pct {
            // Spike above SMA → potential SHORT entry
            if self.spike_state != Some(SpikeDirection::Above) {
                self.spike_state = Some(SpikeDirection::Above);
                self.reversal_ticks = 0;
                debug!(
                    "[mean-reversion] Spike ABOVE detected: price={:.2}, sma={:.2}, deviation={:.2}% (threshold={:.2}%)",
                    current_price, sma, deviation_pct, self.params.deviation_threshold_pct
                );
            }
        } else if deviation_pct < -self.params.deviation_threshold_pct {
            // Spike below SMA → potential LONG entry
            if self.spike_state != Some(SpikeDirection::Below) {
                self.spike_state = Some(SpikeDirection::Below);
                self.reversal_ticks = 0;
                debug!(
                    "[mean-reversion] Spike BELOW detected: price={:.2}, sma={:.2}, deviation={:.2}% (threshold={:.2}%)",
                    current_price, sma, deviation_pct, self.params.deviation_threshold_pct
                );
            }
        }

        // Check for reversal confirmation
        if let Some(ref spike_dir) = self.spike_state {
            let prev_price = match self.prev_price() {
                Some(p) => p,
                None => return Signal::NoSignal,
            };

            // Is this tick moving back toward the mean?
            let moving_toward_mean = match spike_dir {
                SpikeDirection::Above => {
                    // Price spiked above, reversal means price is coming down (current < prev)
                    current_price < prev_price
                }
                SpikeDirection::Below => {
                    // Price spiked below, reversal means price is coming up (current > prev)
                    current_price > prev_price
                }
            };

            if moving_toward_mean {
                self.reversal_ticks += 1;
                debug!(
                    "[mean-reversion] Reversal tick {} (need {}): price={:.2}, prev={:.2}, sma={:.2}",
                    self.reversal_ticks,
                    self.params.reversal_confirmation_ticks,
                    current_price,
                    prev_price,
                    sma
                );

                if self.reversal_ticks >= self.params.reversal_confirmation_ticks {
                    // CONFIRMED REVERSAL — generate entry signal
                    // Fading the spike: spike above → SHORT, spike below → LONG
                    let strength = (deviation_pct.abs() / self.params.deviation_threshold_pct
                        * 50.0)
                        .min(100.0)
                        .max(50.0);
                    let velocity_pct = deviation_pct.abs();

                    // Clone spike_dir before resetting state
                    let spike_dir_clone = spike_dir.clone();

                    // Reset spike state after generating signal
                    self.spike_state = None;
                    self.reversal_ticks = 0;

                    match spike_dir_clone {
                        SpikeDirection::Above => {
                            info!(
                                "[mean-reversion] SHORT signal: price={:.2}, sma={:.2}, deviation={:.2}%, reversal_ticks={}",
                                current_price,
                                sma,
                                deviation_pct,
                                self.params.reversal_confirmation_ticks
                            );
                            Signal::MomentumShort {
                                strength,
                                velocity_pct,
                            }
                        }
                        SpikeDirection::Below => {
                            info!(
                                "[mean-reversion] LONG signal: price={:.2}, sma={:.2}, deviation={:.2}%, reversal_ticks={}",
                                current_price,
                                sma,
                                deviation_pct,
                                self.params.reversal_confirmation_ticks
                            );
                            Signal::MomentumLong {
                                strength,
                                velocity_pct,
                            }
                        }
                    }
                } else {
                    Signal::NoSignal
                }
            } else {
                // Not moving toward mean — if still in deviation zone, keep spike state;
                // if back within threshold, reset
                if deviation_pct.abs() <= self.params.deviation_threshold_pct {
                    debug!(
                        "[mean-reversion] Spike expired: deviation {:.2}% back within threshold {:.2}%",
                        deviation_pct, self.params.deviation_threshold_pct
                    );
                    self.spike_state = None;
                    self.reversal_ticks = 0;
                }
                // If spike is still active but this tick moved away from mean, don't reset
                // (allow resumption of reversal counting on next tick)
                Signal::NoSignal
            }
        } else {
            // No active spike — check if we're in a gradual move (no signal for mean reversion)
            Signal::NoSignal
        }
    }

    fn detect_exit(&self, _snapshot: &MomentumSnapshot, ctx: &PositionContext) -> Option<Signal> {
        let current_price = ctx.current_price;

        // PnL from entry
        let pnl_pct = if ctx.is_long {
            (current_price - ctx.entry_price) / ctx.entry_price * 100.0
        } else {
            (ctx.entry_price - current_price) / ctx.entry_price * 100.0
        };

        // 1. Stop loss (highest priority)
        if pnl_pct <= -ctx.stop_loss_pct {
            warn!(
                "[mean-reversion] STOP LOSS: pnl={:.2}%, threshold=-{:.2}%",
                pnl_pct, ctx.stop_loss_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::StopLoss,
            ));
        }

        // 2. Mean return: price returns to within tolerance of computed mean
        if let Some(sma) = self.compute_sma(self.params.mean_lookback) {
            if sma > 0.0 {
                let deviation_from_mean = (current_price - sma).abs() / sma * 100.0;
                if deviation_from_mean <= self.params.mean_tolerance_pct {
                    info!(
                        "[mean-reversion] MEAN RETURN exit: price={:.2}, sma={:.2}, deviation={:.2}% (tolerance={:.2}%), pnl={:.2}%",
                        current_price,
                        sma,
                        deviation_from_mean,
                        self.params.mean_tolerance_pct,
                        pnl_pct
                    );
                    return Some(exit_signal(
                        ctx.is_long,
                        crate::signal::ExitReason::TakeProfit,
                    ));
                }
            }
        }

        // 3. Take profit (absolute gain from entry)
        if pnl_pct >= ctx.take_profit_pct {
            info!(
                "[mean-reversion] TAKE PROFIT: pnl={:.2}%, threshold={:.2}%",
                pnl_pct, ctx.take_profit_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TakeProfit,
            ));
        }

        // 4. Time stop
        if ctx.hold_secs >= ctx.max_hold_secs {
            warn!(
                "[mean-reversion] TIME STOP: held {}s, max={}s",
                ctx.hold_secs, ctx.max_hold_secs
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TimeStop,
            ));
        }

        None
    }

    fn parameters(&self) -> &StrategyParams {
        &self.generic_params
    }

    fn push_price(&mut self, price: f64, timestamp_ms: i64) {
        self.prices.push_back(crate::signal::PricePoint {
            price,
            timestamp_ms,
        });
        // Keep buffer at 2x mean_lookback to have enough history
        while self.prices.len() > self.params.mean_lookback * 2 {
            self.prices.pop_front();
        }
    }

    fn snapshot(&self) -> MomentumSnapshot {
        let lookback = self.params.mean_lookback;
        let sma = self.compute_sma(lookback);
        let current_price = self.current_price().unwrap_or(0.0);

        let velocity_pct = if let Some(sma) = sma {
            if sma > 0.0 {
                (current_price - sma) / sma * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let direction = if velocity_pct > 0.0 {
            crate::signal::TradeDirection::Long
        } else if velocity_pct < 0.0 {
            crate::signal::TradeDirection::Short
        } else {
            crate::signal::TradeDirection::Neutral
        };

        MomentumSnapshot {
            price_count: self.prices.len(),
            current_price,
            price_velocity_pct: velocity_pct,
            direction,
            strength: 0.0,
            volatility_pct: 0.0,
            pool_data: None,
        }
    }
}

// ---------------------------------------------------------------------------
// TrendFollowerStrategy
// ---------------------------------------------------------------------------

/// Trend Follower strategy.
///
/// Enters on confirmed momentum breakouts with wider stops and trailing exits.
/// Designed for longer holds than scalper strategies — accepts a lower win rate
/// but aims for larger average winners.
///
/// **Entry conditions:**
/// 1. Sufficient price history (>= min_price_count)
/// 2. Price velocity exceeds breakout_threshold_pct (higher than scalper threshold)
/// 3. Velocity sustained for confirmation_ticks consecutive ticks
/// 4. Direction matches the breakout direction
///
/// **Exit conditions (priority order):**
/// 1. Stop-loss (wider than scalper)
/// 2. Take-profit (wider than scalper)
/// 3. Trailing stop (after activation threshold — wider activation & trail)
/// 4. Time stop (much longer than scalper)
/// 5. Trend exhaustion (velocity drops near zero while in position)
///
/// Sensible defaults are used since M1 blueprints did not produce a dedicated
/// trend-following pattern. The parameters are tuned for Flash Trade perp markets
/// with 5-second polling.
pub struct TrendFollowerStrategy {
    params: TrendFollowerParams,
    generic_params: StrategyParams,
    /// Internal momentum detector for price-based signal computation.
    detector: MomentumDetector,
    /// Rolling price buffer for independent velocity calculation.
    prices: VecDeque<crate::signal::PricePoint>,
    /// Number of consecutive ticks with velocity above breakout threshold.
    /// Positive = upward breakout, negative = downward breakout.
    consecutive_breakout: i32,
    /// Previous velocity reading for trend exhaustion detection.
    prev_velocity_pct: f64,
}

impl TrendFollowerStrategy {
    pub fn new(params: TrendFollowerParams) -> Self {
        let generic = params.to_strategy_params();
        let detector = MomentumDetector::new(params.breakout_threshold_pct, params.min_price_count);
        Self {
            generic_params: generic,
            detector,
            prices: VecDeque::with_capacity(params.min_price_count * 2),
            consecutive_breakout: 0,
            prev_velocity_pct: 0.0,
            params,
        }
    }

    /// Compute price velocity over the recent lookback window.
    /// Returns (velocity_pct, is_sufficient_data).
    fn compute_velocity(&self) -> (f64, bool) {
        let lookback = self.params.min_price_count;
        if self.prices.len() < lookback {
            return (0.0, false);
        }

        let recent: Vec<_> = self.prices.iter().rev().take(lookback).collect();
        let current_price = recent[0].price;
        let oldest_price = recent.last().unwrap().price;

        if oldest_price <= 0.0 {
            return (0.0, true);
        }

        let velocity_pct = (current_price - oldest_price) / oldest_price * 100.0;
        (velocity_pct, true)
    }

    /// Get the current (most recent) price.
    #[allow(dead_code)]
    fn current_price(&self) -> Option<f64> {
        self.prices.back().map(|p| p.price)
    }
}

impl Strategy for TrendFollowerStrategy {
    fn name(&self) -> &str {
        "trend-follower"
    }

    fn detect_entry(&mut self, _snapshot: &MomentumSnapshot) -> Signal {
        // Need sufficient price history
        if self.prices.len() < self.params.min_price_count {
            debug!(
                "[trend-follower] Insufficient price history: {}/{} prices",
                self.prices.len(),
                self.params.min_price_count
            );
            return Signal::NoSignal;
        }

        let (velocity_pct, sufficient) = self.compute_velocity();
        if !sufficient {
            return Signal::NoSignal;
        }

        let threshold = self.params.breakout_threshold_pct;

        // Track consecutive breakout ticks
        if velocity_pct > threshold {
            // Upward breakout
            if self.consecutive_breakout > 0 {
                self.consecutive_breakout += 1;
            } else {
                self.consecutive_breakout = 1;
            }
        } else if velocity_pct < -threshold {
            // Downward breakout
            if self.consecutive_breakout < 0 {
                self.consecutive_breakout -= 1;
            } else {
                self.consecutive_breakout = -1;
            }
        } else {
            // Velocity within range — reset consecutive count
            if self.consecutive_breakout != 0 {
                debug!(
                    "[trend-follower] Breakout reset: velocity={:.3}% within threshold ±{:.3}%",
                    velocity_pct, threshold
                );
            }
            self.consecutive_breakout = 0;
        }

        self.prev_velocity_pct = velocity_pct;

        let consec_count = self.consecutive_breakout.abs() as usize;
        debug!(
            "[trend-follower] Velocity: {:.3}%, threshold: ±{:.3}%, consecutive: {} (need {})",
            velocity_pct, threshold, consec_count, self.params.confirmation_ticks
        );

        // Check confirmation — need N consecutive ticks above threshold
        if consec_count < self.params.confirmation_ticks {
            return Signal::NoSignal;
        }

        // ENTRY SIGNAL
        let strength = (velocity_pct.abs() / threshold * 50.0).min(100.0).max(50.0);

        if self.consecutive_breakout > 0 {
            info!(
                "[trend-follower] LONG breakout confirmed: velocity={:.3}% (threshold={:.3}%), consecutive={}, strength={:.1}",
                velocity_pct, threshold, consec_count, strength
            );
            Signal::MomentumLong {
                strength,
                velocity_pct,
            }
        } else {
            info!(
                "[trend-follower] SHORT breakout confirmed: velocity={:.3}% (threshold={:.3}%), consecutive={}, strength={:.1}",
                velocity_pct, threshold, consec_count, strength
            );
            Signal::MomentumShort {
                strength,
                velocity_pct,
            }
        }
    }

    fn detect_exit(&self, _snapshot: &MomentumSnapshot, ctx: &PositionContext) -> Option<Signal> {
        let pnl_pct = if ctx.is_long {
            (ctx.current_price - ctx.entry_price) / ctx.entry_price * 100.0
        } else {
            (ctx.entry_price - ctx.current_price) / ctx.entry_price * 100.0
        };

        let peak_profit_pct = if ctx.entry_price > 0.0 {
            if ctx.is_long {
                (ctx.peak_price - ctx.entry_price) / ctx.entry_price * 100.0
            } else {
                (ctx.entry_price - ctx.peak_price) / ctx.entry_price * 100.0
            }
        } else {
            0.0
        };

        let retracement_pct = if ctx.peak_price > 0.0 && peak_profit_pct > 0.0 {
            if ctx.is_long {
                (ctx.peak_price - ctx.current_price) / ctx.peak_price * 100.0
            } else {
                (ctx.current_price - ctx.peak_price) / ctx.peak_price * 100.0
            }
        } else {
            0.0
        };

        // 1. Stop loss (highest priority)
        if pnl_pct <= -ctx.stop_loss_pct {
            warn!(
                "[trend-follower] STOP LOSS: pnl={:.2}%, threshold=-{:.2}%",
                pnl_pct, ctx.stop_loss_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::StopLoss,
            ));
        }

        // 2. Take profit
        if pnl_pct >= ctx.take_profit_pct {
            info!(
                "[trend-follower] TAKE PROFIT: pnl={:.2}%, threshold={:.2}%",
                pnl_pct, ctx.take_profit_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TakeProfit,
            ));
        }

        // 3. Trailing stop (after activation threshold)
        if peak_profit_pct >= ctx.trailing_activation_pct
            && retracement_pct >= ctx.trailing_stop_pct
        {
            warn!(
                "[trend-follower] TRAILING STOP: retracement={:.2}%, trail={:.2}%, peak_profit={:.2}%",
                retracement_pct, ctx.trailing_stop_pct, peak_profit_pct
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TrailingStop,
            ));
        }

        // 4. Time stop
        if ctx.hold_secs >= ctx.max_hold_secs {
            warn!(
                "[trend-follower] TIME STOP: held {}s, max={}s",
                ctx.hold_secs, ctx.max_hold_secs
            );
            return Some(exit_signal(
                ctx.is_long,
                crate::signal::ExitReason::TimeStop,
            ));
        }

        // 5. Trend exhaustion: velocity has dropped to near zero while in position.
        // Only fire if we've been in the trade long enough to establish a trend (>= 60s)
        // and the trend is clearly fading (velocity near zero or reversing).
        if ctx.hold_secs >= 60 {
            let (velocity_pct, sufficient) = self.compute_velocity();
            if sufficient && velocity_pct.abs() < self.params.breakout_threshold_pct * 0.2 {
                // Trend has faded to less than 20% of breakout threshold
                debug!(
                    "[trend-follower] Trend exhaustion: velocity={:.3}% (< 20% of threshold {:.3}%)",
                    velocity_pct,
                    self.params.breakout_threshold_pct * 0.2
                );
                // Only exit if we've captured some profit or held a long time
                if pnl_pct > 0.0 || ctx.hold_secs > ctx.max_hold_secs / 2 {
                    return Some(exit_signal(
                        ctx.is_long,
                        crate::signal::ExitReason::MomentumLost,
                    ));
                }
            }
        }

        None
    }

    fn parameters(&self) -> &StrategyParams {
        &self.generic_params
    }

    fn push_price(&mut self, price: f64, timestamp_ms: i64) {
        self.detector.push_price(price, timestamp_ms);
        self.prices.push_back(crate::signal::PricePoint {
            price,
            timestamp_ms,
        });
        while self.prices.len() > self.params.min_price_count * 2 {
            self.prices.pop_front();
        }
    }

    fn snapshot(&self) -> MomentumSnapshot {
        self.detector.analyze()
    }
}

// ---------------------------------------------------------------------------
// Strategy Factory
// ---------------------------------------------------------------------------

/// Canonical list of all registered strategy names.
pub fn available_strategies() -> &'static [&'static str] {
    &[
        "momentum-scalper",
        "lp-consumption",
        "mean-reversion",
        "trend-follower",
    ]
}

/// Create a strategy instance by name and parameters.
///
/// This is the single point where strategy names are mapped to concrete types.
/// Both `ScalperEngine` and `PaperEngine` should use this function.
pub fn create_strategy(name: &str, params: StrategyParams) -> anyhow::Result<Box<dyn Strategy>> {
    match name {
        "momentum-scalper" => {
            // Validate params before creating
            if let Err(e) = params.validate() {
                anyhow::bail!("Invalid strategy parameters for '{}': {}", name, e);
            }
            Ok(Box::new(MomentumScalperStrategy::new(params)))
        }
        _ => {
            let available = available_strategies().join(", ");
            anyhow::bail!(
                "Unknown strategy '{}'. Available strategies: {}",
                name,
                available
            );
        }
    }
}

/// Create an LP Consumption strategy from its specific parameters.
pub fn create_lp_consumption_strategy(
    params: LpConsumptionParams,
) -> anyhow::Result<Box<dyn Strategy>> {
    if let Err(e) = params.validate() {
        anyhow::bail!("Invalid LP consumption parameters: {}", e);
    }
    Ok(Box::new(LpConsumptionStrategy::new(params)))
}

/// Create a Mean Reversion strategy from its specific parameters.
pub fn create_mean_reversion_strategy(
    params: MeanReversionParams,
) -> anyhow::Result<Box<dyn Strategy>> {
    if let Err(e) = params.validate() {
        anyhow::bail!("Invalid mean reversion parameters: {}", e);
    }
    Ok(Box::new(MeanReversionStrategy::new(params)))
}

/// Create a Trend Follower strategy from its specific parameters.
pub fn create_trend_follower_strategy(
    params: TrendFollowerParams,
) -> anyhow::Result<Box<dyn Strategy>> {
    if let Err(e) = params.validate() {
        anyhow::bail!("Invalid trend follower parameters: {}", e);
    }
    Ok(Box::new(TrendFollowerStrategy::new(params)))
}

/// Create a strategy by name using config-provided TOML sub-table.
///
/// This is the primary factory function used by engines. It handles both
/// momentum-scalper (via StrategyParams) and lp-consumption (via LpConsumptionParams).
pub fn create_strategy_from_config(
    name: &str,
    sub_table: Option<&toml::Value>,
    fallback_params: StrategyParams,
) -> anyhow::Result<Box<dyn Strategy>> {
    match name {
        "momentum-scalper" => {
            if let Err(e) = fallback_params.validate() {
                anyhow::bail!("Invalid strategy parameters for '{}': {}", name, e);
            }
            Ok(Box::new(MomentumScalperStrategy::new(fallback_params)))
        }
        "lp-consumption" => {
            let lp_params = if let Some(table) = sub_table {
                let params: LpConsumptionParams = table.clone().try_into().map_err(|e| {
                    anyhow::anyhow!("Failed to parse [strategy.lp-consumption] sub-table: {}", e)
                })?;
                params
            } else {
                // Use defaults
                LpConsumptionParams {
                    consumption_velocity_threshold: 0.5,
                    lp_concentration_min: 0.7,
                    confirmation_ticks: 3,
                    max_utilization: 0.9,
                    direction_bias: "neutral".to_string(),
                    clip_size_usd: fallback_params.clip_size_usd,
                    max_hold_secs: 3600,
                    take_profit_pct: 2.0,
                    stop_loss_pct: 1.0,
                    trailing_stop_pct: 0.8,
                    trailing_activation_pct: 1.2,
                    cooldown_after_loss_secs: 300,
                    use_native_tp_sl: true,
                    leverage: 2.5,
                    scale_in_clips: 1,
                }
            };
            create_lp_consumption_strategy(lp_params)
        }
        "mean-reversion" => {
            let mr_params = if let Some(table) = sub_table {
                let params: MeanReversionParams = table.clone().try_into().map_err(|e| {
                    anyhow::anyhow!("Failed to parse [strategy.mean-reversion] sub-table: {}", e)
                })?;
                params
            } else {
                // Use defaults
                MeanReversionParams {
                    mean_lookback: 120,
                    deviation_threshold_pct: 1.5,
                    reversal_confirmation_ticks: 2,
                    mean_tolerance_pct: 0.3,
                    direction_bias: "neutral".to_string(),
                    clip_size_usd: fallback_params.clip_size_usd,
                    max_hold_secs: 1800,
                    take_profit_pct: 1.0,
                    stop_loss_pct: 1.5,
                    trailing_stop_pct: 0.0,
                    trailing_activation_pct: 0.0,
                    cooldown_after_loss_secs: 300,
                    use_native_tp_sl: true,
                    leverage: 3.0,
                    scale_in_clips: 1,
                }
            };
            create_mean_reversion_strategy(mr_params)
        }
        "trend-follower" => {
            let tf_params = if let Some(table) = sub_table {
                let params: TrendFollowerParams = table.clone().try_into().map_err(|e| {
                    anyhow::anyhow!("Failed to parse [strategy.trend-follower] sub-table: {}", e)
                })?;
                params
            } else {
                // Use sensible defaults (no M1 blueprint available)
                TrendFollowerParams {
                    breakout_threshold_pct: 0.25,
                    confirmation_ticks: 4,
                    min_price_count: 30,
                    direction_bias: "neutral".to_string(),
                    clip_size_usd: fallback_params.clip_size_usd,
                    max_hold_secs: 7200,
                    take_profit_pct: 5.0,
                    stop_loss_pct: 2.0,
                    trailing_stop_pct: 1.5,
                    trailing_activation_pct: 2.5,
                    cooldown_after_loss_secs: 300,
                    use_native_tp_sl: true,
                    leverage: 5.0,
                    scale_in_clips: 1,
                }
            };
            create_trend_follower_strategy(tf_params)
        }
        _ => {
            let available = available_strategies().join(", ");
            anyhow::bail!(
                "Unknown strategy '{}'. Available strategies: {}",
                name,
                available
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{ExitReason, MomentumSnapshot, PoolSnapshot, Signal, TradeDirection};

    // ===== Momentum Scalper Tests =====

    /// Helper: create default strategy params for testing.
    fn default_params() -> StrategyParams {
        StrategyParams {
            direction_bias: "neutral".to_string(),
            momentum_threshold_pct: 0.15,
            lookback_count: 60,
            scale_in_clips: 1,
            clip_size_usd: 100.0,
            max_hold_secs: 1800,
            take_profit_pct: 2.5,
            stop_loss_pct: 1.0,
            trailing_stop_pct: 0.8,
            trailing_activation_pct: 1.5,
            cooldown_after_loss_secs: 300,
            use_native_tp_sl: true,
        }
    }

    /// Helper: build a price series that should produce a LONG signal.
    fn feed_rising_prices(strategy: &mut dyn Strategy, start: f64, count: usize) {
        let base_ts = 1000000_i64;
        for i in 0..count {
            let price = start * (1.0 + 0.005 * (i as f64));
            strategy.push_price(price, base_ts + (i as i64) * 1000);
        }
    }

    /// Helper: build a price series that should produce a SHORT signal.
    fn feed_falling_prices(strategy: &mut dyn Strategy, start: f64, count: usize) {
        let base_ts = 1000000_i64;
        for i in 0..count {
            let price = start * (1.0 - 0.005 * (i as f64));
            strategy.push_price(price, base_ts + (i as i64) * 1000);
        }
    }

    /// Helper: build a position context for exit tests.
    fn default_exit_context(
        is_long: bool,
        entry_price: f64,
        current_price: f64,
        peak_price: f64,
        hold_secs: u64,
    ) -> PositionContext {
        PositionContext {
            is_long,
            entry_price,
            current_price,
            peak_price,
            hold_secs,
            max_hold_secs: 1800,
            take_profit_pct: 2.5,
            stop_loss_pct: 1.0,
            trailing_stop_pct: 0.8,
            trailing_activation_pct: 1.5,
        }
    }

    #[test]
    fn test_long_signal() {
        let params = default_params();
        let mut strategy = MomentumScalperStrategy::new(params);
        feed_rising_prices(&mut strategy, 100.0, 10);

        let snapshot = strategy.snapshot();
        let signal = strategy.detect_entry(&snapshot);

        match signal {
            Signal::MomentumLong {
                strength,
                velocity_pct,
            } => {
                assert!(velocity_pct >= 0.15);
                assert!(strength >= 50.0);
            }
            other => panic!("Expected MomentumLong, got {:?}", other),
        }
    }

    #[test]
    fn test_short_signal() {
        let params = default_params();
        let mut strategy = MomentumScalperStrategy::new(params);
        feed_falling_prices(&mut strategy, 100.0, 10);

        let snapshot = strategy.snapshot();
        let signal = strategy.detect_entry(&snapshot);

        match signal {
            Signal::MomentumShort {
                strength,
                velocity_pct,
            } => {
                assert!(velocity_pct >= 0.15);
                assert!(strength >= 50.0);
            }
            other => panic!("Expected MomentumShort, got {:?}", other),
        }
    }

    #[test]
    fn test_no_signal_insufficient_prices() {
        let params = default_params();
        let mut strategy = MomentumScalperStrategy::new(params);
        strategy.push_price(100.0, 1000);
        strategy.push_price(101.0, 2000);
        strategy.push_price(102.0, 3000);

        let snapshot = strategy.snapshot();
        let signal = strategy.detect_entry(&snapshot);
        assert_eq!(signal, Signal::NoSignal);
    }

    #[test]
    fn test_no_signal_flat_prices() {
        let params = default_params();
        let mut strategy = MomentumScalperStrategy::new(params);
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        let snapshot = strategy.snapshot();
        let signal = strategy.detect_entry(&snapshot);
        assert_eq!(signal, Signal::NoSignal);
    }

    #[test]
    fn test_stop_loss_fires_before_take_profit() {
        let params = default_params();
        let strategy = MomentumScalperStrategy::new(params);
        let ctx = default_exit_context(true, 100.0, 98.0, 100.0, 10);

        let mut detector_feed = MomentumDetector::new(0.15, 60);
        for i in 0..10 {
            detector_feed.push_price(98.0, 1000 + (i as i64) * 1000);
        }
        let snapshot = detector_feed.analyze();

        let result = strategy.detect_exit(&snapshot, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::StopLoss);
            }
            other => panic!("Expected ExitLong(StopLoss), got {:?}", other),
        }
    }

    #[test]
    fn test_take_profit_exit() {
        let params = default_params();
        let strategy = MomentumScalperStrategy::new(params);
        let ctx = default_exit_context(true, 100.0, 103.0, 103.0, 10);

        let mut detector_feed = MomentumDetector::new(0.15, 60);
        for i in 0..10 {
            detector_feed.push_price(103.0, 1000 + (i as i64) * 1000);
        }
        let snapshot = detector_feed.analyze();

        let result = strategy.detect_exit(&snapshot, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TakeProfit);
            }
            other => panic!("Expected ExitLong(TakeProfit), got {:?}", other),
        }
    }

    #[test]
    fn test_time_stop_exit() {
        let params = default_params();
        let strategy = MomentumScalperStrategy::new(params);
        let ctx = default_exit_context(true, 100.0, 100.5, 100.5, 2000);

        let mut detector_feed = MomentumDetector::new(0.15, 60);
        for i in 0..10 {
            detector_feed.push_price(100.5, 1000 + (i as i64) * 1000);
        }
        let snapshot = detector_feed.analyze();

        let result = strategy.detect_exit(&snapshot, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TimeStop);
            }
            other => panic!("Expected ExitLong(TimeStop), got {:?}", other),
        }
    }

    #[test]
    fn test_no_exit_when_position_stable() {
        let params = default_params();
        let strategy = MomentumScalperStrategy::new(params);
        let ctx = default_exit_context(true, 100.0, 100.3, 100.3, 30);

        let mut detector_feed = MomentumDetector::new(0.15, 60);
        for i in 0..10 {
            detector_feed.push_price(100.3, 1000 + (i as i64) * 1000);
        }
        let snapshot = detector_feed.analyze();

        let result = strategy.detect_exit(&snapshot, &ctx);
        assert!(
            result.is_none()
                || matches!(
                    result,
                    Some(Signal::ExitLong {
                        reason: ExitReason::MomentumLost
                    })
                ),
            "Expected no exit or momentum-lost for stable position, got {:?}",
            result
        );
    }

    // ---- Factory Tests ----

    #[test]
    fn test_factory_creates_momentum_scalper() {
        let params = default_params();
        let strategy = create_strategy("momentum-scalper", params).unwrap();
        assert_eq!(strategy.name(), "momentum-scalper");
    }

    #[test]
    fn test_factory_rejects_unknown_strategy() {
        let params = default_params();
        let result = create_strategy("nonexistent", params);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("nonexistent"));
        assert!(err.contains("momentum-scalper"));
    }

    #[test]
    fn test_params_validation_rejects_zero_threshold() {
        let mut params = default_params();
        params.momentum_threshold_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_params_validation_rejects_zero_lookback() {
        let mut params = default_params();
        params.lookback_count = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_params_validation_rejects_zero_clip_size() {
        let mut params = default_params();
        params.clip_size_usd = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_params_validation_rejects_zero_tp() {
        let mut params = default_params();
        params.take_profit_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_params_validation_rejects_zero_sl() {
        let mut params = default_params();
        params.stop_loss_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_params_validation_accepts_valid() {
        let params = default_params();
        assert!(params.validate().is_ok());
    }

    // ===== LP Consumption Strategy Tests =====

    /// Helper: create default LP consumption params for testing.
    fn default_lp_params() -> LpConsumptionParams {
        LpConsumptionParams {
            consumption_velocity_threshold: 0.5,
            lp_concentration_min: 0.7,
            confirmation_ticks: 3,
            max_utilization: 0.9,
            direction_bias: "neutral".to_string(),
            clip_size_usd: 100.0,
            max_hold_secs: 3600,
            take_profit_pct: 2.0,
            stop_loss_pct: 1.0,
            trailing_stop_pct: 0.8,
            trailing_activation_pct: 1.2,
            cooldown_after_loss_secs: 300,
            use_native_tp_sl: true,
            leverage: 2.5,
            scale_in_clips: 1,
        }
    }

    /// Helper: build a snapshot with pool data simulating LP consumption.
    fn snapshot_with_pool(
        price: f64,
        long_util_velocity: f64,
        short_util_velocity: f64,
        long_utilization: f64,
        short_utilization: f64,
    ) -> MomentumSnapshot {
        MomentumSnapshot {
            price_count: 10,
            current_price: price,
            price_velocity_pct: 0.0,
            direction: TradeDirection::Neutral,
            strength: 0.0,
            volatility_pct: 0.0,
            pool_data: Some(PoolSnapshot {
                aum_usd: 1_000_000.0,
                long_utilization,
                short_utilization,
                long_utilization_velocity: long_util_velocity,
                short_utilization_velocity: short_util_velocity,
            }),
        }
    }

    /// Helper: build a snapshot with no pool data.
    fn snapshot_no_pool(price: f64) -> MomentumSnapshot {
        MomentumSnapshot {
            price_count: 10,
            current_price: price,
            price_velocity_pct: 0.0,
            direction: TradeDirection::Neutral,
            strength: 0.0,
            volatility_pct: 0.0,
            pool_data: None,
        }
    }

    /// Helper: LP consumption exit context.
    fn lp_exit_context(
        is_long: bool,
        entry_price: f64,
        current_price: f64,
        peak_price: f64,
        hold_secs: u64,
    ) -> PositionContext {
        PositionContext {
            is_long,
            entry_price,
            current_price,
            peak_price,
            hold_secs,
            max_hold_secs: 3600,
            take_profit_pct: 2.0,
            stop_loss_pct: 1.0,
            trailing_stop_pct: 0.8,
            trailing_activation_pct: 1.2,
        }
    }

    #[test]
    fn test_lp_consumption_entry_long_when_consumption_exceeds_threshold() {
        let params = default_lp_params();
        let mut strategy = LpConsumptionStrategy::new(params);

        // Simulate 3 consecutive ticks with strong long-side LP consumption
        // Long velocity = 0.8 (above threshold 0.5), concentration = 80% (above 0.7)
        for _ in 0..3 {
            let snap = snapshot_with_pool(100.0, 0.8, 0.2, 0.3, 0.2);
            let _signal = strategy.detect_entry(&snap);
            // First two ticks should be NoSignal (need confirmation_ticks=3)
            // Third tick should produce a LONG signal
        }

        let snap = snapshot_with_pool(100.0, 0.8, 0.2, 0.3, 0.2);
        let signal = strategy.detect_entry(&snap);

        match signal {
            Signal::MomentumLong { strength, .. } => {
                assert!(
                    strength >= 50.0,
                    "strength should be >= 50, got {}",
                    strength
                );
            }
            other => panic!("Expected MomentumLong, got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_entry_short_when_ask_side_consumed() {
        let params = default_lp_params();
        let mut strategy = LpConsumptionStrategy::new(params);

        // Short-side consumption: velocity = 0.9 (above threshold), concentration = 85%
        for _ in 0..3 {
            let snap = snapshot_with_pool(100.0, 0.15, 0.9, 0.2, 0.3);
            let _ = strategy.detect_entry(&snap);
        }

        let snap = snapshot_with_pool(100.0, 0.15, 0.9, 0.2, 0.3);
        let signal = strategy.detect_entry(&snap);

        match signal {
            Signal::MomentumShort { strength, .. } => {
                assert!(strength >= 50.0);
            }
            other => panic!("Expected MomentumShort, got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_no_signal_when_velocity_below_threshold() {
        let params = default_lp_params();
        let mut strategy = LpConsumptionStrategy::new(params);

        // Velocity = 0.3, below threshold of 0.5
        let snap = snapshot_with_pool(100.0, 0.3, 0.1, 0.3, 0.2);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not signal when velocity is below threshold"
        );
    }

    #[test]
    fn test_lp_consumption_no_signal_when_concentration_too_low() {
        let params = default_lp_params();
        let mut strategy = LpConsumptionStrategy::new(params);

        // Velocity above threshold (0.8) but concentration is balanced (55%/45%)
        let snap = snapshot_with_pool(100.0, 0.55, 0.45, 0.3, 0.2);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not signal when directional concentration is below minimum"
        );
    }

    #[test]
    fn test_lp_consumption_no_signal_when_pool_data_missing() {
        let params = default_lp_params();
        let mut strategy = LpConsumptionStrategy::new(params);

        let snap = snapshot_no_pool(100.0);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should return NoSignal when pool data is missing"
        );
    }

    #[test]
    fn test_lp_consumption_no_signal_when_utilization_maxed() {
        let params = default_lp_params();
        let mut strategy = LpConsumptionStrategy::new(params);

        // Velocity and concentration good, but utilization at 92% (above max_utilization=0.9)
        let snap = snapshot_with_pool(100.0, 0.8, 0.1, 0.92, 0.3);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not enter when pool utilization is above max"
        );
    }

    #[test]
    fn test_lp_consumption_exit_on_stall() {
        let params = default_lp_params();
        let strategy = LpConsumptionStrategy::new(params);

        // Position entered when long velocity was 0.8, now velocity dropped to 0.05.
        // Entry at 100, current at 100.3 (0.3% gain, below TP of 2.0%, not enough for SL).
        // Held for 200s (> 300 threshold in code, but close).
        let snap = snapshot_with_pool(100.3, 0.05, 0.02, 0.3, 0.2);
        let ctx = lp_exit_context(true, 100.0, 100.3, 100.5, 400);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(
                    reason,
                    ExitReason::MomentumLost,
                    "Should exit on consumption stall"
                );
            }
            other => panic!("Expected ExitLong(MomentumLost), got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_exit_on_stop_loss() {
        let params = default_lp_params();
        let strategy = LpConsumptionStrategy::new(params);

        // Price dropped 1.5% (SL is 1.0%)
        let snap = snapshot_with_pool(98.5, 0.1, 0.05, 0.3, 0.2);
        let ctx = lp_exit_context(true, 100.0, 98.5, 100.0, 30);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::StopLoss);
            }
            other => panic!("Expected ExitLong(StopLoss), got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_exit_on_take_profit() {
        let params = default_lp_params();
        let strategy = LpConsumptionStrategy::new(params);

        // Price up 2.5% (TP is 2.0%)
        let snap = snapshot_with_pool(102.5, 0.3, 0.1, 0.3, 0.2);
        let ctx = lp_exit_context(true, 100.0, 102.5, 102.5, 60);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TakeProfit);
            }
            other => panic!("Expected ExitLong(TakeProfit), got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_exit_on_time_stop() {
        let params = default_lp_params();
        let strategy = LpConsumptionStrategy::new(params);

        // Held for 4000s, max is 3600s
        let snap = snapshot_with_pool(100.3, 0.1, 0.05, 0.3, 0.2);
        let ctx = lp_exit_context(true, 100.0, 100.3, 100.5, 4000);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TimeStop);
            }
            other => panic!("Expected ExitLong(TimeStop), got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_exit_on_reversal() {
        let params = default_lp_params();
        let strategy = LpConsumptionStrategy::new(params);

        // We're long, but short-side consumption is now accelerating.
        // Long velocity=0.2 (> stall threshold 0.15) so stall doesn't fire.
        // Short velocity=0.9 with concentration 0.9/1.1=0.818 > 0.7 min → reversal.
        let snap = snapshot_with_pool(100.3, 0.2, 0.9, 0.3, 0.35);
        let ctx = lp_exit_context(true, 100.0, 100.3, 100.5, 60);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(
                    reason,
                    ExitReason::ReversalDetected,
                    "Should detect reversal when opposite-side consumption accelerates"
                );
            }
            other => panic!("Expected ExitLong(ReversalDetected), got {:?}", other),
        }
    }

    #[test]
    fn test_lp_consumption_no_exit_when_position_stable() {
        let params = default_lp_params();
        let strategy = LpConsumptionStrategy::new(params);

        // Stable position, velocity still good
        let snap = snapshot_with_pool(100.3, 0.6, 0.1, 0.3, 0.2);
        let ctx = lp_exit_context(true, 100.0, 100.3, 100.5, 30);

        let result = strategy.detect_exit(&snap, &ctx);
        assert!(
            result.is_none(),
            "Expected no exit for stable LP consumption, got {:?}",
            result
        );
    }

    #[test]
    fn test_lp_consumption_factory_creates_strategy() {
        let params = default_lp_params();
        let strategy = create_lp_consumption_strategy(params).unwrap();
        assert_eq!(strategy.name(), "lp-consumption");
    }

    #[test]
    fn test_lp_consumption_factory_from_config() {
        let fallback = default_params();
        let strategy = create_strategy_from_config(
            "lp-consumption",
            None, // No sub-table, use defaults
            fallback,
        )
        .unwrap();
        assert_eq!(strategy.name(), "lp-consumption");
    }

    #[test]
    fn test_lp_consumption_factory_from_config_with_table() {
        let fallback = default_params();
        let toml_str = r#"
            consumption_velocity_threshold = 0.3
            lp_concentration_min = 0.6
            confirmation_ticks = 2
            max_utilization = 0.85
            clip_size_usd = 50.0
            take_profit_pct = 1.5
            stop_loss_pct = 0.5
            trailing_stop_pct = 0.6
            trailing_activation_pct = 1.0
            max_hold_secs = 2400
            cooldown_after_loss_secs = 120
            direction_bias = "long"
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let strategy =
            create_strategy_from_config("lp-consumption", Some(&value), fallback).unwrap();
        assert_eq!(strategy.name(), "lp-consumption");
    }

    #[test]
    fn test_lp_consumption_params_validation_rejects_zero_velocity() {
        let mut params = default_lp_params();
        params.consumption_velocity_threshold = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_lp_consumption_params_validation_rejects_invalid_concentration() {
        let mut params = default_lp_params();
        params.lp_concentration_min = 1.5;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_lp_consumption_params_validation_rejects_zero_confirmation() {
        let mut params = default_lp_params();
        params.confirmation_ticks = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_lp_consumption_params_validation_accepts_valid() {
        let params = default_lp_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_lp_consumption_directional_concentration_pure_long() {
        let pool = PoolSnapshot {
            aum_usd: 1_000_000.0,
            long_utilization: 0.3,
            short_utilization: 0.1,
            long_utilization_velocity: 1.0,
            short_utilization_velocity: 0.0,
        };
        let (long_c, short_c) = LpConsumptionStrategy::directional_concentration(&pool);
        assert!(
            (long_c - 1.0).abs() < 0.01,
            "long_conc should be ~1.0, got {}",
            long_c
        );
        assert!(
            (short_c - 0.0).abs() < 0.01,
            "short_conc should be ~0.0, got {}",
            short_c
        );
    }

    #[test]
    fn test_lp_consumption_directional_concentration_balanced() {
        let pool = PoolSnapshot {
            aum_usd: 1_000_000.0,
            long_utilization: 0.3,
            short_utilization: 0.3,
            long_utilization_velocity: 0.5,
            short_utilization_velocity: 0.5,
        };
        let (long_c, short_c) = LpConsumptionStrategy::directional_concentration(&pool);
        assert!(
            (long_c - 0.5).abs() < 0.01,
            "long_conc should be ~0.5, got {}",
            long_c
        );
        assert!(
            (short_c - 0.5).abs() < 0.01,
            "short_conc should be ~0.5, got {}",
            short_c
        );
    }

    #[test]
    fn test_lp_consumption_directional_concentration_zero_velocity() {
        let pool = PoolSnapshot {
            aum_usd: 1_000_000.0,
            long_utilization: 0.3,
            short_utilization: 0.3,
            long_utilization_velocity: 0.0,
            short_utilization_velocity: 0.0,
        };
        let (long_c, short_c) = LpConsumptionStrategy::directional_concentration(&pool);
        // When both are zero, should return 0.5/0.5 (neutral)
        assert!((long_c - 0.5).abs() < 0.01);
        assert!((short_c - 0.5).abs() < 0.01);
    }

    // ===== Mean Reversion Strategy Tests =====

    /// Helper: create default mean reversion params for testing.
    fn default_mr_params() -> MeanReversionParams {
        MeanReversionParams {
            mean_lookback: 20, // Small for testing
            deviation_threshold_pct: 1.5,
            reversal_confirmation_ticks: 2,
            mean_tolerance_pct: 0.3,
            direction_bias: "neutral".to_string(),
            clip_size_usd: 100.0,
            max_hold_secs: 1800,
            take_profit_pct: 1.0,
            stop_loss_pct: 1.5,
            trailing_stop_pct: 0.0,
            trailing_activation_pct: 0.0,
            cooldown_after_loss_secs: 300,
            use_native_tp_sl: true,
            leverage: 3.0,
            scale_in_clips: 1,
        }
    }

    /// Helper: build a snapshot for mean reversion (no pool data needed).
    fn mr_snapshot(price: f64) -> MomentumSnapshot {
        MomentumSnapshot {
            price_count: 20,
            current_price: price,
            price_velocity_pct: 0.0,
            direction: TradeDirection::Neutral,
            strength: 0.0,
            volatility_pct: 0.0,
            pool_data: None,
        }
    }

    /// Helper: mean reversion exit context.
    fn mr_exit_context(
        is_long: bool,
        entry_price: f64,
        current_price: f64,
        peak_price: f64,
        hold_secs: u64,
    ) -> PositionContext {
        PositionContext {
            is_long,
            entry_price,
            current_price,
            peak_price,
            hold_secs,
            max_hold_secs: 1800,
            take_profit_pct: 1.0,
            stop_loss_pct: 1.5,
            trailing_stop_pct: 0.0,
            trailing_activation_pct: 0.0,
        }
    }

    #[test]
    fn test_mean_reversion_entry_long_after_downward_spike_and_reversal() {
        // Scenario: prices trade around 100, then spike DOWN to 96 (4% below SMA),
        // then start reversing upward → should generate LONG signal.
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        // Feed 20 stable prices around 100 to establish SMA
        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Spike down: 4% below SMA (100), well above 1.5% threshold
        // Spike tick: push price to 96
        strategy.push_price(96.0, 30_000);
        let snap = mr_snapshot(96.0);
        let signal = strategy.detect_entry(&snap);
        // Spike detected but no reversal yet
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not signal on spike without reversal"
        );

        // First reversal tick: price moves back up toward mean
        strategy.push_price(96.5, 31_000);
        let snap = mr_snapshot(96.5);
        let signal = strategy.detect_entry(&snap);
        // Only 1 reversal tick, need 2
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not signal with only 1 reversal tick"
        );

        // Second reversal tick: price continues up toward mean
        strategy.push_price(97.0, 32_000);
        let snap = mr_snapshot(97.0);
        let signal = strategy.detect_entry(&snap);
        // Should generate LONG signal (fade the downward spike)
        match signal {
            Signal::MomentumLong { strength, .. } => {
                assert!(
                    strength >= 50.0,
                    "strength should be >= 50, got {}",
                    strength
                );
            }
            other => panic!(
                "Expected MomentumLong after downward spike + reversal, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_mean_reversion_entry_short_after_upward_spike_and_reversal() {
        // Scenario: prices trade around 100, then spike UP to 104 (4% above SMA),
        // then start reversing downward → should generate SHORT signal.
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        // Feed 20 stable prices around 100 to establish SMA
        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Spike up: 4% above SMA
        strategy.push_price(104.0, 30_000);
        let snap = mr_snapshot(104.0);
        let _ = strategy.detect_entry(&snap);

        // First reversal tick: price moves down
        strategy.push_price(103.5, 31_000);
        let snap = mr_snapshot(103.5);
        let _ = strategy.detect_entry(&snap);

        // Second reversal tick: price continues down
        strategy.push_price(103.0, 32_000);
        let snap = mr_snapshot(103.0);
        let signal = strategy.detect_entry(&snap);

        match signal {
            Signal::MomentumShort { strength, .. } => {
                assert!(strength >= 50.0);
            }
            other => panic!(
                "Expected MomentumShort after upward spike + reversal, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_mean_reversion_no_entry_on_gradual_move_without_spike() {
        // Gradual drift without exceeding deviation threshold → no signal
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        // Feed 20 stable prices
        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Gradual upward drift: 0.1% per tick for 10 ticks = 1% total (below 1.5% threshold)
        for i in 0..10 {
            let price = 100.0 * (1.0 + 0.001 * (i as f64));
            strategy.push_price(price, 20_000 + (i as i64) * 1000);
            let snap = mr_snapshot(price);
            let signal = strategy.detect_entry(&snap);
            assert_eq!(
                signal,
                Signal::NoSignal,
                "Should not signal on gradual move without spike (tick {})",
                i
            );
        }
    }

    #[test]
    fn test_mean_reversion_exit_on_mean_return() {
        // Position is LONG, entered at 96 (below SMA of 100).
        // Price returns to 100 (within tolerance of SMA) → exit.
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        // Feed 20 stable prices at 100 to establish SMA
        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        // Current price is at 100 (exactly at mean)
        strategy.push_price(100.0, 21_000);

        let snap = mr_snapshot(100.0);
        let ctx = mr_exit_context(true, 96.0, 100.0, 100.0, 60);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(
                    reason,
                    ExitReason::TakeProfit,
                    "Mean return should use TakeProfit exit reason"
                );
            }
            other => panic!(
                "Expected ExitLong(TakeProfit) on mean return, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_mean_reversion_exit_on_stop_loss() {
        // Price drops 2% (SL is 1.5%)
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        // Need prices for SMA computation
        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        strategy.push_price(98.0, 21_000);

        let snap = mr_snapshot(98.0);
        let ctx = mr_exit_context(true, 100.0, 98.0, 100.5, 30);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::StopLoss);
            }
            other => panic!("Expected ExitLong(StopLoss), got {:?}", other),
        }
    }

    #[test]
    fn test_mean_reversion_exit_on_time_stop() {
        // Held for 2000s, max is 1800s. Price hasn't moved enough for TP or SL.
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        // Push a price far from mean so mean-return exit doesn't fire
        strategy.push_price(96.3, 21_000);

        let snap = mr_snapshot(96.3);
        // Entry at 96.0, current at 96.3 → 0.31% gain, below 1.0% TP
        let ctx = mr_exit_context(true, 96.0, 96.3, 96.5, 2000);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TimeStop);
            }
            other => panic!("Expected ExitLong(TimeStop), got {:?}", other),
        }
    }

    #[test]
    fn test_mean_reversion_no_signal_insufficient_history() {
        // With fewer prices than mean_lookback, should return NoSignal
        let params = default_mr_params(); // mean_lookback = 20
        let mut strategy = MeanReversionStrategy::new(params);

        // Only 10 prices (less than mean_lookback of 20)
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        let snap = mr_snapshot(100.0);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should return NoSignal when price history is insufficient"
        );

        // With 0 prices
        let mut strategy2 = MeanReversionStrategy::new(default_mr_params());
        let snap2 = mr_snapshot(100.0);
        let signal2 = strategy2.detect_entry(&snap2);
        assert_eq!(
            signal2,
            Signal::NoSignal,
            "Should return NoSignal with zero prices"
        );
    }

    #[test]
    fn test_mean_reversion_no_exit_when_price_still_far_from_mean() {
        // Price is still far from mean, no SL/TP hit, no time stop
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        for i in 0..20 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        // Push price far from mean, still within SL/TP range from entry
        strategy.push_price(96.3, 21_000);

        let snap = mr_snapshot(96.3);
        // Entry at 96.0, current at 96.3 → 0.31% gain (below 1.0% TP, above -1.5% SL)
        // Deviation from SMA ≈ 3.5% (above 0.3% tolerance)
        let ctx = mr_exit_context(true, 96.0, 96.3, 96.5, 60);

        let result = strategy.detect_exit(&snap, &ctx);
        assert!(
            result.is_none(),
            "Expected no exit when price is still far from mean, got {:?}",
            result
        );
    }

    #[test]
    fn test_mean_reversion_factory_creates_strategy() {
        let params = default_mr_params();
        let strategy = create_mean_reversion_strategy(params).unwrap();
        assert_eq!(strategy.name(), "mean-reversion");
    }

    #[test]
    fn test_mean_reversion_factory_from_config() {
        let fallback = default_params();
        let strategy = create_strategy_from_config(
            "mean-reversion",
            None, // No sub-table, use defaults
            fallback,
        )
        .unwrap();
        assert_eq!(strategy.name(), "mean-reversion");
    }

    #[test]
    fn test_mean_reversion_factory_from_config_with_table() {
        let fallback = default_params();
        let toml_str = r#"
            mean_lookback = 60
            deviation_threshold_pct = 2.0
            reversal_confirmation_ticks = 3
            mean_tolerance_pct = 0.5
            clip_size_usd = 50.0
            take_profit_pct = 1.5
            stop_loss_pct = 2.0
            trailing_stop_pct = 0.0
            trailing_activation_pct = 0.0
            max_hold_secs = 2400
            cooldown_after_loss_secs = 120
            direction_bias = "long"
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let strategy =
            create_strategy_from_config("mean-reversion", Some(&value), fallback).unwrap();
        assert_eq!(strategy.name(), "mean-reversion");
    }

    #[test]
    fn test_mean_reversion_params_validation_rejects_zero_lookback() {
        let mut params = default_mr_params();
        params.mean_lookback = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_mean_reversion_params_validation_rejects_zero_deviation() {
        let mut params = default_mr_params();
        params.deviation_threshold_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_mean_reversion_params_validation_rejects_zero_reversal_ticks() {
        let mut params = default_mr_params();
        params.reversal_confirmation_ticks = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_mean_reversion_params_validation_accepts_valid() {
        let params = default_mr_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_mean_reversion_sma_computation() {
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        // Push 20 prices: 100, 101, 102, ..., 119
        for i in 0..20 {
            strategy.push_price(100.0 + i as f64, 1000 + (i as i64) * 1000);
        }

        let sma = strategy.compute_sma(20).unwrap();
        // Average of 100..119 = (100+119)/2 = 109.5
        assert!(
            (sma - 109.5).abs() < 0.01,
            "SMA should be 109.5, got {:.2}",
            sma
        );
    }

    #[test]
    fn test_mean_reversion_sma_insufficient_data() {
        let params = default_mr_params();
        let mut strategy = MeanReversionStrategy::new(params);

        strategy.push_price(100.0, 1000);
        strategy.push_price(101.0, 2000);

        let sma = strategy.compute_sma(20);
        assert!(sma.is_none(), "SMA should be None with insufficient data");
    }

    // ===== Trend Follower Strategy Tests =====

    /// Helper: create default trend follower params for testing.
    fn default_tf_params() -> TrendFollowerParams {
        TrendFollowerParams {
            breakout_threshold_pct: 0.25,
            confirmation_ticks: 3, // Small for testing
            min_price_count: 10,   // Small for testing
            direction_bias: "neutral".to_string(),
            clip_size_usd: 100.0,
            max_hold_secs: 7200,
            take_profit_pct: 5.0,
            stop_loss_pct: 2.0,
            trailing_stop_pct: 1.5,
            trailing_activation_pct: 2.5,
            cooldown_after_loss_secs: 300,
            use_native_tp_sl: true,
            leverage: 5.0,
            scale_in_clips: 1,
        }
    }

    /// Helper: build a snapshot for trend follower (no pool data needed).
    fn tf_snapshot(price: f64) -> MomentumSnapshot {
        MomentumSnapshot {
            price_count: 30,
            current_price: price,
            price_velocity_pct: 0.0,
            direction: TradeDirection::Neutral,
            strength: 0.0,
            volatility_pct: 0.0,
            pool_data: None,
        }
    }

    /// Helper: trend follower exit context.
    fn tf_exit_context(
        is_long: bool,
        entry_price: f64,
        current_price: f64,
        peak_price: f64,
        hold_secs: u64,
    ) -> PositionContext {
        PositionContext {
            is_long,
            entry_price,
            current_price,
            peak_price,
            hold_secs,
            max_hold_secs: 7200,
            take_profit_pct: 5.0,
            stop_loss_pct: 2.0,
            trailing_stop_pct: 1.5,
            trailing_activation_pct: 2.5,
        }
    }

    #[test]
    fn test_trend_follower_entry_long_on_breakout() {
        // Feed rising prices that produce a sustained upward breakout.
        // Start stable, then rapid rise to exceed breakout_threshold_pct = 0.25%.
        let params = default_tf_params(); // confirmation_ticks = 3, min_price_count = 10
        let mut strategy = TrendFollowerStrategy::new(params);

        // Feed 10 stable prices around 100 (to meet min_price_count)
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Now feed strongly rising prices to trigger breakout.
        // Each tick adds +0.3% from the initial price, building a strong trend.
        // Need to call detect_entry on each tick to accumulate consecutive_breakout count.
        for i in 0..5 {
            let price = 100.0 * (1.0 + 0.003 * ((i + 1) as f64));
            strategy.push_price(price, 20_000 + (i as i64) * 1000);
            let snap = tf_snapshot(price);
            let _signal = strategy.detect_entry(&snap);
        }

        // After 5 ticks of upward breakout (need 3 consecutive), should produce LONG
        let last_price = 100.0 * (1.0 + 0.003 * 6.0);
        strategy.push_price(last_price, 25_000);
        let snap = tf_snapshot(last_price);
        let signal = strategy.detect_entry(&snap);

        match signal {
            Signal::MomentumLong {
                strength,
                velocity_pct,
            } => {
                assert!(
                    strength >= 50.0,
                    "strength should be >= 50, got {}",
                    strength
                );
                assert!(
                    velocity_pct > 0.25,
                    "velocity should exceed breakout threshold"
                );
            }
            other => panic!("Expected MomentumLong on upward breakout, got {:?}", other),
        }
    }

    #[test]
    fn test_trend_follower_entry_short_on_breakout() {
        // Feed falling prices that produce a sustained downward breakout.
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        // Feed 10 stable prices around 100
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Now feed strongly falling prices
        for i in 0..5 {
            let price = 100.0 * (1.0 - 0.003 * ((i + 1) as f64));
            strategy.push_price(price, 20_000 + (i as i64) * 1000);
            let snap = tf_snapshot(price);
            let _signal = strategy.detect_entry(&snap);
        }

        let last_price = 100.0 * (1.0 - 0.003 * 6.0);
        strategy.push_price(last_price, 25_000);
        let snap = tf_snapshot(last_price);
        let signal = strategy.detect_entry(&snap);

        match signal {
            Signal::MomentumShort {
                strength,
                velocity_pct,
            } => {
                assert!(strength >= 50.0);
                assert!(
                    velocity_pct < -0.25,
                    "velocity should be below negative breakout threshold"
                );
            }
            other => panic!(
                "Expected MomentumShort on downward breakout, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_trend_follower_no_signal_insufficient_prices() {
        let params = default_tf_params(); // min_price_count = 10
        let mut strategy = TrendFollowerStrategy::new(params);

        // Only 5 prices, below min_price_count
        for i in 0..5 {
            strategy.push_price(100.0 + (i as f64) * 0.5, 1000 + (i as i64) * 1000);
        }

        let snap = tf_snapshot(strategy.current_price().unwrap());
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should return NoSignal with insufficient price history"
        );
    }

    #[test]
    fn test_trend_follower_no_signal_flat_prices() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        // Feed 15 flat prices (no breakout)
        for i in 0..15 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        let snap = tf_snapshot(100.0);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(signal, Signal::NoSignal, "Should not signal on flat prices");
    }

    #[test]
    fn test_trend_follower_no_signal_below_threshold() {
        // Velocity slightly below breakout threshold
        let params = default_tf_params(); // threshold = 0.25
        let mut strategy = TrendFollowerStrategy::new(params);

        // Feed 10 stable prices
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Rising but only 0.1% per tick (below 0.25% threshold)
        for i in 0..5 {
            let price = 100.0 * (1.0 + 0.001 * ((i + 1) as f64));
            strategy.push_price(price, 20_000 + (i as i64) * 1000);
        }

        let snap = tf_snapshot(strategy.current_price().unwrap());
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not signal when velocity is below breakout threshold"
        );
    }

    #[test]
    fn test_trend_follower_no_signal_without_confirmation() {
        // One big spike but not enough consecutive ticks
        let params = default_tf_params(); // confirmation_ticks = 3
        let mut strategy = TrendFollowerStrategy::new(params);

        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }

        // Only 1 tick with high velocity
        strategy.push_price(100.5, 20_000); // 0.5% jump, above 0.25% threshold

        let snap = tf_snapshot(100.5);
        let signal = strategy.detect_entry(&snap);
        assert_eq!(
            signal,
            Signal::NoSignal,
            "Should not signal with only 1 breakout tick (need 3)"
        );
    }

    #[test]
    fn test_trend_follower_exit_stop_loss() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        // Need price history for velocity computation in exit
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        strategy.push_price(97.5, 20_000);

        let snap = tf_snapshot(97.5);
        let ctx = tf_exit_context(true, 100.0, 97.5, 100.5, 30);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::StopLoss);
            }
            other => panic!("Expected ExitLong(StopLoss), got {:?}", other),
        }
    }

    #[test]
    fn test_trend_follower_exit_take_profit() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        strategy.push_price(106.0, 20_000);

        let snap = tf_snapshot(106.0);
        let ctx = tf_exit_context(true, 100.0, 106.0, 106.0, 120);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TakeProfit);
            }
            other => panic!("Expected ExitLong(TakeProfit), got {:?}", other),
        }
    }

    #[test]
    fn test_trend_follower_exit_trailing_stop() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        strategy.push_price(103.0, 20_000);

        // Peak was at 104 (4% gain, above activation of 2.5%).
        // Current is 103 (1% retracement from peak, but retracement from entry: 4% - 3% = 1% pnl).
        // Retracement from peak: (104 - 103) / 104 * 100 = 0.96% — below 1.5% trailing.
        // Need bigger retracement.
        let snap = tf_snapshot(102.0);
        // Peak at 105 (5% gain, above activation 2.5%), current at 102 (2% gain).
        // Retracement from peak: (105 - 102) / 105 * 100 = 2.86% > 1.5% trailing.
        let ctx = PositionContext {
            is_long: true,
            entry_price: 100.0,
            current_price: 102.0,
            peak_price: 105.0,
            hold_secs: 300,
            max_hold_secs: 7200,
            take_profit_pct: 5.0,
            stop_loss_pct: 2.0,
            trailing_stop_pct: 1.5,
            trailing_activation_pct: 2.5,
        };

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TrailingStop);
            }
            other => panic!("Expected ExitLong(TrailingStop), got {:?}", other),
        }
    }

    #[test]
    fn test_trend_follower_exit_time_stop() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        strategy.push_price(100.5, 20_000);

        let snap = tf_snapshot(100.5);
        // Held 8000s, max is 7200s
        let ctx = tf_exit_context(true, 100.0, 100.5, 100.5, 8000);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(reason, ExitReason::TimeStop);
            }
            other => panic!("Expected ExitLong(TimeStop), got {:?}", other),
        }
    }

    #[test]
    fn test_trend_follower_exit_on_trend_exhaustion() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        // Feed stable prices (velocity ≈ 0%, well below threshold)
        for i in 0..10 {
            strategy.push_price(100.0, 1000 + (i as i64) * 1000);
        }
        // Push current price that makes velocity very low
        strategy.push_price(100.01, 20_000);

        let snap = tf_snapshot(100.01);
        // In profit, held long enough for exhaustion check
        let ctx = tf_exit_context(true, 100.0, 100.01, 100.5, 300);

        let result = strategy.detect_exit(&snap, &ctx);
        match result {
            Some(Signal::ExitLong { reason }) => {
                assert_eq!(
                    reason,
                    ExitReason::MomentumLost,
                    "Should exit on trend exhaustion when velocity drops"
                );
            }
            other => panic!(
                "Expected ExitLong(MomentumLost) on trend exhaustion, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_trend_follower_no_exit_when_position_stable() {
        let params = default_tf_params();
        let mut strategy = TrendFollowerStrategy::new(params);

        // Feed prices with some upward trend (velocity above threshold)
        for i in 0..10 {
            strategy.push_price(100.0 + (i as f64) * 0.1, 1000 + (i as i64) * 1000);
        }
        strategy.push_price(101.5, 20_000);

        let snap = tf_snapshot(101.5);
        // Small profit, well within all exit thresholds, trend still active
        let ctx = tf_exit_context(true, 100.0, 101.5, 101.5, 60);

        let result = strategy.detect_exit(&snap, &ctx);
        assert!(
            result.is_none(),
            "Expected no exit for stable trending position, got {:?}",
            result
        );
    }

    #[test]
    fn test_trend_follower_factory_creates_strategy() {
        let params = default_tf_params();
        let strategy = create_trend_follower_strategy(params).unwrap();
        assert_eq!(strategy.name(), "trend-follower");
    }

    #[test]
    fn test_trend_follower_factory_from_config() {
        let fallback = default_params();
        let strategy = create_strategy_from_config(
            "trend-follower",
            None, // No sub-table, use defaults
            fallback,
        )
        .unwrap();
        assert_eq!(strategy.name(), "trend-follower");
    }

    #[test]
    fn test_trend_follower_factory_from_config_with_table() {
        let fallback = default_params();
        let toml_str = r#"
            breakout_threshold_pct = 0.35
            confirmation_ticks = 5
            min_price_count = 40
            clip_size_usd = 50.0
            take_profit_pct = 6.0
            stop_loss_pct = 2.5
            trailing_stop_pct = 2.0
            trailing_activation_pct = 3.0
            max_hold_secs = 10800
            cooldown_after_loss_secs = 120
            direction_bias = "long"
        "#;
        let value: toml::Value = toml::from_str(toml_str).unwrap();
        let strategy =
            create_strategy_from_config("trend-follower", Some(&value), fallback).unwrap();
        assert_eq!(strategy.name(), "trend-follower");
    }

    #[test]
    fn test_trend_follower_params_validation_rejects_zero_threshold() {
        let mut params = default_tf_params();
        params.breakout_threshold_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_trend_follower_params_validation_rejects_zero_confirmation() {
        let mut params = default_tf_params();
        params.confirmation_ticks = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_trend_follower_params_validation_rejects_zero_min_price_count() {
        let mut params = default_tf_params();
        params.min_price_count = 0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_trend_follower_params_validation_rejects_zero_clip_size() {
        let mut params = default_tf_params();
        params.clip_size_usd = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_trend_follower_params_validation_rejects_zero_tp() {
        let mut params = default_tf_params();
        params.take_profit_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_trend_follower_params_validation_rejects_zero_sl() {
        let mut params = default_tf_params();
        params.stop_loss_pct = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_trend_follower_params_validation_accepts_valid() {
        let params = default_tf_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_trend_follower_available_in_strategies_list() {
        let strategies = available_strategies();
        assert!(
            strategies.contains(&"trend-follower"),
            "trend-follower should be in available_strategies"
        );
    }

    #[test]
    fn test_trend_follower_unknown_strategy_lists_trend_follower() {
        let params = default_params();
        let result = create_strategy("nonexistent", params);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("trend-follower"),
            "Error message should list trend-follower as available, got: {}",
            err
        );
    }
}
