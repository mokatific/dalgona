use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Tracks previous pool utilization values to compute utilization velocity.
/// Used by engine tick loops to construct `PoolSnapshot` with accurate velocity data.
#[derive(Debug, Clone, Default)]
pub struct PoolStateTracker {
    prev_long_utilization: Option<f64>,
    prev_short_utilization: Option<f64>,
}

impl PoolStateTracker {
    /// Create a new tracker with no previous state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `PoolSnapshot` from raw pool info, computing velocity from the
    /// previous tick's utilization values.
    ///
    /// On the first call, velocity is 0.0 for both sides (no previous data).
    pub fn compute_snapshot(
        &mut self,
        aum_usd: f64,
        long_utilization: f64,
        short_utilization: f64,
    ) -> PoolSnapshot {
        let long_vel = match self.prev_long_utilization {
            Some(prev) => long_utilization - prev,
            None => 0.0,
        };
        let short_vel = match self.prev_short_utilization {
            Some(prev) => short_utilization - prev,
            None => 0.0,
        };

        self.prev_long_utilization = Some(long_utilization);
        self.prev_short_utilization = Some(short_utilization);

        PoolSnapshot {
            aum_usd,
            long_utilization,
            short_utilization,
            long_utilization_velocity: long_vel,
            short_utilization_velocity: short_vel,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub enum Signal {
    MomentumLong { strength: f64, velocity_pct: f64 },
    MomentumShort { strength: f64, velocity_pct: f64 },
    ExitLong { reason: ExitReason },
    ExitShort { reason: ExitReason },
    NoSignal,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExitReason {
    MomentumLost,
    ReversalDetected,
    StopLoss,
    TakeProfit,
    TrailingStop,
    TimeStop,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PricePoint {
    pub price: f64,
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone)]
pub struct MomentumDetector {
    pub threshold_pct: f64,
    pub lookback_count: usize,
    prices: VecDeque<PricePoint>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MomentumSnapshot {
    pub price_count: usize,
    pub current_price: f64,
    pub price_velocity_pct: f64,
    pub direction: TradeDirection,
    pub strength: f64,
    pub volatility_pct: f64,
    /// Pool utilization data for LP consumption detection.
    /// None when pool data is unavailable (e.g., API failure or strategy doesn't use it).
    pub pool_data: Option<PoolSnapshot>,
}

/// Pool utilization snapshot used by LP consumption strategies.
/// Represents the current state of a Flash Trade liquidity pool's custodies.
#[derive(Debug, Clone)]
pub struct PoolSnapshot {
    /// Total assets under management in the pool, in USD.
    pub aum_usd: f64,
    /// Utilization ratio for the long side (0.0-1.0).
    pub long_utilization: f64,
    /// Utilization ratio for the short side (0.0-1.0).
    pub short_utilization: f64,
    /// How much the long side utilization has changed over the lookback window (velocity).
    /// Positive = long utilization is increasing (LP bid-side being consumed).
    pub long_utilization_velocity: f64,
    /// How much the short side utilization has changed over the lookback window (velocity).
    /// Positive = short utilization is increasing (LP ask-side being consumed).
    pub short_utilization_velocity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TradeDirection {
    Long,
    Short,
    Neutral,
}

impl MomentumDetector {
    pub fn new(threshold_pct: f64, lookback_count: usize) -> Self {
        Self {
            threshold_pct,
            lookback_count,
            prices: VecDeque::with_capacity(lookback_count * 2),
        }
    }

    pub fn push_price(&mut self, price: f64, timestamp_ms: i64) {
        self.prices.push_back(PricePoint {
            price,
            timestamp_ms,
        });
        while self.prices.len() > self.lookback_count * 2 {
            self.prices.pop_front();
        }
    }

    pub fn analyze(&self) -> MomentumSnapshot {
        if self.prices.len() < 3 {
            return MomentumSnapshot {
                price_count: self.prices.len(),
                current_price: self.prices.back().map(|p| p.price).unwrap_or(0.0),
                price_velocity_pct: 0.0,
                direction: TradeDirection::Neutral,
                strength: 0.0,
                volatility_pct: 0.0,
                pool_data: None,
            };
        }

        let lookback = self.lookback_count.min(self.prices.len());
        // prices ordered newest-first
        let prices: Vec<&PricePoint> = self.prices.iter().rev().take(lookback).collect();
        let current_price = prices[0].price;
        let oldest_price = prices.last().unwrap().price;

        let velocity_pct = if oldest_price > 0.0 {
            (current_price - oldest_price) / oldest_price * 100.0
        } else {
            0.0
        };

        // Volatility: max drawdown from peak in the lookback window
        let mut peak = 0.0_f64;
        let mut max_drawdown = 0.0_f64;
        for p in prices.iter().rev() {
            if p.price > peak {
                peak = p.price;
            }
            let dd = (peak - p.price) / peak * 100.0;
            if dd > max_drawdown {
                max_drawdown = dd;
            }
        }

        // Count consecutive moves in same direction (reset on direction change)
        let mut max_consecutive: i32 = 0;
        let mut current_run: i32 = 0;
        for i in 1..prices.len() {
            let up = prices[i - 1].price > prices[i].price;
            let run_dir = if up { 1 } else { -1 };

            if i == 1 {
                current_run = run_dir;
            } else {
                let prev_up = prices[i - 2].price > prices[i - 1].price;
                let prev_dir = if prev_up { 1 } else { -1 };
                if run_dir == prev_dir {
                    current_run += run_dir;
                } else {
                    // Direction changed — check if this run beats the max
                    if current_run.abs() > max_consecutive.abs() {
                        max_consecutive = current_run;
                    }
                    current_run = run_dir;
                }
            }
        }
        // Final check for last run
        if current_run.abs() > max_consecutive.abs() {
            max_consecutive = current_run;
        }

        let direction = if velocity_pct > self.threshold_pct * 0.3 {
            TradeDirection::Long
        } else if velocity_pct < -self.threshold_pct * 0.3 {
            TradeDirection::Short
        } else {
            TradeDirection::Neutral
        };

        let strength = compute_signal_strength(
            velocity_pct,
            self.threshold_pct,
            max_consecutive,
            max_drawdown,
            lookback,
        );

        MomentumSnapshot {
            price_count: self.prices.len(),
            current_price,
            price_velocity_pct: velocity_pct,
            direction,
            strength,
            volatility_pct: max_drawdown,
            pool_data: None,
        }
    }

    pub fn detect_signal(&self, snapshot: &MomentumSnapshot) -> Signal {
        if snapshot.price_count < 5 {
            return Signal::NoSignal;
        }

        let abs_velocity = snapshot.price_velocity_pct.abs();
        let has_momentum = abs_velocity >= self.threshold_pct;
        let strong_enough = snapshot.strength >= 50.0;

        if has_momentum && strong_enough {
            match snapshot.direction {
                TradeDirection::Long => {
                    info!(
                        "LONG signal: velocity={:.3}%, strength={:.0}",
                        abs_velocity, snapshot.strength
                    );
                    Signal::MomentumLong {
                        strength: snapshot.strength,
                        velocity_pct: abs_velocity,
                    }
                }
                TradeDirection::Short => {
                    info!(
                        "SHORT signal: velocity={:.3}%, strength={:.0}",
                        abs_velocity, snapshot.strength
                    );
                    Signal::MomentumShort {
                        strength: snapshot.strength,
                        velocity_pct: abs_velocity,
                    }
                }
                TradeDirection::Neutral => Signal::NoSignal,
            }
        } else {
            debug!(
                "No signal: velocity={:.3}% (threshold={:.3}%), strength={:.0}",
                abs_velocity, self.threshold_pct, snapshot.strength
            );
            Signal::NoSignal
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn detect_exit(
        &self,
        snapshot: &MomentumSnapshot,
        is_long: bool,
        entry_price: f64,
        current_price: f64,
        peak_price: f64,
        hold_secs: u64,
        max_hold_secs: u64,
        tp_pct: f64,
        sl_pct: f64,
        trail_pct: f64,
        trail_act_pct: f64,
    ) -> Option<Signal> {
        // PnL from entry
        let pnl_pct = if is_long {
            (current_price - entry_price) / entry_price * 100.0
        } else {
            (entry_price - current_price) / entry_price * 100.0
        };

        // Peak profit from entry (how far price moved in our favor at peak)
        let peak_profit_pct = if entry_price > 0.0 {
            if is_long {
                (peak_price - entry_price) / entry_price * 100.0
            } else {
                // For shorts, peak_price is the LOWEST price seen
                (entry_price - peak_price) / entry_price * 100.0
            }
        } else {
            0.0
        };

        // Retracement from peak
        let retracement_pct = if peak_price > 0.0 && peak_profit_pct > 0.0 {
            if is_long {
                (peak_price - current_price) / peak_price * 100.0
            } else {
                (current_price - peak_price) / peak_price * 100.0
            }
        } else {
            0.0
        };

        // Stop loss check (absolute loss from entry)
        if pnl_pct <= -sl_pct {
            warn!("STOP LOSS: pnl={:.2}%, threshold=-{:.2}%", pnl_pct, sl_pct);
            return Some(exit_signal(is_long, ExitReason::StopLoss));
        }

        // Take profit check
        if pnl_pct >= tp_pct {
            info!("TAKE PROFIT: pnl={:.2}%, threshold={:.2}%", pnl_pct, tp_pct);
            return Some(exit_signal(is_long, ExitReason::TakeProfit));
        }

        // Trailing stop: activated after peak_profit exceeds trail_act_pct
        if peak_profit_pct >= trail_act_pct && retracement_pct >= trail_pct {
            warn!(
                "TRAILING STOP: retracement={:.2}%, trail={:.2}%, peak_profit={:.2}%",
                retracement_pct, trail_pct, peak_profit_pct
            );
            return Some(exit_signal(is_long, ExitReason::TrailingStop));
        }

        // Time stop
        if hold_secs >= max_hold_secs {
            warn!("TIME STOP: held {}s, max={}s", hold_secs, max_hold_secs);
            return Some(exit_signal(is_long, ExitReason::TimeStop));
        }

        // Momentum lost while in profit (hold > 2 min)
        if snapshot.direction == TradeDirection::Neutral && pnl_pct > 0.0 && hold_secs > 120 {
            debug!("Momentum lost, in profit — suggesting exit");
            return Some(exit_signal(is_long, ExitReason::MomentumLost));
        }

        // Reversal detection
        if is_long && snapshot.direction == TradeDirection::Short && snapshot.strength > 40.0 {
            warn!("REVERSAL detected while long");
            return Some(Signal::ExitLong {
                reason: ExitReason::ReversalDetected,
            });
        }
        if !is_long && snapshot.direction == TradeDirection::Long && snapshot.strength > 40.0 {
            warn!("REVERSAL detected while short");
            return Some(Signal::ExitShort {
                reason: ExitReason::ReversalDetected,
            });
        }

        None
    }
}

fn exit_signal(is_long: bool, reason: ExitReason) -> Signal {
    if is_long {
        Signal::ExitLong { reason }
    } else {
        Signal::ExitShort { reason }
    }
}

fn compute_signal_strength(
    velocity: f64,
    threshold: f64,
    consecutive: i32,
    volatility: f64,
    lookback: usize,
) -> f64 {
    let velocity_score = (velocity.abs() / threshold).min(1.0) * 50.0;
    let consecutive_score = (consecutive.abs() as f64 / lookback as f64).min(1.0) * 30.0;
    let volatility_penalty = volatility.min(20.0);
    (velocity_score + consecutive_score - volatility_penalty).max(0.0)
}

#[cfg(test)]
mod pool_tracker_tests {
    use super::*;

    #[test]
    fn test_pool_tracker_first_tick_zero_velocity() {
        let mut tracker = PoolStateTracker::new();
        let snap = tracker.compute_snapshot(1_000_000.0, 0.3, 0.2);
        assert!((snap.aum_usd - 1_000_000.0).abs() < 0.01);
        assert!((snap.long_utilization - 0.3).abs() < 0.001);
        assert!((snap.short_utilization - 0.2).abs() < 0.001);
        // First tick: velocity should be 0 (no previous data)
        assert!((snap.long_utilization_velocity - 0.0).abs() < 0.001);
        assert!((snap.short_utilization_velocity - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_pool_tracker_computes_velocity() {
        let mut tracker = PoolStateTracker::new();

        // Tick 1: long=0.3, short=0.2
        let snap1 = tracker.compute_snapshot(1_000_000.0, 0.3, 0.2);
        assert!((snap1.long_utilization_velocity).abs() < 0.001);

        // Tick 2: long=0.5, short=0.15
        let snap2 = tracker.compute_snapshot(1_000_000.0, 0.5, 0.15);
        assert!(
            (snap2.long_utilization_velocity - 0.2).abs() < 0.001,
            "long velocity should be 0.5 - 0.3 = 0.2"
        );
        assert!(
            (snap2.short_utilization_velocity - (-0.05)).abs() < 0.001,
            "short velocity should be 0.15 - 0.2 = -0.05"
        );

        // Tick 3: long=0.8, short=0.1
        let snap3 = tracker.compute_snapshot(1_200_000.0, 0.8, 0.1);
        assert!(
            (snap3.long_utilization_velocity - 0.3).abs() < 0.001,
            "long velocity should be 0.8 - 0.5 = 0.3"
        );
        assert!((snap3.short_utilization_velocity - (-0.05)).abs() < 0.001);
        assert!((snap3.aum_usd - 1_200_000.0).abs() < 0.01);
    }

    #[test]
    fn test_pool_tracker_velocity_with_stable_utilization() {
        let mut tracker = PoolStateTracker::new();
        tracker.compute_snapshot(1_000_000.0, 0.5, 0.5);

        // Same utilization → zero velocity
        let snap = tracker.compute_snapshot(1_000_000.0, 0.5, 0.5);
        assert!((snap.long_utilization_velocity).abs() < 0.001);
        assert!((snap.short_utilization_velocity).abs() < 0.001);
    }

    #[test]
    fn test_pool_tracker_default_is_new() {
        let tracker = PoolStateTracker::default();
        assert!(tracker.prev_long_utilization.is_none());
        assert!(tracker.prev_short_utilization.is_none());
    }
}
