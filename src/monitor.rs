//! Monitoring loop skeleton for continuous position/PnL tracking.
//!
//! Provides structured logging of:
//! - Open positions with unrealized PnL
//! - Account balance and circuit breaker status
//! - Per-strategy performance metrics
//! - Fee breakdown
//!
//! This module is designed to be called periodically from the main trading loop
//! (either paper or live) to produce structured monitoring output.

use chrono::{DateTime, Utc};
use serde::Serialize;
use tracing::info;

/// A snapshot of a single open position for monitoring purposes.
#[derive(Debug, Clone, Serialize)]
pub struct PositionSnapshot {
    pub strategy: String,
    pub market: String,
    pub direction: String,
    pub entry_price: f64,
    pub current_price: f64,
    pub size_usd: f64,
    pub unrealized_pnl_usd: f64,
    pub unrealized_pnl_pct: f64,
    pub entry_fee: f64,
    pub accrued_borrow_fee: f64,
    pub hold_secs: u64,
}

/// Per-strategy performance metrics for monitoring.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StrategyMetrics {
    pub strategy: String,
    pub market: String,
    pub trade_count: usize,
    pub win_count: usize,
    pub loss_count: usize,
    pub net_pnl: f64,
    pub total_fees: f64,
    pub entry_fees: f64,
    pub exit_fees: f64,
    pub borrow_fees: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
}

/// A complete monitoring snapshot, logged periodically.
#[derive(Debug, Clone, Serialize)]
pub struct MonitoringSnapshot {
    pub timestamp: String,
    pub elapsed_secs: f64,
    pub balance: f64,
    pub circuit_breaker_active: bool,
    pub open_positions: Vec<PositionSnapshot>,
    pub total_open_notional: f64,
    pub strategy_metrics: Vec<StrategyMetrics>,
    pub total_net_pnl: f64,
    pub total_fees: f64,
}

impl MonitoringSnapshot {
    /// Create a new monitoring snapshot from current state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        balance: f64,
        circuit_breaker_active: bool,
        start_time: DateTime<Utc>,
        positions: Vec<PositionSnapshot>,
        strategy_metrics: Vec<StrategyMetrics>,
    ) -> Self {
        let total_open_notional: f64 = positions.iter().map(|p| p.size_usd).sum();
        let total_net_pnl: f64 = strategy_metrics.iter().map(|m| m.net_pnl).sum();
        let total_fees: f64 = strategy_metrics.iter().map(|m| m.total_fees).sum();

        Self {
            timestamp: Utc::now().to_rfc3339(),
            elapsed_secs: (Utc::now() - start_time).num_seconds() as f64,
            balance,
            circuit_breaker_active,
            open_positions: positions,
            total_open_notional,
            strategy_metrics,
            total_net_pnl,
            total_fees,
        }
    }

    /// Log the monitoring snapshot with structured output.
    pub fn log(&self) {
        let hours = self.elapsed_secs / 3600.0;
        let mins = (self.elapsed_secs % 3600.0) / 60.0;

        info!("╔══════════════════════════════════════════════════════════════════════╗");
        info!(
            "║              MONITORING SNAPSHOT ({:.0}h {:.0}m elapsed)             ",
            hours, mins
        );
        info!("╠══════════════════════════════════════════════════════════════════════╣");
        info!(
            "║ Balance: ${:.2} | Circuit breaker: {} | Open positions: {}",
            self.balance,
            if self.circuit_breaker_active {
                "ACTIVE ⚠"
            } else {
                "OK"
            },
            self.open_positions.len(),
        );
        info!(
            "║ Total open notional: ${:.2} | Net PnL: ${:.2} | Total fees: ${:.2}",
            self.total_open_notional, self.total_net_pnl, self.total_fees
        );

        if !self.open_positions.is_empty() {
            info!("╠══════════════════════════════════════════════════════════════════════╣");
            info!(
                "║ {:<20} {:<5} {:>8} {:>8} {:>8} {:>6}",
                "Strategy:Market", "Dir", "Size$", "uPnL$", "Fees$", "Hold"
            );
            for pos in &self.open_positions {
                info!(
                    "║ {:<20} {:<5} {:>8.0} {:>8.2} {:>8.4} {:>4}m",
                    format!("{}:{}", pos.strategy, pos.market),
                    pos.direction,
                    pos.size_usd,
                    pos.unrealized_pnl_usd,
                    pos.entry_fee + pos.accrued_borrow_fee,
                    pos.hold_secs / 60,
                );
            }
        }

        if !self.strategy_metrics.is_empty() {
            info!("╠══════════════════════════════════════════════════════════════════════╣");
            info!(
                "║ {:<20} {:>5} {:>8} {:>8} {:>8} {:>5}",
                "Strategy:Market", "Trds", "Net$", "Fees$", "Win%", "Sharpe"
            );
            for m in &self.strategy_metrics {
                info!(
                    "║ {:<20} {:>5} {:>8.2} {:>8.2} {:>5.1}% {:>5.2}",
                    format!("{}:{}", m.strategy, m.market),
                    m.trade_count,
                    m.net_pnl,
                    m.total_fees,
                    m.win_rate,
                    m.sharpe_ratio,
                );
            }
        }

        info!("╚══════════════════════════════════════════════════════════════════════╝");
    }

    /// Write the monitoring snapshot to a JSON file (atomic write).
    pub fn write_to_file(&self, path: &str) -> anyhow::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = format!("{}.tmp", path);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }
}

/// Configuration for the monitoring loop.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// How often to emit monitoring snapshots (seconds).
    pub log_interval_secs: u64,
    /// Path to write monitoring snapshots (JSON).
    pub snapshot_path: String,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            log_interval_secs: 3600, // 1 hour
            snapshot_path: "data/monitoring/latest-snapshot.json".to_string(),
        }
    }
}

/// Tracker for monitoring loop timing.
#[derive(Debug)]
pub struct MonitorLoop {
    pub config: MonitorConfig,
    #[allow(dead_code)]
    pub start_time: DateTime<Utc>,
    pub last_log_time: DateTime<Utc>,
    pub tick_count: u64,
}

impl MonitorLoop {
    pub fn new(config: MonitorConfig) -> Self {
        let now = Utc::now();
        Self {
            config,
            start_time: now,
            last_log_time: now,
            tick_count: 0,
        }
    }

    /// Check if it's time to emit a monitoring snapshot.
    pub fn should_log(&self) -> bool {
        let elapsed = (Utc::now() - self.last_log_time).num_seconds() as u64;
        elapsed >= self.config.log_interval_secs
    }

    /// Record that a monitoring snapshot was emitted.
    pub fn mark_logged(&mut self) {
        self.last_log_time = Utc::now();
        self.tick_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_snapshot_creation() {
        let snapshot = MonitoringSnapshot::new(
            1000.0,
            false,
            Utc::now(),
            vec![PositionSnapshot {
                strategy: "momentum-scalper".to_string(),
                market: "SOL".to_string(),
                direction: "LONG".to_string(),
                entry_price: 150.0,
                current_price: 152.0,
                size_usd: 500.0,
                unrealized_pnl_usd: 6.67,
                unrealized_pnl_pct: 1.33,
                entry_fee: 0.5,
                accrued_borrow_fee: 0.1,
                hold_secs: 1800,
            }],
            vec![StrategyMetrics {
                strategy: "momentum-scalper".to_string(),
                market: "SOL".to_string(),
                trade_count: 5,
                win_count: 3,
                loss_count: 2,
                net_pnl: 12.5,
                total_fees: 3.0,
                entry_fees: 1.5,
                exit_fees: 1.5,
                borrow_fees: 0.0,
                win_rate: 60.0,
                sharpe_ratio: 1.2,
            }],
        );

        assert!((snapshot.balance - 1000.0).abs() < 0.01);
        assert!(!snapshot.circuit_breaker_active);
        assert_eq!(snapshot.open_positions.len(), 1);
        assert_eq!(snapshot.strategy_metrics.len(), 1);
        assert!((snapshot.total_open_notional - 500.0).abs() < 0.01);
        assert!((snapshot.total_net_pnl - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_monitoring_snapshot_serialization() {
        let snapshot = MonitoringSnapshot::new(1000.0, false, Utc::now(), vec![], vec![]);
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("\"balance\":1000.0"));
        assert!(json.contains("\"circuit_breaker_active\":false"));
        assert!(json.contains("\"open_positions\":[]"));
        assert!(json.contains("\"strategy_metrics\":[]"));
    }

    #[test]
    fn test_monitoring_snapshot_write() {
        let snapshot = MonitoringSnapshot::new(500.0, true, Utc::now(), vec![], vec![]);
        let tmp_dir = std::env::temp_dir().join("dalgona_monitor_test");
        std::fs::create_dir_all(&tmp_dir).ok();
        let path = tmp_dir.join("snapshot.json").to_str().unwrap().to_string();
        snapshot.write_to_file(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        // serde_json pretty-print uses spaces around colons: "circuit_breaker_active": true
        assert!(
            content.contains("circuit_breaker_active"),
            "Expected circuit_breaker_active field in output"
        );
        // Re-parse to verify the actual value
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["circuit_breaker_active"], true);
        assert_eq!(parsed["balance"], 500.0);
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_monitor_loop_timing() {
        let config = MonitorConfig {
            log_interval_secs: 60,
            snapshot_path: String::new(),
        };
        let mut monitor = MonitorLoop::new(config);
        assert!(!monitor.should_log()); // Just created, not enough time elapsed

        // Simulate time passing by manually setting last_log_time to the past
        monitor.last_log_time = Utc::now() - chrono::Duration::seconds(120);
        assert!(monitor.should_log());

        monitor.mark_logged();
        assert!(!monitor.should_log()); // Just logged
        assert_eq!(monitor.tick_count, 1);
    }

    #[test]
    fn test_monitor_config_default() {
        let config = MonitorConfig::default();
        assert_eq!(config.log_interval_secs, 3600);
        assert_eq!(config.snapshot_path, "data/monitoring/latest-snapshot.json");
    }
}
