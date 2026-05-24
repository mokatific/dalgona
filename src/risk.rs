use crate::config::RiskConfig;
use chrono::{DateTime, Datelike, Utc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use tracing::info;

#[derive(Debug, Clone)]
pub struct Position {
    pub position_key: String,
    pub symbol: String,
    pub asset: String,
    pub is_long: bool,
    pub entry_price: f64,
    pub current_price: f64,
    pub peak_price: f64,
    pub size_usd: f64,
    pub leverage: f64,
    pub open_time: DateTime<Utc>,
}

impl Position {
    pub fn unrealized_pnl_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        if self.is_long {
            (self.current_price - self.entry_price) / self.entry_price * 100.0
        } else {
            (self.entry_price - self.current_price) / self.entry_price * 100.0
        }
    }

    pub fn unrealized_pnl_usd(&self) -> f64 {
        self.size_usd * self.unrealized_pnl_pct() / 100.0
    }

    pub fn hold_duration_secs(&self) -> u64 {
        (Utc::now() - self.open_time).num_seconds().max(0) as u64
    }

    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
        if self.is_long {
            // Longs: peak is the highest price seen
            if price > self.peak_price {
                self.peak_price = price;
            }
        } else {
            // Shorts: peak is the LOWEST price seen (best for shorts)
            if self.peak_price == 0.0 || price < self.peak_price {
                self.peak_price = price;
            }
        }
    }
}

#[derive(Debug)]
pub struct RiskManager {
    config: RiskConfig,
    daily_pnl: Mutex<f64>,
    total_fees: Mutex<f64>,
    daily_peak_balance: Mutex<f64>,
    initial_balance: Mutex<f64>,
    trade_date: Mutex<u32>,
    halted: AtomicBool,
    cooldown_until: Mutex<Option<DateTime<Utc>>>,
}

impl RiskManager {
    pub fn new(config: RiskConfig, initial_balance: f64) -> Self {
        let peak = if initial_balance > 0.0 { initial_balance } else { 0.0 };
        let today = Utc::now().day();
        Self {
            config,
            daily_pnl: Mutex::new(0.0),
            total_fees: Mutex::new(0.0),
            daily_peak_balance: Mutex::new(peak),
            initial_balance: Mutex::new(initial_balance),
            trade_date: Mutex::new(today),
            halted: AtomicBool::new(false),
            cooldown_until: Mutex::new(None),
        }
    }

    /// Check if the day has rolled over and reset daily counters if so.
    fn maybe_reset_day(&self) {
        let today = Utc::now().day();
        let mut date_guard = self.trade_date.lock().unwrap();
        if *date_guard != today {
            info!("New day detected ({} -> {}), resetting daily PnL", *date_guard, today);
            *date_guard = today;
            *self.daily_pnl.lock().unwrap() = 0.0;
            // Reset peak to current known balance
            let balance = *self.initial_balance.lock().unwrap()
                + *self.daily_pnl.lock().unwrap();
            *self.daily_peak_balance.lock().unwrap() = balance;
        }
    }

    pub fn check_can_trade(&self, balance: f64) -> Result<(), String> {
        self.maybe_reset_day();

        if self.halted.load(Ordering::Relaxed) {
            return Err("Trading HALTED — circuit breaker triggered".into());
        }

        if let Ok(guard) = self.cooldown_until.lock() {
            if let Some(until) = *guard {
                if Utc::now() < until {
                    return Err(format!("Cooldown active until {}", until.format("%H:%M:%S")));
                }
            }
        }

        let daily_pnl = *self.daily_pnl.lock().unwrap();
        if daily_pnl.abs() >= self.config.max_daily_loss_usd && daily_pnl < 0.0 {
            self.halted.store(true, Ordering::Relaxed);
            return Err(format!(
                "Daily loss limit reached: ${:.2} / ${:.2}",
                daily_pnl.abs(), self.config.max_daily_loss_usd
            ));
        }

        if balance > 0.0 {
            let peak = *self.daily_peak_balance.lock().unwrap();
            let effective_peak = if peak > 0.0 { peak } else { balance };
            let drawdown = (effective_peak - balance) / effective_peak * 100.0;
            if drawdown >= self.config.max_drawdown_pct {
                self.halted.store(true, Ordering::Relaxed);
                return Err(format!(
                    "Max drawdown reached: {:.1}% / {:.1}%",
                    drawdown, self.config.max_drawdown_pct
                ));
            }
        }

        Ok(())
    }

    /// Validate position size against configured maximum.
    pub fn check_position_size(&self, notional_usd: f64) -> Result<(), String> {
        if notional_usd > self.config.max_position_notional_usd {
            return Err(format!(
                "Position size ${:.2} exceeds max ${:.2}",
                notional_usd, self.config.max_position_notional_usd
            ));
        }
        Ok(())
    }

    pub fn record_trade_result(&self, pnl: f64, fees: f64, balance: f64) {
        self.maybe_reset_day();

        {
            let mut daily = self.daily_pnl.lock().unwrap();
            let prev = *daily;
            *daily = prev + pnl;
        }
        {
            let mut total_fees = self.total_fees.lock().unwrap();
            *total_fees += fees;
        }

        // Update peak balance
        if balance > 0.0 {
            let mut peak = self.daily_peak_balance.lock().unwrap();
            if balance > *peak {
                *peak = balance;
            }
            // Also update the tracked initial balance
            let mut init = self.initial_balance.lock().unwrap();
            *init = balance;
        }

        let daily_pnl = *self.daily_pnl.lock().unwrap();
        if pnl < 0.0 {
            info!("Loss recorded: ${:.2}, daily PnL: ${:.2}", pnl, daily_pnl);
        } else {
            info!("Win recorded: ${:.2}, daily PnL: ${:.2}", pnl, daily_pnl);
        }
    }

    pub fn set_cooldown(&self, secs: u64) {
        if let Ok(mut guard) = self.cooldown_until.lock() {
            *guard = Some(Utc::now() + chrono::Duration::seconds(secs as i64));
        }
    }

    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    pub fn total_fees(&self) -> f64 {
        *self.total_fees.lock().unwrap()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size_usd: f64,
    pub pnl: f64,
    pub fees: f64,
    pub hold_secs: u64,
    pub exit_reason: String,
    pub timestamp: DateTime<Utc>,
}

pub struct TradeLog {
    trades: Vec<TradeRecord>,
    filepath: String,
}

impl TradeLog {
    pub fn new(filepath: &str) -> Self {
        Self {
            trades: Vec::new(),
            filepath: filepath.to_string(),
        }
    }

    pub fn record(&mut self, trade: TradeRecord) {
        let is_win = trade.pnl > 0.0;
        info!(
            "TRADE CLOSED: {} {} @ ${:.2} -> ${:.2} | PnL: ${:.2} ({}) | fees: ${:.2} | hold: {}s",
            trade.direction, trade.symbol, trade.entry_price, trade.exit_price,
            trade.pnl, if is_win { "WIN" } else { "LOSS" }, trade.fees, trade.hold_secs
        );
        self.trades.push(trade);
        self.flush();
    }

    pub fn stats(&self) -> TradeStats {
        if self.trades.is_empty() {
            return TradeStats::default();
        }
        let wins = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let total = self.trades.len();
        let total_pnl: f64 = self.trades.iter().map(|t| t.pnl).sum();
        let total_fees: f64 = self.trades.iter().map(|t| t.fees).sum();
        let avg_hold = self.trades.iter().map(|t| t.hold_secs as f64).sum::<f64>() / total as f64;
        let best = self.trades.iter().map(|t| t.pnl).fold(f64::NEG_INFINITY, f64::max);
        let worst = self.trades.iter().map(|t| t.pnl).fold(f64::INFINITY, f64::min);

        TradeStats {
            total_trades: total,
            wins,
            win_rate: wins as f64 / total as f64 * 100.0,
            total_pnl,
            total_fees,
            net_pnl: total_pnl - total_fees,
            avg_hold_secs: avg_hold,
            best_trade: best,
            worst_trade: worst,
        }
    }

    /// Atomic write: write to temp file then rename.
    fn flush(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.trades) {
            let tmp_path = format!("{}.tmp", self.filepath);
            if std::fs::write(&tmp_path, &json).is_ok() {
                if std::fs::rename(&tmp_path, &self.filepath).is_err() {
                    let _ = std::fs::write(&self.filepath, &json);
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct TradeStats {
    pub total_trades: usize,
    pub wins: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub total_fees: f64,
    pub net_pnl: f64,
    pub avg_hold_secs: f64,
    pub best_trade: f64,
    pub worst_trade: f64,
}
