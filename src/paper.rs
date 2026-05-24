use crate::config::Config;
use crate::flash_api::FlashClient;
use crate::risk::{RiskManager, TradeLog, TradeRecord};
use crate::signal::{ExitReason, MomentumDetector, MomentumSnapshot, Signal};
use chrono::Utc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

/// Fallback taker fee rate if API preview fails
const FALLBACK_FEE_RATE: f64 = 0.001; // 0.1%

/// Estimated hourly borrow/margin fee rate (varies by pool utilization)
/// Flash charges dynamic borrow fees; 0.01%/hr is a conservative estimate
const BORROW_FEE_HOURLY: f64 = 0.0001; // 0.01% per hour on position notional

pub struct PaperEngine {
    config: Config,
    flash: FlashClient,
    detector: MomentumDetector,
    risk: Arc<RiskManager>,
    trade_log: TradeLog,
    position: Option<PaperPosition>,
    running: Arc<AtomicBool>,
    sim_balance: f64,
    /// Entry fee captured from live API preview at open time
    pending_entry_fee: f64,
}

/// Extended position with fee tracking for paper trades.
#[derive(Debug, Clone)]
struct PaperPosition {
    inner: crate::risk::Position,
    /// Entry fee paid (from live API preview)
    entry_fee: f64,
    /// Accumulated borrow/funding fee estimate (notional * hourly_rate * hours)
    accrued_borrow_fee: f64,
}

impl PaperPosition {
    fn update_price(&mut self, price: f64) {
        self.inner.update_price(price);
        // Accrue borrow fee: hourly rate on notional, pro-rated per tick
        // This is called each tick so we accrue incrementally
        let hours_held = 1.0 / 3600.0 * (self.config_poll_interval_secs() as f64);
        self.accrued_borrow_fee += self.inner.size_usd * BORROW_FEE_HOURLY * hours_held;
    }

    fn config_poll_interval_secs(&self) -> u64 {
        // Default 5s; can't easily access config here, so hardcode
        5
    }

    fn total_fees(&self) -> f64 {
        self.entry_fee + self.accrued_borrow_fee
    }
}

impl PaperEngine {
    pub fn new(config: Config, starting_balance: f64) -> Self {
        let flash = FlashClient::new(&config.flash.api_url);
        let detector = MomentumDetector::new(
            config.strategy.momentum_threshold_pct,
            config.strategy.lookback_count,
        );
        let risk = Arc::new(RiskManager::new(config.risk.clone(), starting_balance));
        let trade_log = TradeLog::new("paper-trades.json");

        Self {
            config,
            flash,
            detector,
            risk,
            trade_log,
            position: None,
            running: Arc::new(AtomicBool::new(true)),
            sim_balance: starting_balance,
            pending_entry_fee: 0.0,
        }
    }

    pub fn shutdown_handle(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        info!("=== Dalgona PAPER Trading Mode ===");
        info!("Market: {}", self.config.flash.market);
        info!("Leverage: {}x", self.config.flash.leverage);
        info!("Clip: ${:.0}", self.config.strategy.clip_size_usd);
        info!("Simulated balance: ${:.2}", self.sim_balance);
        info!("Fee estimation: live API preview (entry + exit) + {}%/hr borrow", BORROW_FEE_HOURLY * 100.0);
        warn!("PAPER MODE -- no real transactions will be signed");

        let initial_price = self.flash.get_price(&self.config.flash.market).await?;
        info!("Initial price: ${:.2}", initial_price);
        self.detector.push_price(initial_price, now_ms());

        loop {
            if !self.running.load(Ordering::Relaxed) {
                info!("Shutdown signal received");
                break;
            }

            if self.risk.is_halted() {
                error!("Circuit breaker active -- stopping paper trading");
                break;
            }

            if let Err(e) = self.tick().await {
                error!("Tick error: {:#}", e);
                sleep(Duration::from_secs(10)).await;
                continue;
            }

            sleep(self.config.poll_interval()).await;
        }

        // Show open position at shutdown with fee breakdown
        if let Some(ref pos) = self.position {
            let pnl = pos.inner.unrealized_pnl_usd();
            info!(
                "Open paper position at shutdown: {} {} @ ${:.2} | uPnL: ${:.2} | accrued fees: ${:.4} (entry=${:.4} borrow=${:.4})",
                if pos.inner.is_long { "LONG" } else { "SHORT" },
                pos.inner.asset, pos.inner.current_price, pnl,
                pos.total_fees(), pos.entry_fee, pos.accrued_borrow_fee
            );
        }

        let stats = self.trade_log.stats();
        info!("=== Paper Trading Final Stats ===");
        info!("Trades: {} | Win rate: {:.1}%", stats.total_trades, stats.win_rate);
        info!("Gross PnL: ${:.2} | Fees: ${:.2} | Net PnL: ${:.2}", stats.total_pnl, stats.total_fees, stats.net_pnl);
        info!("Simulated balance: ${:.2}", self.sim_balance);

        Ok(())
    }

    async fn tick(&mut self) -> anyhow::Result<()> {
        let market = &self.config.flash.market;
        let price = self.flash.get_price(market).await?;
        self.detector.push_price(price, now_ms());

        let snapshot = self.detector.analyze();

        debug!(
            "[{}] prices={} velocity={:.4}% dir={:?} strength={:.0}",
            market, snapshot.price_count, snapshot.price_velocity_pct,
            snapshot.direction, snapshot.strength,
        );

        match &self.position {
            None => {
                self.handle_no_position(&snapshot, price).await?;
            }
            Some(_) => {
                self.manage_position(&snapshot, price).await?;
            }
        }

        Ok(())
    }

    async fn handle_no_position(
        &mut self,
        snapshot: &MomentumSnapshot,
        current_price: f64,
    ) -> anyhow::Result<()> {
        if let Err(e) = self.risk.check_can_trade(self.sim_balance) {
            debug!("Cannot trade: {}", e);
            return Ok(());
        }

        let clip = self.config.strategy.clip_size_usd;
        let notional = clip * self.config.flash.leverage;
        if let Err(e) = self.risk.check_position_size(notional) {
            warn!("{}", e);
            return Ok(());
        }

        if self.sim_balance < clip {
            debug!("Insufficient simulated balance: ${:.2} < ${:.2}", self.sim_balance, clip);
            return Ok(());
        }

        let bias = self.config.strategy.direction_bias.to_lowercase();
        let leverage = self.config.flash.leverage;
        let signal = self.detector.detect_signal(snapshot);

        match signal {
            Signal::MomentumLong { strength, velocity_pct } if bias != "short" => {
                self.paper_open(true, clip, leverage, current_price, strength, velocity_pct).await?;
            }
            Signal::MomentumShort { strength, velocity_pct } if bias != "long" => {
                self.paper_open(false, clip, leverage, current_price, strength, velocity_pct).await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn paper_open(
        &mut self,
        is_long: bool,
        clip_usd: f64,
        leverage: f64,
        current_price: f64,
        strength: f64,
        velocity_pct: f64,
    ) -> anyhow::Result<()> {
        let trade_type = if is_long { "LONG" } else { "SHORT" };

        // Preview against live API to get REAL entry fee and price
        let preview = self.flash.preview_open_position(
            &self.config.flash.input_token,
            &self.config.flash.market,
            clip_usd,
            leverage,
            trade_type,
        ).await?;

        let entry_fee = preview.entry_fee.as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(clip_usd * FALLBACK_FEE_RATE);

        let entry_price = preview.new_entry_price.as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(current_price);

        let liq_price = preview.new_liquidation_price.as_deref().unwrap_or("N/A");
        let notional = clip_usd * leverage;

        if let Some(ref err) = preview.err {
            warn!("Preview returned error (proceeding with estimates): {}", err);
        }

        info!(
            ">>> [PAPER] OPENING {} ${:.0} x {:.0}x @ ${:.2} | liq=${} | entry_fee=${:.4} ({:.3}%) | strength={:.0} velocity={:.3}%",
            trade_type, clip_usd, leverage, entry_price, liq_price,
            entry_fee, (entry_fee / clip_usd) * 100.0,
            strength, velocity_pct
        );

        self.position = Some(PaperPosition {
            inner: crate::risk::Position {
                position_key: format!("paper-{}", Utc::now().timestamp()),
                symbol: format!("{}-USD", self.config.flash.market),
                asset: self.config.flash.market.clone(),
                is_long,
                entry_price,
                current_price: entry_price,
                peak_price: entry_price,
                size_usd: notional,
                leverage,
                open_time: Utc::now(),
            },
            entry_fee,
            accrued_borrow_fee: 0.0,
        });

        self.sim_balance -= entry_fee;
        self.pending_entry_fee = entry_fee;

        Ok(())
    }

    async fn manage_position(
        &mut self,
        snapshot: &MomentumSnapshot,
        current_price: f64,
    ) -> anyhow::Result<()> {
        let pos = match &mut self.position {
            Some(p) => p,
            None => return Ok(()),
        };
        pos.update_price(current_price);

        let exit_signal = self.detector.detect_exit(
            snapshot,
            pos.inner.is_long,
            pos.inner.entry_price,
            current_price,
            pos.inner.peak_price,
            pos.inner.hold_duration_secs(),
            self.config.strategy.max_hold_secs,
            self.config.strategy.take_profit_pct,
            self.config.strategy.stop_loss_pct,
            self.config.strategy.trailing_stop_pct,
            self.config.strategy.trailing_activation_pct,
        );

        match exit_signal {
            Some(Signal::ExitLong { reason } | Signal::ExitShort { reason }) => {
                self.paper_close(current_price, reason).await?;
            }
            Some(_) => {}
            None => {
                debug!(
                    "[PAPER] Holding {} {} @ ${:.2} | uPnL: ${:.2} ({:.2}%) | fees: ${:.4} (borrow: ${:.4}) | hold: {}s",
                    if pos.inner.is_long { "LONG" } else { "SHORT" },
                    pos.inner.asset, current_price,
                    pos.inner.unrealized_pnl_usd(),
                    pos.inner.unrealized_pnl_pct(),
                    pos.total_fees(), pos.accrued_borrow_fee,
                    pos.inner.hold_duration_secs()
                );
            }
        }

        Ok(())
    }

    async fn paper_close(
        &mut self,
        exit_price: f64,
        reason: ExitReason,
    ) -> anyhow::Result<()> {
        let pos = match self.position.take() {
            Some(p) => p,
            None => return Ok(()),
        };

        let gross_pnl = pos.inner.unrealized_pnl_usd();

        // Estimate exit fee from live API (with fallback)
        let exit_fee = match self.flash.preview_exit_fee(
            &pos.inner.position_key,
            pos.inner.size_usd,
        ).await {
            Ok(fee) => {
                debug!("Exit fee from API: ${:.4}", fee);
                fee
            }
            Err(e) => {
                debug!("Exit fee preview failed (using fallback): {:#}", e);
                pos.inner.size_usd * FALLBACK_FEE_RATE
            }
        };

        let total_fees = pos.entry_fee + exit_fee + pos.accrued_borrow_fee;
        let net_pnl = gross_pnl - exit_fee - pos.accrued_borrow_fee;

        let hold_mins = pos.inner.hold_duration_secs() as f64 / 60.0;

        info!(
            "<<< [PAPER] CLOSING {} {} | ${:.2} -> ${:.2} | reason={:?} | hold={:.1}min",
            if pos.inner.is_long { "LONG" } else { "SHORT" },
            pos.inner.asset, pos.inner.entry_price, exit_price, reason, hold_mins
        );
        info!(
            "    gross_pnl=${:.2} | entry_fee=${:.4} exit_fee=${:.4} borrow_fee=${:.4} | total_fees=${:.4} ({:.3}%) | net=${:.2}",
            gross_pnl, pos.entry_fee, exit_fee, pos.accrued_borrow_fee,
            total_fees, (total_fees / pos.inner.size_usd) * 100.0, net_pnl
        );

        // Update simulated balance
        self.sim_balance += net_pnl;

        self.risk.record_trade_result(net_pnl, total_fees, self.sim_balance);

        self.trade_log.record(TradeRecord {
            symbol: pos.inner.symbol.clone(),
            direction: if pos.inner.is_long { "LONG".into() } else { "SHORT".into() },
            entry_price: pos.inner.entry_price,
            exit_price,
            size_usd: pos.inner.size_usd,
            pnl: net_pnl,
            fees: total_fees,
            hold_secs: pos.inner.hold_duration_secs(),
            exit_reason: format!("{:?}", reason),
            timestamp: Utc::now(),
        });

        if net_pnl < 0.0 {
            self.risk.set_cooldown(self.config.strategy.cooldown_after_loss_secs);
        }

        info!("[PAPER] Simulated balance: ${:.2}", self.sim_balance);

        Ok(())
    }
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}
