use crate::config::Config;
use crate::flash_api::FlashClient;
use crate::risk::{RiskManager, TradeLog, TradeRecord};
use crate::signal::{ExitReason, MomentumSnapshot, Signal};
use crate::strategy::{self, PositionContext, Strategy};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::{Duration, sleep};
use tracing::{debug, error, info, warn};

/// Fallback taker fee rate if API preview fails
const FALLBACK_FEE_RATE: f64 = 0.001; // 0.1%

/// Estimated hourly borrow/margin fee rate (varies by pool utilization)
/// Flash charges dynamic borrow fees; 0.01%/hr is a conservative estimate
const BORROW_FEE_HOURLY: f64 = 0.0001; // 0.01% per hour on position notional

pub struct PaperEngine {
    config: Config,
    flash: FlashClient,
    strategy: Box<dyn Strategy>,
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
    pub fn new(
        config: Config,
        starting_balance: f64,
        strategy_name: Option<&str>,
    ) -> anyhow::Result<Self> {
        let flash = FlashClient::new(&config.flash.api_url);
        let resolved_name = config.strategy.resolve_active(strategy_name);
        let sub_table = config.strategy.get_sub_table(&resolved_name);
        let fallback_params = config.strategy.get_params(&resolved_name)?;
        let strat =
            strategy::create_strategy_from_config(&resolved_name, sub_table, fallback_params)?;
        let risk = Arc::new(RiskManager::new(config.risk.clone(), starting_balance));
        let trade_log = TradeLog::new("paper-trades.json");

        info!(
            "Strategy: {} (from {})",
            strat.name(),
            if strategy_name.is_some() {
                "CLI flag"
            } else {
                "config"
            }
        );

        Ok(Self {
            config,
            flash,
            strategy: strat,
            risk,
            trade_log,
            position: None,
            running: Arc::new(AtomicBool::new(true)),
            sim_balance: starting_balance,
            pending_entry_fee: 0.0,
        })
    }

    pub fn shutdown_handle(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        info!("=== Dalgona PAPER Trading Mode ===");
        info!("Market: {}", self.config.flash.market);
        info!("Leverage: {}x", self.config.flash.leverage);
        let params = self.strategy.parameters();
        info!("Clip: ${:.0}", params.clip_size_usd);
        info!("Simulated balance: ${:.2}", self.sim_balance);
        info!(
            "Fee estimation: live API preview (entry + exit) + {}%/hr borrow",
            BORROW_FEE_HOURLY * 100.0
        );
        warn!("PAPER MODE -- no real transactions will be signed");

        let initial_price = self.flash.get_price(&self.config.flash.market).await?;
        info!("Initial price: ${:.2}", initial_price);
        self.strategy.push_price(initial_price, now_ms());

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
                pos.inner.asset,
                pos.inner.current_price,
                pnl,
                pos.total_fees(),
                pos.entry_fee,
                pos.accrued_borrow_fee
            );
        }

        let stats = self.trade_log.stats();
        info!("=== Paper Trading Final Stats ===");
        info!(
            "Trades: {} | Win rate: {:.1}%",
            stats.total_trades, stats.win_rate
        );
        info!(
            "Gross PnL: ${:.2} | Fees: ${:.2} | Net PnL: ${:.2}",
            stats.total_pnl, stats.total_fees, stats.net_pnl
        );
        info!("Simulated balance: ${:.2}", self.sim_balance);

        Ok(())
    }

    async fn tick(&mut self) -> anyhow::Result<()> {
        let market = &self.config.flash.market;
        let price = self.flash.get_price(market).await?;
        self.strategy.push_price(price, now_ms());

        let snapshot = self.strategy.snapshot();

        debug!(
            "[{}] prices={} velocity={:.4}% dir={:?} strength={:.0}",
            market,
            snapshot.price_count,
            snapshot.price_velocity_pct,
            snapshot.direction,
            snapshot.strength,
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

        let clip = self.strategy.parameters().clip_size_usd;
        let notional = clip * self.config.flash.leverage;
        if let Err(e) = self.risk.check_position_size(notional) {
            warn!("{}", e);
            return Ok(());
        }

        if self.sim_balance < clip {
            debug!(
                "Insufficient simulated balance: ${:.2} < ${:.2}",
                self.sim_balance, clip
            );
            return Ok(());
        }

        let bias = self.strategy.parameters().direction_bias.to_lowercase();
        let leverage = self.config.flash.leverage;
        let signal = self.strategy.detect_entry(snapshot);

        match signal {
            Signal::MomentumLong {
                strength,
                velocity_pct,
            } if bias != "short" => {
                self.paper_open(true, clip, leverage, current_price, strength, velocity_pct)
                    .await?;
            }
            Signal::MomentumShort {
                strength,
                velocity_pct,
            } if bias != "long" => {
                self.paper_open(false, clip, leverage, current_price, strength, velocity_pct)
                    .await?;
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
        let preview = self
            .flash
            .preview_open_position(
                &self.config.flash.input_token,
                &self.config.flash.market,
                clip_usd,
                leverage,
                trade_type,
            )
            .await?;

        let entry_fee = preview
            .entry_fee
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(clip_usd * FALLBACK_FEE_RATE);

        let entry_price = preview
            .new_entry_price
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(current_price);

        let liq_price = preview.new_liquidation_price.as_deref().unwrap_or("N/A");
        let notional = clip_usd * leverage;

        if let Some(ref err) = preview.err {
            warn!(
                "Preview returned error (proceeding with estimates): {}",
                err
            );
        }

        info!(
            ">>> [PAPER] OPENING {} ${:.0} x {:.0}x @ ${:.2} | liq=${} | entry_fee=${:.4} ({:.3}%) | strength={:.0} velocity={:.3}%",
            trade_type,
            clip_usd,
            leverage,
            entry_price,
            liq_price,
            entry_fee,
            (entry_fee / clip_usd) * 100.0,
            strength,
            velocity_pct
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

        let params = self.strategy.parameters();
        let ctx = PositionContext {
            is_long: pos.inner.is_long,
            entry_price: pos.inner.entry_price,
            current_price,
            peak_price: pos.inner.peak_price,
            hold_secs: pos.inner.hold_duration_secs(),
            max_hold_secs: params.max_hold_secs,
            take_profit_pct: params.take_profit_pct,
            stop_loss_pct: params.stop_loss_pct,
            trailing_stop_pct: params.trailing_stop_pct,
            trailing_activation_pct: params.trailing_activation_pct,
        };

        let exit_signal = self.strategy.detect_exit(snapshot, &ctx);

        match exit_signal {
            Some(Signal::ExitLong { reason } | Signal::ExitShort { reason }) => {
                self.paper_close(current_price, reason).await?;
            }
            Some(_) => {}
            None => {
                debug!(
                    "[PAPER] Holding {} {} @ ${:.2} | uPnL: ${:.2} ({:.2}%) | fees: ${:.4} (borrow: ${:.4}) | hold: {}s",
                    if pos.inner.is_long { "LONG" } else { "SHORT" },
                    pos.inner.asset,
                    current_price,
                    pos.inner.unrealized_pnl_usd(),
                    pos.inner.unrealized_pnl_pct(),
                    pos.total_fees(),
                    pos.accrued_borrow_fee,
                    pos.inner.hold_duration_secs()
                );
            }
        }

        Ok(())
    }

    async fn paper_close(&mut self, exit_price: f64, reason: ExitReason) -> anyhow::Result<()> {
        let pos = match self.position.take() {
            Some(p) => p,
            None => return Ok(()),
        };

        let gross_pnl = pos.inner.unrealized_pnl_usd();

        // Estimate exit fee from live API (with fallback)
        let exit_fee = match self
            .flash
            .preview_exit_fee(&pos.inner.position_key, pos.inner.size_usd)
            .await
        {
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
            pos.inner.asset,
            pos.inner.entry_price,
            exit_price,
            reason,
            hold_mins
        );
        info!(
            "    gross_pnl=${:.2} | entry_fee=${:.4} exit_fee=${:.4} borrow_fee=${:.4} | total_fees=${:.4} ({:.3}%) | net=${:.2}",
            gross_pnl,
            pos.entry_fee,
            exit_fee,
            pos.accrued_borrow_fee,
            total_fees,
            (total_fees / pos.inner.size_usd) * 100.0,
            net_pnl
        );

        // Update simulated balance
        self.sim_balance += net_pnl;

        self.risk
            .record_trade_result(net_pnl, total_fees, self.sim_balance);

        self.trade_log.record(TradeRecord {
            symbol: pos.inner.symbol.clone(),
            direction: if pos.inner.is_long {
                "LONG".into()
            } else {
                "SHORT".into()
            },
            entry_price: pos.inner.entry_price,
            exit_price,
            size_usd: pos.inner.size_usd,
            pnl: net_pnl,
            fees: total_fees,
            hold_secs: pos.inner.hold_duration_secs(),
            exit_reason: format!("{:?}", reason),
            timestamp: Utc::now(),
            strategy: self.strategy.name().to_string(),
            market: self.config.flash.market.clone(),
            entry_fee: pos.entry_fee,
            exit_fee,
            borrow_fee: pos.accrued_borrow_fee,
            gross_pnl,
        });

        if net_pnl < 0.0 {
            let cooldown = self.strategy.parameters().cooldown_after_loss_secs;
            self.risk.set_cooldown(cooldown);
        }

        info!("[PAPER] Simulated balance: ${:.2}", self.sim_balance);

        Ok(())
    }
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

// ===========================================================================
// MultiPaperEngine — Multi-strategy, multi-market paper trading
// ===========================================================================

/// Fallback taker fee rate if API preview fails
const MULTI_FALLBACK_FEE_RATE: f64 = 0.001; // 0.1%

/// Estimated hourly borrow/margin fee rate (configurable via config, defaults to 0.01%/hr)
const DEFAULT_BORROW_FEE_HOURLY: f64 = 0.0001;

/// Key for the position matrix: (strategy_name, market_symbol).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CellKey {
    strategy: String,
    market: String,
}

impl std::fmt::Display for CellKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.strategy, self.market)
    }
}

/// Per-cell position with full fee tracking.
#[derive(Debug, Clone)]
struct CellPosition {
    position_key: String,
    symbol: String,
    asset: String,
    is_long: bool,
    entry_price: f64,
    current_price: f64,
    peak_price: f64,
    size_usd: f64,
    leverage: f64,
    open_time: DateTime<Utc>,
    /// Entry fee captured from live API preview at open time.
    entry_fee: f64,
    /// Accumulated borrow/funding fee.
    accrued_borrow_fee: f64,
    /// Per-cell cooldown until time (if any).
    cooldown_until: Option<DateTime<Utc>>,
}

impl CellPosition {
    fn unrealized_pnl_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        if self.is_long {
            (self.current_price - self.entry_price) / self.entry_price * 100.0
        } else {
            (self.entry_price - self.current_price) / self.entry_price * 100.0
        }
    }

    fn unrealized_pnl_usd(&self) -> f64 {
        self.size_usd * self.unrealized_pnl_pct() / 100.0
    }

    fn hold_duration_secs(&self) -> u64 {
        (Utc::now() - self.open_time).num_seconds().max(0) as u64
    }

    fn update_price(&mut self, price: f64, poll_interval_secs: u64) {
        self.current_price = price;
        if self.is_long {
            if price > self.peak_price {
                self.peak_price = price;
            }
        } else {
            if self.peak_price == 0.0 || price < self.peak_price {
                self.peak_price = price;
            }
        }
        // Accrue borrow fee incrementally
        let hours_held = poll_interval_secs as f64 / 3600.0;
        self.accrued_borrow_fee += self.size_usd * DEFAULT_BORROW_FEE_HOURLY * hours_held;
    }

    fn total_fees(&self) -> f64 {
        self.entry_fee + self.accrued_borrow_fee
    }
}

/// Per-cell statistics for the summary report.
#[derive(Debug, Clone, Default, serde::Serialize)]
struct CellStats {
    strategy: String,
    market: String,
    trade_count: usize,
    win_count: usize,
    loss_count: usize,
    gross_pnl: f64,
    total_fees: f64,
    entry_fees_total: f64,
    exit_fees_total: f64,
    borrow_fees_total: f64,
    net_pnl: f64,
    /// Total fees / gross_pnl * 100 (fee ratio %)
    fee_ratio: f64,
    win_rate: f64,
    /// Sharpe-like ratio: mean return / std dev of returns
    sharpe_ratio: f64,
    max_drawdown_usd: f64,
    avg_hold_secs: f64,
    /// Individual trade PnLs for sharpe calculation
    #[serde(skip_serializing)]
    trade_pnls: Vec<f64>,
    /// Running peak balance for drawdown tracking
    #[serde(skip_serializing)]
    peak_cell_balance: f64,
    /// Running cell-level balance for drawdown
    #[serde(skip_serializing)]
    cell_balance: f64,
}

impl CellStats {
    fn record_trade(
        &mut self,
        pnl: f64,
        entry_fee: f64,
        exit_fee: f64,
        borrow_fee: f64,
        hold_secs: u64,
    ) {
        self.trade_count += 1;
        self.trade_pnls.push(pnl);
        let total_trade_fees = entry_fee + exit_fee + borrow_fee;
        self.gross_pnl += pnl + total_trade_fees; // gross = net_pnl + all fees
        self.entry_fees_total += entry_fee;
        self.exit_fees_total += exit_fee;
        self.borrow_fees_total += borrow_fee;
        self.total_fees += total_trade_fees;
        self.net_pnl += pnl;

        if pnl > 0.0 {
            self.win_count += 1;
        } else {
            self.loss_count += 1;
        }

        // Update cell balance for drawdown tracking
        self.cell_balance += pnl - entry_fee; // entry_fee already deducted from balance at open
        // Actually: at open, we deduct entry_fee from sim_balance. At close, we add net_pnl.
        // For drawdown tracking, just track cumulative net flow.
        self.cell_balance = self.net_pnl;

        if self.cell_balance > self.peak_cell_balance {
            self.peak_cell_balance = self.cell_balance;
        }
        let dd = self.peak_cell_balance - self.cell_balance;
        if dd > self.max_drawdown_usd {
            self.max_drawdown_usd = dd;
        }

        // Avg hold
        let total_hold: f64 = self.trade_pnls.len() as f64 * self.avg_hold_secs + hold_secs as f64;
        self.avg_hold_secs = total_hold / (self.trade_pnls.len() as f64 + 1.0);
        // More accurate running average
        self.avg_hold_secs = if self.trade_count == 1 {
            hold_secs as f64
        } else {
            (self.avg_hold_secs * (self.trade_count - 1) as f64 + hold_secs as f64)
                / self.trade_count as f64
        };
    }

    fn finalize(&mut self) {
        self.win_rate = if self.trade_count > 0 {
            self.win_count as f64 / self.trade_count as f64 * 100.0
        } else {
            0.0
        };
        self.fee_ratio = if self.gross_pnl.abs() > 0.0001 {
            self.total_fees / self.gross_pnl.abs() * 100.0
        } else {
            0.0
        };
        // Sharpe-like ratio: mean(pnls) / std(pnls)
        if self.trade_pnls.len() >= 2 {
            let mean: f64 = self.trade_pnls.iter().sum::<f64>() / self.trade_pnls.len() as f64;
            let variance: f64 = self
                .trade_pnls
                .iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>()
                / (self.trade_pnls.len() - 1) as f64;
            let std_dev = variance.sqrt();
            self.sharpe_ratio = if std_dev > 0.0 { mean / std_dev } else { 0.0 };
        } else {
            self.sharpe_ratio = 0.0;
        }
    }
}

/// Summary output written to `data/paper-results/summary.json`.
#[derive(Debug, serde::Serialize)]
struct PaperSummary {
    start_time: String,
    end_time: String,
    duration_secs: f64,
    starting_balance: f64,
    final_balance: f64,
    total_strategies: usize,
    total_markets: usize,
    total_cells: usize,
    total_trades: usize,
    total_net_pnl: f64,
    total_fees: f64,
    results: Vec<CellStats>,
    /// Per-strategy best market (strategy_name -> (best_market, net_pnl))
    best_market_per_strategy: HashMap<String, String>,
}

/// The multi-strategy, multi-market paper trading engine.
///
/// Maintains a position matrix: `(strategy_name, market) → Option<CellPosition>`.
/// Each cell operates independently with its own price buffer, strategy state,
/// position tracking, fee accounting, and cooldown.
pub struct MultiPaperEngine {
    config: crate::config::Config,
    flash: FlashClient,
    running: Arc<AtomicBool>,
    output_dir: String,

    /// Per-cell strategy instances (each with independent state/price buffer).
    strategies: HashMap<CellKey, Box<dyn Strategy>>,
    /// Per-cell positions.
    positions: HashMap<CellKey, CellPosition>,
    /// Per-cell statistics.
    stats: HashMap<CellKey, CellStats>,

    /// All configured strategy names (preserves order).
    strategy_names: Vec<String>,
    /// All configured market symbols (preserves order).
    markets: Vec<String>,

    /// Shared trade log for all cells.
    trade_log: TradeLog,
    /// Shared simulated balance.
    sim_balance: f64,
    starting_balance: f64,

    /// Start time for duration tracking.
    start_time: DateTime<Utc>,
    /// Last hourly status log time.
    last_hourly_log: DateTime<Utc>,

    /// Risk manager (shared across all cells for circuit breaker).
    risk: Arc<RiskManager>,
}

impl MultiPaperEngine {
    pub fn new(
        config: crate::config::Config,
        starting_balance: f64,
        strategy_names: Vec<&str>,
        markets: Vec<String>,
        output_dir: &str,
    ) -> anyhow::Result<Self> {
        let flash = FlashClient::new(&config.flash.api_url);
        let risk = Arc::new(RiskManager::new(config.risk.clone(), starting_balance));
        let trade_log = TradeLog::new("paper-trades.json");

        let strategy_names_owned: Vec<String> =
            strategy_names.iter().map(|s| s.to_string()).collect();

        // Validate all strategy names
        let available = strategy::available_strategies();
        for name in &strategy_names_owned {
            if !available.iter().any(|a| *a == name.as_str()) {
                anyhow::bail!(
                    "Unknown strategy '{}'. Available strategies: {}",
                    name,
                    available.join(", ")
                );
            }
        }

        // Create strategy instances for each cell
        let mut strategies = HashMap::new();
        for strat_name in &strategy_names_owned {
            for market in &markets {
                let key = CellKey {
                    strategy: strat_name.clone(),
                    market: market.clone(),
                };
                let sub_table = config.strategy.get_sub_table(strat_name);
                let fallback_params = config.strategy.get_params(strat_name).unwrap_or_else(|_| {
                    crate::strategy::StrategyParams {
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
                });
                let strat =
                    strategy::create_strategy_from_config(strat_name, sub_table, fallback_params)?;
                strategies.insert(key, strat);
            }
        }

        // Initialize stats for every cell
        let mut stats = HashMap::new();
        for strat_name in &strategy_names_owned {
            for market in &markets {
                let key = CellKey {
                    strategy: strat_name.clone(),
                    market: market.clone(),
                };
                stats.insert(
                    key,
                    CellStats {
                        strategy: strat_name.clone(),
                        market: market.clone(),
                        peak_cell_balance: 0.0,
                        cell_balance: 0.0,
                        ..Default::default()
                    },
                );
            }
        }

        let num_cells = strategy_names_owned.len() * markets.len();
        info!(
            "Multi-Paper Engine: {} strategies × {} markets = {} cells",
            strategy_names_owned.len(),
            markets.len(),
            num_cells
        );
        info!("Strategies: {}", strategy_names_owned.join(", "));
        info!("Markets: {}", markets.join(", "));
        info!("Simulated balance: ${:.2}", starting_balance);

        Ok(Self {
            config,
            flash,
            running: Arc::new(AtomicBool::new(true)),
            output_dir: output_dir.to_string(),
            strategies,
            positions: HashMap::new(),
            stats,
            strategy_names: strategy_names_owned,
            markets,
            trade_log,
            sim_balance: starting_balance,
            starting_balance,
            start_time: Utc::now(),
            last_hourly_log: Utc::now(),
            risk,
        })
    }

    pub fn shutdown_handle(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        info!("=== Dalgona MULTI-PAPER Trading Mode ===");
        info!(
            "Fee estimation: live API preview (entry + exit) + {}%/hr borrow",
            DEFAULT_BORROW_FEE_HOURLY * 100.0
        );
        warn!("MULTI-PAPER MODE -- no real transactions will be signed");

        // Fetch initial prices for each market
        for market in &self.markets.clone() {
            match self.flash.get_price(market).await {
                Ok(price) => {
                    info!("[{}] Initial price: ${:.2}", market, price);
                    // Push initial price to all strategy cells for this market
                    for strat_name in &self.strategy_names.clone() {
                        let key = CellKey {
                            strategy: strat_name.clone(),
                            market: market.clone(),
                        };
                        if let Some(strat) = self.strategies.get_mut(&key) {
                            strat.push_price(price, now_ms());
                        }
                    }
                }
                Err(e) => {
                    warn!("[{}] Failed to fetch initial price: {:#}", market, e);
                }
            }
        }

        loop {
            if !self.running.load(Ordering::Relaxed) {
                info!("Shutdown signal received");
                break;
            }

            if self.risk.is_halted() {
                error!("Circuit breaker active -- stopping multi-paper trading");
                break;
            }

            if let Err(e) = self.tick().await {
                error!("Tick error: {:#}", e);
                sleep(Duration::from_secs(10)).await;
                continue;
            }

            // Hourly status log
            self.maybe_hourly_log();

            sleep(self.config.poll_interval()).await;
        }

        // Graceful shutdown
        self.shutdown_report().await?;
        self.write_summary()?;

        Ok(())
    }

    async fn tick(&mut self) -> anyhow::Result<()> {
        // Fetch prices for all markets
        let mut prices: HashMap<String, f64> = HashMap::new();
        for market in &self.markets {
            match self.flash.get_price(market).await {
                Ok(price) => {
                    debug!("[{}] price=${:.2}", market, price);
                    prices.insert(market.clone(), price);
                }
                Err(e) => {
                    warn!("[{}] Price fetch error: {:#}", market, e);
                }
            }
        }

        // Process each cell
        let keys: Vec<CellKey> = self.strategies.keys().cloned().collect();
        for key in keys {
            let price = match prices.get(&key.market) {
                Some(p) => *p,
                None => continue, // Skip cells with no price this tick
            };

            // Push price to strategy
            if let Some(strat) = self.strategies.get_mut(&key) {
                strat.push_price(price, now_ms());
            }

            // Check if cell has an open position
            let has_position = self.positions.contains_key(&key);

            if has_position {
                self.manage_cell(&key, price).await?;
            } else {
                self.handle_no_position_cell(&key, price).await?;
            }
        }

        Ok(())
    }

    async fn handle_no_position_cell(
        &mut self,
        key: &CellKey,
        current_price: f64,
    ) -> anyhow::Result<()> {
        // Check per-cell cooldown (position may be a cooldown marker)
        if let Some(pos) = self.positions.get(key) {
            if pos.size_usd == 0.0 {
                // This is a cooldown marker
                if let Some(until) = pos.cooldown_until {
                    if Utc::now() < until {
                        return Ok(());
                    } else {
                        // Cooldown expired — remove the marker
                        // We need to do this after the borrow, so just note it
                    }
                }
            }
        }

        // Remove expired cooldown markers
        let should_remove_cooldown = {
            if let Some(pos) = self.positions.get(key) {
                pos.size_usd == 0.0 && pos.cooldown_until.map_or(false, |u| Utc::now() >= u)
            } else {
                false
            }
        };
        if should_remove_cooldown {
            self.positions.remove(key);
        }

        // Check if we can trade
        if let Err(e) = self.risk.check_can_trade(self.sim_balance) {
            debug!("[{}] Cannot trade: {}", key, e);
            return Ok(());
        }

        let strat = match self.strategies.get_mut(key) {
            Some(s) => s,
            None => return Ok(()),
        };
        let params = strat.parameters().clone();
        let clip = params.clip_size_usd;
        let notional = clip * self.config.flash.leverage;

        if let Err(e) = self.risk.check_position_size(notional) {
            warn!("[{}] {}", key, e);
            return Ok(());
        }

        if self.sim_balance < clip {
            debug!(
                "[{}] Insufficient simulated balance: ${:.2} < ${:.2}",
                key, self.sim_balance, clip
            );
            return Ok(());
        }

        let bias = params.direction_bias.to_lowercase();
        let snapshot = strat.snapshot();
        let signal = strat.detect_entry(&snapshot);

        match signal {
            Signal::MomentumLong {
                strength,
                velocity_pct,
            } if bias != "short" => {
                self.cell_open(
                    key,
                    true,
                    clip,
                    self.config.flash.leverage,
                    current_price,
                    strength,
                    velocity_pct,
                )
                .await?;
            }
            Signal::MomentumShort {
                strength,
                velocity_pct,
            } if bias != "long" => {
                self.cell_open(
                    key,
                    false,
                    clip,
                    self.config.flash.leverage,
                    current_price,
                    strength,
                    velocity_pct,
                )
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn cell_open(
        &mut self,
        key: &CellKey,
        is_long: bool,
        clip_usd: f64,
        leverage: f64,
        current_price: f64,
        strength: f64,
        velocity_pct: f64,
    ) -> anyhow::Result<()> {
        let trade_type = if is_long { "LONG" } else { "SHORT" };

        // Preview against live API for real entry fee
        let preview = self
            .flash
            .preview_open_position(
                &self.config.flash.input_token,
                &key.market,
                clip_usd,
                leverage,
                trade_type,
            )
            .await?;

        let entry_fee = preview
            .entry_fee
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(clip_usd * MULTI_FALLBACK_FEE_RATE);

        let entry_price = preview
            .new_entry_price
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(current_price);

        let liq_price = preview.new_liquidation_price.as_deref().unwrap_or("N/A");
        let notional = clip_usd * leverage;

        if let Some(ref err) = preview.err {
            warn!("[{}] Preview returned error: {}", key, err);
        }

        info!(
            ">>> [{}] OPENING {} ${:.0} x {:.0}x @ ${:.2} | liq=${} | entry_fee=${:.4} ({:.3}%) | strength={:.0} velocity={:.3}%",
            key,
            trade_type,
            clip_usd,
            leverage,
            entry_price,
            liq_price,
            entry_fee,
            (entry_fee / clip_usd) * 100.0,
            strength,
            velocity_pct
        );

        let pos = CellPosition {
            position_key: format!("paper-{}-{}", key, Utc::now().timestamp()),
            symbol: format!("{}-USD", key.market),
            asset: key.market.clone(),
            is_long,
            entry_price,
            current_price: entry_price,
            peak_price: entry_price,
            size_usd: notional,
            leverage,
            open_time: Utc::now(),
            entry_fee,
            accrued_borrow_fee: 0.0,
            cooldown_until: None,
        };

        self.positions.insert(key.clone(), pos);
        self.sim_balance -= entry_fee;

        // Try to estimate price impact from pool data
        let has_pool = self
            .strategies
            .get(key)
            .map(|s| s.snapshot().pool_data.is_some())
            .unwrap_or(false);
        if has_pool {
            debug!("[{}] Pool data available for price impact estimation", key);
        } else {
            debug!("[{}] Price impact not estimated (no pool data)", key);
        }

        Ok(())
    }

    async fn manage_cell(&mut self, key: &CellKey, current_price: f64) -> anyhow::Result<()> {
        // Update position price (and accrue borrow fee)
        let poll_secs = self.config.agent.poll_interval_secs;
        if let Some(pos) = self.positions.get_mut(key) {
            pos.update_price(current_price, poll_secs);
        } else {
            return Ok(());
        }

        let strat = match self.strategies.get(key) {
            Some(s) => s,
            None => return Ok(()),
        };
        let params = strat.parameters();

        // Build PositionContext for exit detection
        let ctx = {
            let pos = self.positions.get(key).unwrap();
            PositionContext {
                is_long: pos.is_long,
                entry_price: pos.entry_price,
                current_price,
                peak_price: pos.peak_price,
                hold_secs: pos.hold_duration_secs(),
                max_hold_secs: params.max_hold_secs,
                take_profit_pct: params.take_profit_pct,
                stop_loss_pct: params.stop_loss_pct,
                trailing_stop_pct: params.trailing_stop_pct,
                trailing_activation_pct: params.trailing_activation_pct,
            }
        };

        let snapshot = strat.snapshot();
        let exit_signal = strat.detect_exit(&snapshot, &ctx);

        match exit_signal {
            Some(Signal::ExitLong { reason } | Signal::ExitShort { reason }) => {
                self.cell_close(key, current_price, reason).await?;
            }
            Some(_) => {}
            None => {
                if let Some(pos) = self.positions.get(key) {
                    debug!(
                        "[{}] Holding {} {} @ ${:.2} | uPnL: ${:.2} ({:.2}%) | fees: ${:.4} (borrow: ${:.4}) | hold: {}s",
                        key,
                        if pos.is_long { "LONG" } else { "SHORT" },
                        pos.asset,
                        current_price,
                        pos.unrealized_pnl_usd(),
                        pos.unrealized_pnl_pct(),
                        pos.total_fees(),
                        pos.accrued_borrow_fee,
                        pos.hold_duration_secs()
                    );
                }
            }
        }

        Ok(())
    }

    async fn cell_close(
        &mut self,
        key: &CellKey,
        exit_price: f64,
        reason: ExitReason,
    ) -> anyhow::Result<()> {
        let pos = match self.positions.remove(key) {
            Some(p) => p,
            None => return Ok(()),
        };

        let gross_pnl = pos.unrealized_pnl_usd();

        // Estimate exit fee from live API
        let exit_fee = match self
            .flash
            .preview_exit_fee(&pos.position_key, pos.size_usd)
            .await
        {
            Ok(fee) => {
                debug!("[{}] Exit fee from API: ${:.4}", key, fee);
                fee
            }
            Err(e) => {
                debug!(
                    "[{}] Exit fee preview failed (using fallback): {:#}",
                    key, e
                );
                pos.size_usd * MULTI_FALLBACK_FEE_RATE
            }
        };

        let total_fees = pos.entry_fee + exit_fee + pos.accrued_borrow_fee;
        // Net PnL = gross_pnl - exit_fee - borrow_fee (entry_fee already deducted from balance at open)
        let net_pnl = gross_pnl - exit_fee - pos.accrued_borrow_fee;

        let hold_mins = pos.hold_duration_secs() as f64 / 60.0;

        info!(
            "<<< [{}] CLOSING {} {} | ${:.2} -> ${:.2} | reason={:?} | hold={:.1}min",
            key,
            if pos.is_long { "LONG" } else { "SHORT" },
            pos.asset,
            pos.entry_price,
            exit_price,
            reason,
            hold_mins
        );
        info!(
            "    gross_pnl=${:.2} | entry_fee=${:.4} exit_fee=${:.4} borrow_fee=${:.4} | total_fees=${:.4} ({:.3}%) | net=${:.2}",
            gross_pnl,
            pos.entry_fee,
            exit_fee,
            pos.accrued_borrow_fee,
            total_fees,
            (total_fees / pos.size_usd) * 100.0,
            net_pnl
        );

        // Update simulated balance
        self.sim_balance += net_pnl;

        // Record to shared trade log
        self.trade_log.record(TradeRecord {
            symbol: pos.symbol.clone(),
            direction: if pos.is_long {
                "LONG".into()
            } else {
                "SHORT".into()
            },
            entry_price: pos.entry_price,
            exit_price,
            size_usd: pos.size_usd,
            pnl: net_pnl,
            fees: total_fees,
            hold_secs: pos.hold_duration_secs(),
            exit_reason: format!("{:?}", reason),
            timestamp: Utc::now(),
            strategy: key.strategy.clone(),
            market: key.market.clone(),
            entry_fee: pos.entry_fee,
            exit_fee,
            borrow_fee: pos.accrued_borrow_fee,
            gross_pnl,
        });

        // Update per-cell stats
        if let Some(cell_stats) = self.stats.get_mut(key) {
            cell_stats.record_trade(
                net_pnl,
                pos.entry_fee,
                exit_fee,
                pos.accrued_borrow_fee,
                pos.hold_duration_secs(),
            );
        }

        // Update risk manager
        self.risk
            .record_trade_result(net_pnl, total_fees, self.sim_balance);

        // Per-cell cooldown after loss
        if net_pnl < 0.0 {
            let cooldown_secs = self
                .strategies
                .get(key)
                .map(|s| s.parameters().cooldown_after_loss_secs)
                .unwrap_or(300);
            // Re-insert a "cooldown marker" position
            let cooldown_pos = CellPosition {
                position_key: format!("cooldown-{}", key),
                symbol: String::new(),
                asset: String::new(),
                is_long: false,
                entry_price: 0.0,
                current_price: 0.0,
                peak_price: 0.0,
                size_usd: 0.0,
                leverage: 0.0,
                open_time: Utc::now(),
                entry_fee: 0.0,
                accrued_borrow_fee: 0.0,
                cooldown_until: Some(Utc::now() + chrono::Duration::seconds(cooldown_secs as i64)),
            };
            self.positions.insert(key.clone(), cooldown_pos);
        }

        info!("[{}] Simulated balance: ${:.2}", key, self.sim_balance);

        Ok(())
    }

    /// Emit hourly status log with per-strategy breakdown.
    fn maybe_hourly_log(&mut self) {
        let now = Utc::now();
        let elapsed = (now - self.last_hourly_log).num_seconds();
        if elapsed < 3600 {
            return;
        }
        self.last_hourly_log = now;

        let total_elapsed = (now - self.start_time).num_seconds();
        let hours = total_elapsed / 3600;
        let mins = (total_elapsed % 3600) / 60;

        let open_count = self
            .positions
            .values()
            .filter(|p| p.size_usd > 0.0 && p.cooldown_until.is_none())
            .count();

        info!("=== Hourly Status ({:02}h{:02}m elapsed) ===", hours, mins);
        info!(
            "Balance: ${:.2} | Open positions: {} | Total trades: {}",
            self.sim_balance,
            open_count,
            self.trade_log.stats().total_trades
        );

        // Per-strategy breakdown
        for strat_name in &self.strategy_names {
            let mut strat_trades = 0usize;
            let mut strat_pnl = 0.0f64;
            let mut strat_fees = 0.0f64;

            for market in &self.markets {
                let key = CellKey {
                    strategy: strat_name.clone(),
                    market: market.clone(),
                };
                if let Some(cell_stats) = self.stats.get(&key) {
                    strat_trades += cell_stats.trade_count;
                    strat_pnl += cell_stats.net_pnl;
                    strat_fees += cell_stats.total_fees;
                }
            }

            info!(
                "  [{}] trades={} | net_pnl=${:.2} | fees=${:.2}",
                strat_name, strat_trades, strat_pnl, strat_fees
            );
        }
    }

    /// Print shutdown report with open positions and final stats.
    async fn shutdown_report(&self) -> anyhow::Result<()> {
        // Report open positions
        for (key, pos) in &self.positions {
            if pos.size_usd == 0.0 {
                continue; // Skip cooldown markers
            }
            let pnl = pos.unrealized_pnl_usd();
            info!(
                "[{}] Open position at shutdown: {} {} @ ${:.2} | uPnL: ${:.2} | fees: ${:.4} (entry=${:.4} borrow=${:.4})",
                key,
                if pos.is_long { "LONG" } else { "SHORT" },
                pos.asset,
                pos.current_price,
                pnl,
                pos.total_fees(),
                pos.entry_fee,
                pos.accrued_borrow_fee
            );
        }

        // Final stats
        let stats = self.trade_log.stats();
        info!("=== Multi-Paper Trading Final Stats ===");
        info!(
            "Trades: {} | Win rate: {:.1}%",
            stats.total_trades, stats.win_rate
        );
        info!(
            "Gross PnL: ${:.2} | Fees: ${:.2} | Net PnL: ${:.2}",
            stats.total_pnl, stats.total_fees, stats.net_pnl
        );
        info!("Simulated balance: ${:.2}", self.sim_balance);
        info!(
            "Duration: {:.1} minutes",
            (Utc::now() - self.start_time).num_seconds() as f64 / 60.0
        );

        // Per-strategy comparison table sorted by net PnL
        let mut rows: Vec<(&str, &str, f64, f64, f64, usize)> = Vec::new();
        for strat_name in &self.strategy_names {
            for market in &self.markets {
                let key = CellKey {
                    strategy: strat_name.clone(),
                    market: market.clone(),
                };
                if let Some(cell_stats) = self.stats.get(&key) {
                    rows.push((
                        strat_name.as_str(),
                        market.as_str(),
                        cell_stats.net_pnl,
                        cell_stats.total_fees,
                        cell_stats.gross_pnl,
                        cell_stats.trade_count,
                    ));
                }
            }
        }
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        info!("=== Strategy × Market Comparison (sorted by net PnL) ===");
        info!(
            "{:<25} {:>10} {:>10} {:>10} {:>6}",
            "Strategy:Market", "Gross$", "Fees$", "Net$", "Trades"
        );
        for (strat, market, net_pnl, fees, gross, trades) in &rows {
            info!(
                "{:<25} {:>10.2} {:>10.2} {:>10.2} {:>6}",
                format!("{}:{}", strat, market),
                gross,
                fees,
                net_pnl,
                trades
            );
        }

        Ok(())
    }

    /// Write summary.json to the output directory.
    fn write_summary(&mut self) -> anyhow::Result<()> {
        // Finalize all cell stats
        for cell_stats in self.stats.values_mut() {
            cell_stats.finalize();
        }

        // Compute best market per strategy
        let mut best_market_per_strategy: HashMap<String, String> = HashMap::new();
        for strat_name in &self.strategy_names {
            let mut best_market = String::new();
            let mut best_pnl = f64::NEG_INFINITY;
            for market in &self.markets {
                let key = CellKey {
                    strategy: strat_name.clone(),
                    market: market.clone(),
                };
                if let Some(cell_stats) = self.stats.get(&key) {
                    if cell_stats.net_pnl > best_pnl {
                        best_pnl = cell_stats.net_pnl;
                        best_market = market.clone();
                    }
                }
            }
            if !best_market.is_empty() {
                best_market_per_strategy.insert(strat_name.clone(), best_market);
            }
        }

        // Build results sorted by net PnL
        let mut results: Vec<CellStats> = self.stats.values().cloned().collect();
        results.sort_by(|a, b| {
            b.net_pnl
                .partial_cmp(&a.net_pnl)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let end_time = Utc::now();
        let duration_secs = (end_time - self.start_time).num_seconds() as f64;

        let total_trades: usize = results.iter().map(|r| r.trade_count).sum();
        let total_net_pnl: f64 = results.iter().map(|r| r.net_pnl).sum();
        let total_fees: f64 = results.iter().map(|r| r.total_fees).sum();

        let summary = PaperSummary {
            start_time: self.start_time.to_rfc3339(),
            end_time: end_time.to_rfc3339(),
            duration_secs,
            starting_balance: self.starting_balance,
            final_balance: self.sim_balance,
            total_strategies: self.strategy_names.len(),
            total_markets: self.markets.len(),
            total_cells: self.strategy_names.len() * self.markets.len(),
            total_trades,
            total_net_pnl,
            total_fees,
            results,
            best_market_per_strategy,
        };

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.output_dir)?;

        // Atomic write: write to .tmp then rename
        let output_path = format!("{}/summary.json", self.output_dir);
        let tmp_path = format!("{}.tmp", output_path);
        let json = serde_json::to_string_pretty(&summary)?;
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &output_path)?;

        info!("Summary written to {}", output_path);

        Ok(())
    }
}

#[cfg(test)]
mod multi_tests {
    use super::*;

    #[test]
    fn test_cell_key_equality_and_hash() {
        let k1 = CellKey {
            strategy: "momentum-scalper".to_string(),
            market: "SOL".to_string(),
        };
        let k2 = CellKey {
            strategy: "momentum-scalper".to_string(),
            market: "SOL".to_string(),
        };
        let k3 = CellKey {
            strategy: "lp-consumption".to_string(),
            market: "SOL".to_string(),
        };
        let k4 = CellKey {
            strategy: "momentum-scalper".to_string(),
            market: "BTC".to_string(),
        };

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k1, k4);

        let mut map = HashMap::new();
        map.insert(k1.clone(), 1);
        assert_eq!(map.get(&k2), Some(&1));
        assert_eq!(map.get(&k3), None);
        assert_eq!(map.get(&k4), None);
    }

    #[test]
    fn test_cell_key_display() {
        let key = CellKey {
            strategy: "momentum-scalper".to_string(),
            market: "SOL".to_string(),
        };
        assert_eq!(format!("{}", key), "momentum-scalper:SOL");
    }

    #[test]
    fn test_cell_position_update_price_long() {
        let mut pos = CellPosition {
            position_key: "test".to_string(),
            symbol: "SOL-USD".to_string(),
            asset: "SOL".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 100.0,
            peak_price: 100.0,
            size_usd: 1000.0,
            leverage: 10.0,
            open_time: Utc::now(),
            entry_fee: 0.1,
            accrued_borrow_fee: 0.0,
            cooldown_until: None,
        };

        pos.update_price(105.0, 5);
        assert_eq!(pos.current_price, 105.0);
        assert_eq!(pos.peak_price, 105.0);
        assert!(pos.accrued_borrow_fee > 0.0);

        pos.update_price(102.0, 5);
        assert_eq!(pos.current_price, 102.0);
        assert_eq!(pos.peak_price, 105.0);
    }

    #[test]
    fn test_cell_position_update_price_short() {
        let mut pos = CellPosition {
            position_key: "test".to_string(),
            symbol: "BTC-USD".to_string(),
            asset: "BTC".to_string(),
            is_long: false,
            entry_price: 50000.0,
            current_price: 50000.0,
            peak_price: 50000.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time: Utc::now(),
            entry_fee: 0.5,
            accrued_borrow_fee: 0.0,
            cooldown_until: None,
        };

        pos.update_price(49000.0, 5);
        assert_eq!(pos.current_price, 49000.0);
        assert_eq!(pos.peak_price, 49000.0);

        pos.update_price(49500.0, 5);
        assert_eq!(pos.current_price, 49500.0);
        assert_eq!(pos.peak_price, 49000.0);
    }

    #[test]
    fn test_cell_position_pnl_long() {
        let pos = CellPosition {
            position_key: "test".to_string(),
            symbol: "SOL-USD".to_string(),
            asset: "SOL".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 102.0,
            peak_price: 103.0,
            size_usd: 1000.0,
            leverage: 10.0,
            open_time: Utc::now(),
            entry_fee: 0.1,
            accrued_borrow_fee: 0.01,
            cooldown_until: None,
        };

        let pnl_pct = pos.unrealized_pnl_pct();
        let pnl_usd = pos.unrealized_pnl_usd();
        assert!((pnl_pct - 2.0).abs() < 0.01);
        assert!((pnl_usd - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_cell_position_pnl_short() {
        let pos = CellPosition {
            position_key: "test".to_string(),
            symbol: "SOL-USD".to_string(),
            asset: "SOL".to_string(),
            is_long: false,
            entry_price: 100.0,
            current_price: 98.0,
            peak_price: 97.0,
            size_usd: 1000.0,
            leverage: 10.0,
            open_time: Utc::now(),
            entry_fee: 0.1,
            accrued_borrow_fee: 0.01,
            cooldown_until: None,
        };

        let pnl_pct = pos.unrealized_pnl_pct();
        let pnl_usd = pos.unrealized_pnl_usd();
        assert!((pnl_pct - 2.0).abs() < 0.01);
        assert!((pnl_usd - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_borrow_fee_accrual() {
        let mut pos = CellPosition {
            position_key: "test".to_string(),
            symbol: "SOL-USD".to_string(),
            asset: "SOL".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 100.0,
            peak_price: 100.0,
            size_usd: 1000.0,
            leverage: 10.0,
            open_time: Utc::now(),
            entry_fee: 0.0,
            accrued_borrow_fee: 0.0,
            cooldown_until: None,
        };

        // Simulate 720 ticks of 5 seconds = 1 hour
        for _ in 0..720 {
            pos.update_price(100.0, 5);
        }

        let expected_borrow = 1000.0 * DEFAULT_BORROW_FEE_HOURLY * 1.0; // $0.10
        assert!(
            (pos.accrued_borrow_fee - expected_borrow).abs() < 0.005,
            "accrued_borrow_fee should be ~${:.4}, got ${:.4}",
            expected_borrow,
            pos.accrued_borrow_fee
        );
    }

    #[test]
    fn test_cell_stats_record_trade() {
        let mut stats = CellStats {
            strategy: "momentum-scalper".to_string(),
            market: "SOL".to_string(),
            ..Default::default()
        };

        stats.record_trade(5.0, 0.1, 0.1, 0.01, 300);
        assert_eq!(stats.trade_count, 1);
        assert_eq!(stats.win_count, 1);
        assert_eq!(stats.loss_count, 0);
        assert!((stats.net_pnl - 5.0).abs() < 0.01);

        stats.record_trade(-3.0, 0.1, 0.1, 0.01, 120);
        assert_eq!(stats.trade_count, 2);
        assert_eq!(stats.win_count, 1);
        assert_eq!(stats.loss_count, 1);
        assert!((stats.net_pnl - 2.0).abs() < 0.01);

        stats.finalize();
        assert!((stats.win_rate - 50.0).abs() < 0.01);
        assert!(stats.sharpe_ratio != 0.0);
    }

    #[test]
    fn test_cell_stats_zero_trades() {
        let mut stats = CellStats::default();
        stats.finalize();
        assert_eq!(stats.trade_count, 0);
        assert_eq!(stats.win_rate, 0.0);
        assert_eq!(stats.sharpe_ratio, 0.0);
        assert_eq!(stats.fee_ratio, 0.0);
    }

    #[test]
    fn test_cell_stats_fee_ratio() {
        let mut stats = CellStats {
            strategy: "test".to_string(),
            market: "SOL".to_string(),
            ..Default::default()
        };
        stats.record_trade(10.0, 0.5, 0.5, 0.1, 100);
        stats.finalize();

        assert!(
            stats.fee_ratio > 0.0,
            "fee_ratio should be positive, got {}",
            stats.fee_ratio
        );
        assert!(
            stats.fee_ratio < 20.0,
            "fee_ratio should be reasonable, got {}",
            stats.fee_ratio
        );
    }

    #[test]
    fn test_cell_stats_max_drawdown() {
        let mut stats = CellStats {
            strategy: "test".to_string(),
            market: "SOL".to_string(),
            ..Default::default()
        };

        stats.record_trade(5.0, 0.1, 0.1, 0.01, 100);
        stats.record_trade(-10.0, 0.1, 0.1, 0.01, 100);
        assert!(stats.max_drawdown_usd >= 0.0);
    }

    #[test]
    fn test_trade_record_new_fields() {
        let record = TradeRecord {
            symbol: "SOL-USD".to_string(),
            direction: "LONG".to_string(),
            entry_price: 100.0,
            exit_price: 102.0,
            size_usd: 1000.0,
            pnl: 18.79,
            fees: 1.21,
            hold_secs: 300,
            exit_reason: "TakeProfit".to_string(),
            timestamp: Utc::now(),
            strategy: "momentum-scalper".to_string(),
            market: "SOL".to_string(),
            entry_fee: 0.1,
            exit_fee: 0.1,
            borrow_fee: 0.01,
            gross_pnl: 20.0,
        };

        assert_eq!(record.strategy, "momentum-scalper");
        assert_eq!(record.market, "SOL");
        assert!((record.gross_pnl - 20.0).abs() < 0.01);
        assert!((record.entry_fee - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_summary_serialization() {
        let summary = PaperSummary {
            start_time: "2025-01-01T00:00:00Z".to_string(),
            end_time: "2025-01-02T00:00:00Z".to_string(),
            duration_secs: 86400.0,
            starting_balance: 1000.0,
            final_balance: 1050.0,
            total_strategies: 2,
            total_markets: 3,
            total_cells: 6,
            total_trades: 50,
            total_net_pnl: 50.0,
            total_fees: 12.5,
            results: vec![CellStats {
                strategy: "momentum-scalper".to_string(),
                market: "SOL".to_string(),
                trade_count: 10,
                win_count: 6,
                loss_count: 4,
                net_pnl: 25.0,
                total_fees: 5.0,
                ..Default::default()
            }],
            best_market_per_strategy: {
                let mut m = HashMap::new();
                m.insert("momentum-scalper".to_string(), "SOL".to_string());
                m
            },
        };

        let json = serde_json::to_string_pretty(&summary).unwrap();
        assert!(json.contains("\"strategy\": \"momentum-scalper\""));
        assert!(json.contains("\"market\": \"SOL\""));
        assert!(json.contains("\"net_pnl\""));
        assert!(json.contains("\"sharpe_ratio\""));
        assert!(json.contains("\"max_drawdown_usd\""));
        assert!(json.contains("\"fee_ratio\""));
        assert!(json.contains("\"avg_hold_secs\""));
        assert!(json.contains("\"best_market_per_strategy\""));
        assert!(json.contains("\"duration_secs\": 86400.0"));
    }

    #[test]
    fn test_position_matrix_isolation() {
        let mut positions = HashMap::new();

        let key1 = CellKey {
            strategy: "momentum-scalper".to_string(),
            market: "SOL".to_string(),
        };
        let key2 = CellKey {
            strategy: "momentum-scalper".to_string(),
            market: "BTC".to_string(),
        };

        let pos1 = CellPosition {
            position_key: "p1".to_string(),
            symbol: "SOL-USD".to_string(),
            asset: "SOL".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 102.0,
            peak_price: 103.0,
            size_usd: 1000.0,
            leverage: 10.0,
            open_time: Utc::now(),
            entry_fee: 0.1,
            accrued_borrow_fee: 0.01,
            cooldown_until: None,
        };

        let pos2 = CellPosition {
            position_key: "p2".to_string(),
            symbol: "BTC-USD".to_string(),
            asset: "BTC".to_string(),
            is_long: false,
            entry_price: 50000.0,
            current_price: 49000.0,
            peak_price: 49000.0,
            size_usd: 500.0,
            leverage: 5.0,
            open_time: Utc::now(),
            entry_fee: 0.05,
            accrued_borrow_fee: 0.005,
            cooldown_until: None,
        };

        positions.insert(key1.clone(), pos1);
        positions.insert(key2.clone(), pos2);

        assert!(positions.contains_key(&key1));
        assert!(positions.contains_key(&key2));

        positions.get_mut(&key1).unwrap().current_price = 105.0;
        assert_eq!(positions.get(&key2).unwrap().current_price, 49000.0);
    }
}
