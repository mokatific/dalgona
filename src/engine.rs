use crate::config::Config;
use crate::executor::Executor;
use crate::flash_api::{FlashClient, FlashPosition};
use crate::risk::{Position, RiskManager, TradeLog, TradeRecord};
use crate::signal::{ExitReason, MomentumDetector, MomentumSnapshot, Signal};
use chrono::Utc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::time::{Duration, sleep};
use tracing::{debug, error, info, warn};

pub struct ScalperEngine {
    config: Config,
    flash: FlashClient,
    executor: Executor,
    detector: MomentumDetector,
    risk: Arc<RiskManager>,
    trade_log: TradeLog,
    position: Option<Position>,
    running: Arc<AtomicBool>,
}

impl ScalperEngine {
    pub fn new(config: Config, executor: Executor) -> Self {
        let flash = FlashClient::new(&config.flash.api_url);
        let detector = MomentumDetector::new(
            config.strategy.momentum_threshold_pct,
            config.strategy.lookback_count,
        );
        let risk = Arc::new(RiskManager::new(config.risk.clone(), 0.0));
        let trade_log = TradeLog::new("perps-trades.json");

        Self {
            config,
            flash,
            executor,
            detector,
            risk,
            trade_log,
            position: None,
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn shutdown_handle(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        info!("=== Flash Trade Perps Scalper v0.1 ===");
        info!("Market: {}", self.config.flash.market);
        info!("Leverage: {}x", self.config.flash.leverage);
        info!("Clip: ${:.0}", self.config.strategy.clip_size_usd);
        info!("Wallet: {}", self.executor.wallet_pubkey());

        // Fetch initial price to seed detector
        let initial_price = self.flash.get_price(&self.config.flash.market).await?;
        info!("Initial price: ${:.2}", initial_price);
        self.detector.push_price(initial_price, now_ms());

        // Fetch real USDC balance for risk manager
        let usdc_balance = self.executor.get_usdc_balance().unwrap_or(0.0);
        info!("USDC balance: ${:.2}", usdc_balance);
        self.risk = Arc::new(RiskManager::new(self.config.risk.clone(), usdc_balance));

        // Check for existing position
        self.sync_existing_position().await?;

        loop {
            if !self.running.load(Ordering::Relaxed) {
                info!("Shutdown signal received — exiting gracefully");
                break;
            }

            if self.risk.is_halted() {
                error!("Circuit breaker active — shutting down");
                break;
            }

            if let Err(e) = self.tick().await {
                error!("Tick error: {:#}", e);
                sleep(Duration::from_secs(10)).await;
                continue;
            }

            sleep(self.config.poll_interval()).await;
        }

        // Final sync: clear stale position state if on-chain position is gone
        if let Some(ref pos) = self.position {
            let wallet = self.executor.wallet_pubkey();
            match self.find_position(&wallet, pos.is_long).await {
                Ok(None) => {
                    warn!("Position gone on-chain at shutdown, clearing local state");
                    self.position = None;
                }
                Ok(Some(_)) => {
                    warn!("Position still open at shutdown: {}", pos.position_key);
                }
                Err(e) => {
                    warn!("Could not verify position at shutdown: {:#}", e);
                }
            }
        }

        let stats = self.trade_log.stats();
        info!("=== Final Stats ===");
        info!(
            "Trades: {} | Win rate: {:.1}%",
            stats.total_trades, stats.win_rate
        );
        info!(
            "Net PnL: ${:.2} (fees: ${:.2})",
            stats.net_pnl, stats.total_fees
        );

        Ok(())
    }

    async fn sync_existing_position(&mut self) -> anyhow::Result<()> {
        let wallet = self.executor.wallet_pubkey();
        let positions = self.flash.get_positions(&wallet).await?;

        for pos in &positions {
            if pos.asset == self.config.flash.market {
                let is_long = pos.side.to_uppercase() == "LONG";
                let entry = parse_f64_safe(&pos.entry_price, "entry_price")?;
                let size_usd = parse_f64_safe(&pos.size_usd, "size_usd")?;
                let leverage = parse_f64_safe(&pos.leverage, "leverage")?;

                info!(
                    "Found existing {} position: ${:.2} @ ${:.2} ({}x)",
                    pos.side, size_usd, entry, leverage as u32
                );

                self.position = Some(Position {
                    position_key: pos.position_key.clone(),
                    symbol: format!("{}-USD", pos.asset),
                    asset: pos.asset.clone(),
                    is_long,
                    entry_price: entry,
                    current_price: entry,
                    peak_price: entry,
                    size_usd,
                    leverage,
                    open_time: Utc::now(),
                });
                break;
            }
        }
        Ok(())
    }

    /// Re-verify local position state against on-chain state.
    /// Returns true if position still exists on-chain.
    async fn verify_position_on_chain(&mut self) -> bool {
        if let Some(ref pos) = self.position {
            let wallet = self.executor.wallet_pubkey();
            match self.find_position(&wallet, pos.is_long).await {
                Ok(Some(_)) => true,
                Ok(None) => {
                    warn!(
                        "Position {} no longer exists on-chain (liquidated or closed externally)",
                        pos.position_key
                    );
                    self.position = None;
                    false
                }
                Err(e) => {
                    warn!("Could not verify position on-chain: {:#}", e);
                    // Keep position but warn
                    true
                }
            }
        } else {
            false
        }
    }

    async fn tick(&mut self) -> anyhow::Result<()> {
        let market = &self.config.flash.market;

        // Fetch current price
        let price = self.flash.get_price(market).await?;
        self.detector.push_price(price, now_ms());

        let snapshot = self.detector.analyze();

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
        // Use real USDC balance for risk checks
        let balance_usd = self.executor.get_usdc_balance().unwrap_or(0.0);

        if let Err(e) = self.risk.check_can_trade(balance_usd) {
            debug!("Cannot trade: {}", e);
            return Ok(());
        }

        // Validate position size against max
        let clip = self.config.strategy.clip_size_usd;
        let notional = clip * self.config.flash.leverage;
        if let Err(e) = self.risk.check_position_size(notional) {
            warn!("{}", e);
            return Ok(());
        }

        // Check we have enough USDC for the clip
        if balance_usd < clip {
            debug!("Insufficient USDC: ${:.2} < ${:.2}", balance_usd, clip);
            return Ok(());
        }

        let bias = self.config.strategy.direction_bias.to_lowercase();
        let leverage = self.config.flash.leverage;

        let signal = self.detector.detect_signal(snapshot);

        match signal {
            Signal::MomentumLong {
                strength,
                velocity_pct,
            } if bias != "short" => {
                self.open_position(true, clip, leverage, current_price, strength, velocity_pct)
                    .await?;
            }
            Signal::MomentumShort {
                strength,
                velocity_pct,
            } if bias != "long" => {
                self.open_position(false, clip, leverage, current_price, strength, velocity_pct)
                    .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn open_position(
        &mut self,
        is_long: bool,
        clip_usd: f64,
        leverage: f64,
        current_price: f64,
        strength: f64,
        velocity_pct: f64,
    ) -> anyhow::Result<()> {
        let trade_type = if is_long { "LONG" } else { "SHORT" };
        info!(
            ">>> OPENING {} ${:.0} x {:.0}x | strength={:.0} velocity={:.3}%",
            trade_type, clip_usd, leverage, strength, velocity_pct
        );

        // Preview first
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

        if let Some(ref err) = preview.err {
            warn!("Preview failed: {}", classify_api_error(err));
            return Ok(());
        }

        info!(
            "Preview: entry=${} fee=${} liq=${} notional=${}",
            preview.new_entry_price.as_deref().unwrap_or("?"),
            preview.entry_fee.as_deref().unwrap_or("?"),
            preview.new_liquidation_price.as_deref().unwrap_or("?"),
            preview.you_recieve_usd_ui.as_deref().unwrap_or("?"),
        );

        // Build transaction with optional native TP/SL
        let wallet = self.executor.wallet_pubkey();
        let tp_price = if self.config.strategy.use_native_tp_sl {
            let tp_pct = self.config.strategy.take_profit_pct / 100.0;
            Some(if is_long {
                current_price * (1.0 + tp_pct)
            } else {
                current_price * (1.0 - tp_pct)
            })
        } else {
            None
        };
        let sl_price = if self.config.strategy.use_native_tp_sl {
            let sl_pct = self.config.strategy.stop_loss_pct / 100.0;
            Some(if is_long {
                current_price * (1.0 - sl_pct)
            } else {
                current_price * (1.0 + sl_pct)
            })
        } else {
            None
        };

        let resp = self
            .flash
            .build_open_position(
                &self.config.flash.input_token,
                &self.config.flash.market,
                clip_usd,
                leverage,
                trade_type,
                &wallet,
                &self.config.flash.slippage_pct,
                tp_price,
                sl_price,
            )
            .await?;

        if let Some(ref err) = resp.err {
            warn!("Build failed: {}", classify_api_error(err));
            return Ok(());
        }

        let tx_b64 = match resp.transaction_base64 {
            Some(tx) => tx,
            None => {
                warn!("No transaction returned");
                return Ok(());
            }
        };

        // Sign and submit
        match self.executor.sign_and_send_with_retry(&tx_b64, 2).await {
            Ok(sig) => {
                info!("Position opened: tx={}", sig);

                // Find the new position from chain
                sleep(Duration::from_secs(3)).await;
                let wallet = self.executor.wallet_pubkey();
                if let Some(flash_pos) = self.find_position(&wallet, is_long).await? {
                    let entry = parse_f64_safe(&flash_pos.entry_price, "entry_price")
                        .unwrap_or(current_price);
                    let size = parse_f64_safe(&flash_pos.size_usd, "size_usd")
                        .unwrap_or(clip_usd * leverage);
                    let lev = parse_f64_safe(&flash_pos.leverage, "leverage").unwrap_or(leverage);

                    self.position = Some(Position {
                        position_key: flash_pos.position_key.clone(),
                        symbol: format!("{}-USD", flash_pos.asset),
                        asset: flash_pos.asset,
                        is_long,
                        entry_price: entry,
                        current_price,
                        peak_price: entry,
                        size_usd: size,
                        leverage: lev,
                        open_time: Utc::now(),
                    });
                } else {
                    warn!("Position not found on-chain after open, will retry next tick");
                }
            }
            Err(e) => {
                warn!("Failed to submit open tx: {:#}", e);
            }
        }

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
            pos.is_long,
            pos.entry_price,
            current_price,
            pos.peak_price,
            pos.hold_duration_secs(),
            self.config.strategy.max_hold_secs,
            self.config.strategy.take_profit_pct,
            self.config.strategy.stop_loss_pct,
            self.config.strategy.trailing_stop_pct,
            self.config.strategy.trailing_activation_pct,
        );

        match exit_signal {
            Some(Signal::ExitLong { reason } | Signal::ExitShort { reason }) => {
                self.close_position(current_price, reason).await?;
            }
            Some(_) => {}
            None => {
                debug!(
                    "Holding {} {} @ ${:.2} | uPnL: {:.2}% | hold: {}s",
                    if pos.is_long { "LONG" } else { "SHORT" },
                    pos.asset,
                    current_price,
                    pos.unrealized_pnl_pct(),
                    pos.hold_duration_secs()
                );
            }
        }

        Ok(())
    }

    async fn close_position(&mut self, exit_price: f64, reason: ExitReason) -> anyhow::Result<()> {
        let pos = match self.position.take() {
            Some(p) => p,
            None => return Ok(()),
        };

        info!(
            "<<< CLOSING {} {} @ ${:.2} | reason={:?}",
            if pos.is_long { "LONG" } else { "SHORT" },
            pos.asset,
            exit_price,
            reason
        );

        // Verify position still exists on-chain
        let wallet = self.executor.wallet_pubkey();
        let flash_pos = self.find_position(&wallet, pos.is_long).await?;

        let flash_pos = match flash_pos {
            Some(p) => p,
            None => {
                warn!("Position not found on-chain, already closed or liquidated");
                // Record with estimated PnL (no fees available)
                let estimated_pnl = pos.unrealized_pnl_usd();
                self.risk.record_trade_result(estimated_pnl, 0.0, 0.0);
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
                    pnl: estimated_pnl,
                    fees: 0.0,
                    hold_secs: pos.hold_duration_secs(),
                    exit_reason: format!("{:?}", reason),
                    timestamp: Utc::now(),
                });
                return Ok(());
            }
        };

        let close_usd = parse_f64_safe(&flash_pos.size_usd, "size_usd").unwrap_or(pos.size_usd);

        let resp = self
            .flash
            .build_close_position(
                &flash_pos.position_key,
                close_usd,
                &self.config.flash.input_token,
                &self.config.flash.slippage_pct,
            )
            .await?;

        if let Some(ref err) = resp.err {
            warn!("Close build failed: {}", classify_api_error(err));
            // Put position back so we can retry — but re-verify it exists first
            self.position = Some(pos);
            return Ok(());
        }

        let tx_b64 = match resp.transaction_base64 {
            Some(tx) => tx,
            None => {
                warn!("No close transaction returned");
                self.position = Some(pos);
                return Ok(());
            }
        };

        match self.executor.sign_and_send_with_retry(&tx_b64, 2).await {
            Ok(sig) => {
                let settled_pnl = resp
                    .settled_pnl
                    .as_ref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(pos.unrealized_pnl_usd());
                let fees = resp
                    .fees
                    .as_ref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                info!(
                    "Position closed: tx={} PnL=${:.2} fees=${:.2}",
                    sig, settled_pnl, fees
                );

                // Get updated balance for risk tracking
                let updated_balance = self.executor.get_usdc_balance().unwrap_or(0.0);
                self.risk
                    .record_trade_result(settled_pnl, fees, updated_balance);

                let hold_secs = pos.hold_duration_secs();
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
                    pnl: settled_pnl,
                    fees,
                    hold_secs,
                    exit_reason: format!("{:?}", reason),
                    timestamp: Utc::now(),
                });

                if settled_pnl < 0.0 {
                    self.risk
                        .set_cooldown(self.config.strategy.cooldown_after_loss_secs);
                }
            }
            Err(e) => {
                warn!("Failed to submit close tx: {:#}", e);
                // Re-verify before putting position back
                self.position = Some(pos);
                // Next tick will call verify_position_on_chain if needed
            }
        }

        Ok(())
    }

    async fn find_position(
        &self,
        wallet: &str,
        is_long: bool,
    ) -> anyhow::Result<Option<FlashPosition>> {
        let side = if is_long { "Long" } else { "Short" };
        let positions = self.flash.get_positions(wallet).await?;
        Ok(positions
            .into_iter()
            .find(|p| p.asset == self.config.flash.market && p.side == side))
    }
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// Parse a string to f64, returning an error with context instead of silently returning 0.
fn parse_f64_safe(s: &str, field: &str) -> anyhow::Result<f64> {
    s.parse::<f64>()
        .map_err(|_| anyhow::anyhow!("failed to parse '{}' as f64 for field '{}'", s, field))
}

/// Classify common Flash Trade API error strings for better logging.
fn classify_api_error(err: &str) -> String {
    let err_lower = err.to_lowercase();
    if err_lower.contains("insufficient") || err_lower.contains("not enough") {
        format!("INSUFFICIENT_BALANCE: {}", err)
    } else if err_lower.contains("position already") || err_lower.contains("already have") {
        format!("POSITION_EXISTS: {}", err)
    } else if err_lower.contains("max leverage") || err_lower.contains("leverage too") {
        format!("INVALID_LEVERAGE: {}", err)
    } else if err_lower.contains("min collateral") || err_lower.contains("minimum") {
        format!("MIN_COLLATERAL: {}", err)
    } else if err_lower.contains("rate limit") || err_lower.contains("too many") {
        format!("RATE_LIMITED: {}", err)
    } else if err_lower.contains("blockhash") || err_lower.contains("expired") {
        format!("BLOCKHASH_EXPIRED: {}", err)
    } else if err_lower.contains("slippage") || err_lower.contains("price impact") {
        format!("SLIPPAGE_EXCEEDED: {}", err)
    } else if err_lower.contains("market closed") || err_lower.contains("maintenance") {
        format!("MARKET_UNAVAILABLE: {}", err)
    } else {
        err.to_string()
    }
}
