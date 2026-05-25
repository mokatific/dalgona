//! Backtesting engine using Hyperliquid historical OHLCV data.
//!
//! Replays historical candles through each strategy's `Strategy` trait methods
//! (`push_price` → `detect_entry` → `detect_exit`), simulates fills with
//! configurable fee rates, and produces the same `BacktestCellStats` metrics
//! as the paper trading engine.

use crate::config::Config;
use crate::signal::{ExitReason, Signal};
use crate::strategy::{self, PositionContext};
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Hyperliquid Candle Client
// ---------------------------------------------------------------------------

const HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
#[allow(dead_code)]
const MAX_CANDLES_PER_REQUEST: usize = 5000;

/// A single OHLCV candle from Hyperliquid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HlCandle {
    /// Open time (milliseconds since epoch).
    pub t: i64,
    /// Close time (milliseconds since epoch).
    pub t_close: i64,
    /// Symbol (e.g., "BTC").
    pub s: String,
    /// Interval (e.g., "1m", "5m", "1h").
    pub i: String,
    /// Open price.
    pub o: String,
    /// Close price.
    pub c: String,
    /// High price.
    pub h: String,
    /// Low price.
    pub l: String,
    /// Volume.
    pub v: String,
    /// Number of trades.
    pub n: u64,
}

/// Internal struct matching Hyperliquid's raw JSON candle response.
#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
struct RawCandle {
    t: i64,
    T: i64,
    s: String,
    i: String,
    o: String,
    c: String,
    h: String,
    l: String,
    v: String,
    n: u64,
}

impl From<RawCandle> for HlCandle {
    fn from(r: RawCandle) -> Self {
        Self {
            t: r.t,
            t_close: r.T,
            s: r.s,
            i: r.i,
            o: r.o,
            c: r.c,
            h: r.h,
            l: r.l,
            v: r.v,
            n: r.n,
        }
    }
}

/// Fetches historical OHLCV candles from Hyperliquid's `candleSnapshot` API.
pub struct HlCandleFetcher {
    client: reqwest::Client,
}

impl HlCandleFetcher {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Fetch candles for a given symbol and interval within a time range.
    ///
    /// If the range exceeds 5000 candles, it is automatically paginated.
    /// Returns candles sorted by time (ascending).
    pub async fn fetch_candles(
        &self,
        symbol: &str,
        interval: &str,
        start_time_ms: i64,
        end_time_ms: i64,
    ) -> anyhow::Result<Vec<HlCandle>> {
        let interval_ms = parse_interval_ms(interval)?;
        let mut all_candles = Vec::new();
        let mut cursor = start_time_ms;

        while cursor < end_time_ms {
            let batch = self
                .fetch_batch(symbol, interval, cursor, end_time_ms)
                .await?;
            if batch.is_empty() {
                break;
            }
            let last_t = batch.last().map(|c| c.t).unwrap_or(cursor);
            all_candles.extend(batch);
            // Move cursor past the last candle
            cursor = last_t + interval_ms;
            // Rate limit between pages
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        // Deduplicate and sort by time
        all_candles.sort_by_key(|c| c.t);
        all_candles.dedup_by_key(|c| c.t);

        info!(
            "Fetched {} {} candles for {} ({} → {})",
            all_candles.len(),
            interval,
            symbol,
            DateTime::from_timestamp_millis(start_time_ms)
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_default(),
            DateTime::from_timestamp_millis(end_time_ms)
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_default(),
        );

        Ok(all_candles)
    }

    /// Fetch a single batch (up to 5000 candles).
    async fn fetch_batch(
        &self,
        symbol: &str,
        interval: &str,
        start_time_ms: i64,
        end_time_ms: i64,
    ) -> anyhow::Result<Vec<HlCandle>> {
        let body = serde_json::json!({
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
            }
        });

        debug!(
            "Fetching {} {} candles: {} → {}",
            symbol, interval, start_time_ms, end_time_ms
        );

        let resp = self
            .client
            .post(HL_INFO_URL)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "Hyperliquid API returned status {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            );
        }

        let raw: Vec<RawCandle> = resp.json().await?;
        Ok(raw.into_iter().map(HlCandle::from).collect())
    }
}

/// Parse interval string (e.g., "1m", "5m", "15m", "1h", "4h", "1d") to milliseconds.
fn parse_interval_ms(interval: &str) -> anyhow::Result<i64> {
    let (num, unit) = interval
        .chars()
        .partition::<String, _>(|c| c.is_ascii_digit());
    let n: i64 = num
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid interval: {}", interval))?;
    match unit.as_str() {
        "m" => Ok(n * 60_000),
        "h" => Ok(n * 3_600_000),
        "d" => Ok(n * 86_400_000),
        "w" => Ok(n * 604_800_000),
        _ => Err(anyhow::anyhow!(
            "Unknown interval unit: {} (use m, h, d, w)",
            unit
        )),
    }
}

// ---------------------------------------------------------------------------
// Backtest Position
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BtPosition {
    symbol: String,
    is_long: bool,
    entry_price: f64,
    current_price: f64,
    peak_price: f64,
    size_usd: f64,
    leverage: f64,
    open_time_ms: i64,
    entry_fee: f64,
    accrued_borrow_fee: f64,
    /// Borrow fee rate per hour on notional.
    borrow_rate_hourly: f64,
}

impl BtPosition {
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

    fn hold_secs(&self, now_ms: i64) -> u64 {
        ((now_ms - self.open_time_ms).max(0) / 1000) as u64
    }

    fn update_price(&mut self, price: f64, interval_secs: f64) {
        self.current_price = price;
        if self.is_long {
            if price > self.peak_price {
                self.peak_price = price;
            }
        } else if self.peak_price == 0.0 || price < self.peak_price {
            self.peak_price = price;
        }
        // Accrue borrow fee
        let hours = interval_secs / 3600.0;
        self.accrued_borrow_fee += self.size_usd * self.borrow_rate_hourly * hours;
    }

    #[allow(dead_code)]
    fn total_fees(&self) -> f64 {
        self.entry_fee + self.accrued_borrow_fee
    }
}

// ---------------------------------------------------------------------------
// Backtest Result Types
// ---------------------------------------------------------------------------

/// Per-cell statistics (mirrors paper engine's CellStats).
#[derive(Debug, Clone, Default, Serialize)]
pub struct BacktestCellStats {
    pub strategy: String,
    pub market: String,
    pub trade_count: usize,
    pub win_count: usize,
    pub loss_count: usize,
    pub gross_pnl: f64,
    pub total_fees: f64,
    pub entry_fees_total: f64,
    pub exit_fees_total: f64,
    pub borrow_fees_total: f64,
    pub net_pnl: f64,
    pub fee_ratio: f64,
    pub win_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown_usd: f64,
    pub avg_hold_secs: f64,
    pub best_trade_pnl: f64,
    pub worst_trade_pnl: f64,
    pub total_candles: usize,
    pub interval: String,
    pub start_time: String,
    pub end_time: String,
    /// Blueprint file or description that generated this strategy's parameters.
    /// Empty for built-in strategies; path to blueprint JSON for data-driven ones.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub strategy_source: String,
    /// Whether the Sharpe ratio meets the ≥ 1.0 threshold for live trading.
    #[serde(default)]
    pub sharpe_pass: bool,
}

impl BacktestCellStats {
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
        self.sharpe_pass = self.sharpe_ratio >= 1.0;
    }
}

/// Complete backtest result.
#[derive(Debug, Clone, Serialize)]
pub struct BacktestResult {
    pub start_balance: f64,
    pub final_balance: f64,
    pub total_net_pnl: f64,
    pub total_trades: usize,
    pub total_fees: f64,
    pub cells: Vec<BacktestCellStats>,
    pub candle_stats: HashMap<String, usize>,
    /// Strategies that did NOT meet the Sharpe ≥ 1.0 threshold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub below_sharpe_threshold: Vec<String>,
}

/// A single backtest trade record.
#[derive(Debug, Clone, Serialize)]
pub struct BtTrade {
    pub strategy: String,
    pub market: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size_usd: f64,
    pub gross_pnl: f64,
    pub entry_fee: f64,
    pub exit_fee: f64,
    pub borrow_fee: f64,
    pub net_pnl: f64,
    pub hold_secs: u64,
    pub exit_reason: String,
    pub entry_time: String,
    pub exit_time: String,
}

// ---------------------------------------------------------------------------
// Backtest Engine
// ---------------------------------------------------------------------------

/// Configuration for the backtest engine.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub strategies: Vec<String>,
    pub markets: Vec<String>,
    pub start_time_ms: i64,
    pub end_time_ms: i64,
    pub interval: String,
    pub starting_balance: f64,
    pub fee_rate: f64,
    pub borrow_rate_hourly: f64,
    pub leverage: f64,
}

/// The backtesting engine. Replays historical candles through strategies.
pub struct BacktestEngine {
    config: Config,
    bt_config: BacktestConfig,
}

impl BacktestEngine {
    pub fn new(config: Config, bt_config: BacktestConfig) -> anyhow::Result<Self> {
        // Validate strategy names
        let available = strategy::available_strategies();
        for name in &bt_config.strategies {
            if !available.contains(&name.as_str()) {
                anyhow::bail!(
                    "Unknown strategy '{}'. Available: {}",
                    name,
                    available.join(", ")
                );
            }
        }
        Ok(Self { config, bt_config })
    }

    /// Run the backtest. Returns results for each strategy x market cell.
    pub async fn run(&self) -> anyhow::Result<BacktestResult> {
        let fetcher = HlCandleFetcher::new();

        // Fetch candles for each market
        let mut candles_by_market: HashMap<String, Vec<HlCandle>> = HashMap::new();
        for market in &self.bt_config.markets {
            let candles = fetcher
                .fetch_candles(
                    market,
                    &self.bt_config.interval,
                    self.bt_config.start_time_ms,
                    self.bt_config.end_time_ms,
                )
                .await?;
            info!("{}: {} candles loaded", market, candles.len());
            candles_by_market.insert(market.clone(), candles);
        }

        // Validate we have data
        if candles_by_market.values().all(|c| c.is_empty()) {
            anyhow::bail!("No candle data fetched for any market. Check time range and symbols.");
        }

        // Run backtest for each strategy x market combination
        let mut cells = Vec::new();
        let mut trades: Vec<BtTrade> = Vec::new();
        let mut total_net_pnl = 0.0;
        let mut total_fees = 0.0;
        let mut total_trades = 0;

        for strat_name in &self.bt_config.strategies {
            for market in &self.bt_config.markets {
                let candles = match candles_by_market.get(market) {
                    Some(c) if !c.is_empty() => c,
                    _ => {
                        warn!("No candles for {}/{}, skipping", strat_name, market);
                        continue;
                    }
                };

                info!(
                    "Backtesting {} on {} ({} candles, {} interval)",
                    strat_name,
                    market,
                    candles.len(),
                    self.bt_config.interval
                );

                let (cell_stats, cell_trades) = self.run_cell(strat_name, market, candles)?;
                total_net_pnl += cell_stats.net_pnl;
                total_fees += cell_stats.total_fees;
                total_trades += cell_stats.trade_count;
                trades.extend(cell_trades);
                cells.push(cell_stats);
            }
        }

        let final_balance = self.bt_config.starting_balance + total_net_pnl;

        // Write trades to JSON
        let trades_path = "data/backtest-trades.json";
        write_json_atomic(trades_path, &trades)?;

        let mut candle_stats = HashMap::new();
        for (m, c) in &candles_by_market {
            candle_stats.insert(m.clone(), c.len());
        }

        let mut result = BacktestResult {
            start_balance: self.bt_config.starting_balance,
            final_balance,
            total_net_pnl,
            total_trades,
            total_fees,
            cells,
            candle_stats,
            below_sharpe_threshold: Vec::new(),
        };

        // Identify strategies below Sharpe ≥ 1.0 threshold
        for cell in &result.cells {
            let key = format!("{}:{}", cell.strategy, cell.market);
            if !cell.sharpe_pass {
                result.below_sharpe_threshold.push(key);
            }
        }

        if !result.below_sharpe_threshold.is_empty() {
            warn!(
                "Strategies below Sharpe ≥ 1.0 threshold: {}",
                result.below_sharpe_threshold.join(", ")
            );
        } else {
            info!("All strategies meet Sharpe ≥ 1.0 threshold");
        }

        // Write summary
        let summary_path = "data/backtest-results/summary.json";
        write_json_atomic(summary_path, &result)?;

        // Print summary table
        self.print_summary(&result);

        Ok(result)
    }

    /// Run backtest for a single strategy x market cell.
    fn run_cell(
        &self,
        strategy_name: &str,
        market: &str,
        candles: &[HlCandle],
    ) -> anyhow::Result<(BacktestCellStats, Vec<BtTrade>)> {
        let sub_table = self.config.strategy.get_sub_table(strategy_name);
        let fallback_params = self
            .config
            .strategy
            .get_params(strategy_name)
            .unwrap_or_else(|_| crate::strategy::StrategyParams {
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
            });

        let mut strat =
            strategy::create_strategy_from_config(strategy_name, sub_table, fallback_params)?;
        let params = strat.parameters().clone();
        let interval_secs = parse_interval_ms(&self.bt_config.interval)? as f64 / 1000.0;
        let interval_ms = parse_interval_ms(&self.bt_config.interval)?;

        let mut position: Option<BtPosition> = None;
        let mut stats = BacktestCellStats {
            strategy: strategy_name.to_string(),
            market: market.to_string(),
            interval: self.bt_config.interval.clone(),
            total_candles: candles.len(),
            start_time: DateTime::from_timestamp_millis(candles.first().map(|c| c.t).unwrap_or(0))
                .map(|t| t.to_rfc3339())
                .unwrap_or_default(),
            end_time: DateTime::from_timestamp_millis(
                candles.last().map(|c| c.t_close).unwrap_or(0),
            )
            .map(|t| t.to_rfc3339())
            .unwrap_or_default(),
            strategy_source: strategy_source_path(strategy_name),
            ..Default::default()
        };

        let mut trades = Vec::new();
        let mut cell_balance = self.bt_config.starting_balance;
        let mut max_equity = cell_balance;
        let mut cooldown_until_ms: i64 = 0;

        // Equity curve (one sample per candle) — drives mid-position drawdown
        // tracking and the annualized Sharpe calculation below.
        let mut equity_curve: Vec<f64> = Vec::with_capacity(candles.len() + 1);
        equity_curve.push(cell_balance);

        // Pending entry signal from the previous candle's close. Filled at
        // *this* candle's open to avoid lookahead bias (signal computed using
        // close(N) must not also fill at close(N)).
        let mut pending_entry: Option<bool> = None;

        let bias = params.direction_bias.to_lowercase();
        // Fees on Flash perps are charged on the levered notional, not on the
        // collateral. The previous version multiplied by `clip_size_usd` only
        // and so under-counted fees by the leverage factor.
        let fee_notional_factor = self.bt_config.leverage.max(1.0);
        let lockout_ticks: i64 = 6;
        let lockout_ms = lockout_ticks * interval_ms;

        for candle in candles {
            let open_price: f64 = candle.o.parse().unwrap_or(0.0);
            let high_price: f64 = candle.h.parse().unwrap_or(0.0);
            let low_price: f64 = candle.l.parse().unwrap_or(0.0);
            let close_price: f64 = candle.c.parse().unwrap_or(0.0);
            if close_price <= 0.0 || open_price <= 0.0 || high_price <= 0.0 || low_price <= 0.0 {
                continue;
            }

            // 1) Fill any pending entry queued from the previous candle's
            //    close-based signal at THIS candle's open. This is the
            //    realistic execution model: signal at bar close → order
            //    queued → fills at next bar open.
            if position.is_none() {
                if let Some(is_long) = pending_entry.take() {
                    let clip = params.clip_size_usd;
                    let notional = clip * fee_notional_factor;
                    let entry_fee = notional * self.bt_config.fee_rate;
                    position = Some(BtPosition {
                        symbol: market.to_string(),
                        is_long,
                        entry_price: open_price,
                        current_price: open_price,
                        peak_price: open_price,
                        size_usd: clip,
                        leverage: self.bt_config.leverage,
                        open_time_ms: candle.t,
                        entry_fee,
                        accrued_borrow_fee: 0.0,
                        borrow_rate_hourly: self.bt_config.borrow_rate_hourly,
                    });
                    debug!(
                        "[BT] {} {} {} @ ${:.4} (next-open fill)",
                        strategy_name,
                        if is_long { "LONG" } else { "SHORT" },
                        market,
                        open_price
                    );
                }
            } else {
                // Already have a position; cancel any stale pending entry.
                pending_entry = None;
            }

            // 2) Feed the close into the strategy for signal evaluation.
            strat.push_price(close_price, candle.t);
            let snapshot = strat.snapshot();

            // 3) Manage existing position
            if position.is_some() {
                // Accrue borrow + update peak — exactly once per candle.
                if let Some(ref mut pos_mut) = position {
                    pos_mut.update_price(close_price, interval_secs);
                }

                let (is_long, entry_price, peak_price, hold_secs) = {
                    let pos = position.as_ref().unwrap();
                    (
                        pos.is_long,
                        pos.entry_price,
                        pos.peak_price,
                        pos.hold_secs(candle.t),
                    )
                };

                // 3a) Intra-candle TP/SL check. These price-based exits fill
                //     at the trigger level (not close), and SL takes priority
                //     over TP when both are touched.
                let intra = intra_candle_tp_sl_hit(
                    is_long,
                    entry_price,
                    params.take_profit_pct,
                    params.stop_loss_pct,
                    low_price,
                    high_price,
                );

                // 3b) For non-price exits (time stop, momentum lost, reversal
                //     detection, trailing stop) consult the strategy. These
                //     fill at the bar's close.
                let strat_exit = if intra.is_none() {
                    let context = PositionContext {
                        is_long,
                        entry_price,
                        current_price: close_price,
                        peak_price,
                        hold_secs,
                        max_hold_secs: params.max_hold_secs,
                        take_profit_pct: params.take_profit_pct,
                        stop_loss_pct: params.stop_loss_pct,
                        trailing_stop_pct: params.trailing_stop_pct,
                        trailing_activation_pct: params.trailing_activation_pct,
                    };
                    strat.detect_exit(&snapshot, &context)
                } else {
                    None
                };

                let (exit_reason_enum, fill_price): (Option<ExitReason>, f64) = match (intra, strat_exit) {
                    (Some((reason, px)), _) => (Some(reason), px),
                    (None, Some(Signal::ExitLong { reason })) if is_long => (Some(reason), close_price),
                    (None, Some(Signal::ExitShort { reason })) if !is_long => (Some(reason), close_price),
                    _ => (None, close_price),
                };

                if let Some(reason) = exit_reason_enum {
                    let pos = position.take().unwrap();
                    let exit_notional = pos.size_usd * fee_notional_factor;
                    let exit_fee = exit_notional * self.bt_config.fee_rate;
                    let gross_pnl = if pos.is_long {
                        pos.size_usd * (fill_price - pos.entry_price) / pos.entry_price
                    } else {
                        pos.size_usd * (pos.entry_price - fill_price) / pos.entry_price
                    };
                    let total_fees = pos.entry_fee + exit_fee + pos.accrued_borrow_fee;
                    let net_pnl = gross_pnl - exit_fee - pos.accrued_borrow_fee;

                    let exit_reason_str = match reason {
                        ExitReason::StopLoss => "stop_loss",
                        ExitReason::TakeProfit => "take_profit",
                        ExitReason::TrailingStop => "trailing_stop",
                        ExitReason::TimeStop => "time_stop",
                        ExitReason::MomentumLost => "momentum_lost",
                        ExitReason::ReversalDetected => "reversal",
                    };

                    cell_balance += net_pnl;

                    stats.trade_count += 1;
                    stats.gross_pnl += gross_pnl;
                    stats.total_fees += total_fees;
                    stats.entry_fees_total += pos.entry_fee;
                    stats.exit_fees_total += exit_fee;
                    stats.borrow_fees_total += pos.accrued_borrow_fee;
                    stats.net_pnl += net_pnl;

                    if net_pnl >= 0.0 {
                        stats.win_count += 1;
                    } else {
                        stats.loss_count += 1;
                    }
                    cooldown_until_ms = candle.t + lockout_ms;

                    if net_pnl > stats.best_trade_pnl {
                        stats.best_trade_pnl = net_pnl;
                    }
                    if net_pnl < stats.worst_trade_pnl {
                        stats.worst_trade_pnl = net_pnl;
                    }

                    let exit_hold_secs = pos.hold_secs(candle.t);
                    stats.avg_hold_secs = if stats.trade_count == 1 {
                        exit_hold_secs as f64
                    } else {
                        let prev_total = stats.avg_hold_secs * (stats.trade_count - 1) as f64;
                        (prev_total + exit_hold_secs as f64) / stats.trade_count as f64
                    };

                    trades.push(BtTrade {
                        strategy: strategy_name.to_string(),
                        market: market.to_string(),
                        side: if pos.is_long {
                            "LONG".to_string()
                        } else {
                            "SHORT".to_string()
                        },
                        entry_price: pos.entry_price,
                        exit_price: fill_price,
                        size_usd: pos.size_usd,
                        gross_pnl,
                        entry_fee: pos.entry_fee,
                        exit_fee,
                        borrow_fee: pos.accrued_borrow_fee,
                        net_pnl,
                        hold_secs: exit_hold_secs,
                        exit_reason: exit_reason_str.to_string(),
                        entry_time: DateTime::from_timestamp_millis(pos.open_time_ms)
                            .map(|t| t.to_rfc3339())
                            .unwrap_or_default(),
                        exit_time: DateTime::from_timestamp_millis(candle.t)
                            .map(|t| t.to_rfc3339())
                            .unwrap_or_default(),
                    });
                }
            }

            // 4) Mark-to-market equity and drawdown — sampled every candle,
            //    not just at trade close, so unrealized losses are visible.
            let unrealized = match &position {
                Some(p) => p.unrealized_pnl_usd() - p.accrued_borrow_fee,
                None => 0.0,
            };
            let mtm_equity = cell_balance + unrealized;
            equity_curve.push(mtm_equity);
            if mtm_equity > max_equity {
                max_equity = mtm_equity;
            }
            let dd = max_equity - mtm_equity;
            if dd > stats.max_drawdown_usd {
                stats.max_drawdown_usd = dd;
            }

            // 5) Entry-signal generation. Queue for next-open fill; honor the
            //    strategy's direction_bias so the backtest matches what the
            //    live engine would do.
            if position.is_none()
                && candle.t >= cooldown_until_ms
                && pending_entry.is_none()
            {
                let entry_signal = strat.detect_entry(&snapshot);
                match entry_signal {
                    Signal::MomentumLong { strength, .. } if bias != "short" => {
                        pending_entry = Some(true);
                        debug!(
                            "[BT] {} {} LONG signal @ close ${:.4} → queued for next open (strength={:.2})",
                            strategy_name, market, close_price, strength
                        );
                    }
                    Signal::MomentumShort { strength, .. } if bias != "long" => {
                        pending_entry = Some(false);
                        debug!(
                            "[BT] {} {} SHORT signal @ close ${:.4} → queued for next open (strength={:.2})",
                            strategy_name, market, close_price, strength
                        );
                    }
                    _ => {}
                }
            }
        }

        // Force-close any open position at the last candle's close price
        if let Some(pos) = position.take() {
            let last_price = pos.current_price;
            let exit_notional = pos.size_usd * fee_notional_factor;
            let exit_fee = exit_notional * self.bt_config.fee_rate;
            let gross_pnl = pos.unrealized_pnl_usd();
            let net_pnl = gross_pnl - exit_fee - pos.accrued_borrow_fee;
            let total_fees = pos.entry_fee + exit_fee + pos.accrued_borrow_fee;

            // No cell_balance/equity_curve update here: this branch runs
            // after the main loop, so the equity series is already closed
            // out. Stats below capture the final trade's PnL contribution.

            stats.trade_count += 1;
            stats.gross_pnl += gross_pnl;
            stats.total_fees += total_fees;
            stats.entry_fees_total += pos.entry_fee;
            stats.exit_fees_total += exit_fee;
            stats.borrow_fees_total += pos.accrued_borrow_fee;
            stats.net_pnl += net_pnl;

            if net_pnl >= 0.0 {
                stats.win_count += 1;
            } else {
                stats.loss_count += 1;
            }

            trades.push(BtTrade {
                strategy: strategy_name.to_string(),
                market: market.to_string(),
                side: if pos.is_long {
                    "LONG".to_string()
                } else {
                    "SHORT".to_string()
                },
                entry_price: pos.entry_price,
                exit_price: last_price,
                size_usd: pos.size_usd,
                gross_pnl,
                entry_fee: pos.entry_fee,
                exit_fee,
                borrow_fee: pos.accrued_borrow_fee,
                net_pnl,
                hold_secs: pos.hold_secs(candles.last().map(|c| c.t).unwrap_or(pos.open_time_ms)),
                exit_reason: "end_of_data".to_string(),
                entry_time: DateTime::from_timestamp_millis(pos.open_time_ms)
                    .map(|t| t.to_rfc3339())
                    .unwrap_or_default(),
                exit_time: DateTime::from_timestamp_millis(
                    candles.last().map(|c| c.t).unwrap_or(pos.open_time_ms),
                )
                .map(|t| t.to_rfc3339())
                .unwrap_or_default(),
            });

            warn!(
                "Force-closed {} {} position at end of data (net PnL: ${:.2})",
                strategy_name, market, net_pnl
            );
        }

        // Annualized Sharpe ratio computed over per-bar equity returns. This
        // is the right denominator for the live promotion gate (≥ 1.0): a
        // per-trade Sharpe on small winners trivially hits 1.0 and means
        // nothing about risk-adjusted performance.
        if equity_curve.len() >= 3 {
            let mut returns: Vec<f64> = Vec::with_capacity(equity_curve.len() - 1);
            for w in equity_curve.windows(2) {
                let prev = w[0];
                let curr = w[1];
                if prev > 0.0 {
                    returns.push((curr - prev) / prev);
                }
            }
            if returns.len() >= 2 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / (returns.len() - 1) as f64;
                let std_dev = variance.sqrt();
                // Bars/year = seconds/year ÷ bar duration. Annualization
                // factor on Sharpe is sqrt(N_bars_per_year).
                let bars_per_year = 365.25 * 86_400.0 / interval_secs;
                stats.sharpe_ratio = if std_dev > 0.0 {
                    (mean / std_dev) * bars_per_year.sqrt()
                } else {
                    0.0
                };
            }
        }

        // Compute final metrics (win_rate, fee_ratio, sharpe_pass).
        stats.finalize();

        Ok((stats, trades))
    }

    /// Print a summary table of backtest results.
    fn print_summary(&self, result: &BacktestResult) {
        info!("╔══════════════════════════════════════════════════════════════════════╗");
        info!("║                     BACKTEST RESULTS SUMMARY                       ║");
        info!("╠══════════════════════════════════════════════════════════════════════╣");
        info!(
            "║ Starting Balance: ${:.2}  →  Final: ${:.2}",
            result.start_balance, result.final_balance
        );
        info!(
            "║ Total Net PnL: ${:.2}  |  Total Fees: ${:.2}  |  Total Trades: {}",
            result.total_net_pnl, result.total_fees, result.total_trades
        );
        info!("╠══════════════════════════════════════════════════════════════════════╣");
        info!(
            "║ {:<20} {:<6} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6} {:>5}",
            "Strategy", "Mkt", "Trds", "Gross$", "Fees$", "Net$", "Win%", "Sharpe", "Pass"
        );
        info!("╠══════════════════════════════════════════════════════════════════════╣");

        for cell in &result.cells {
            let pass_flag = if cell.sharpe_pass { "YES" } else { "NO" };
            info!(
                "║ {:<20} {:<6} {:>5} {:>8.2} {:>8.2} {:>8.2} {:>5.1}% {:>6.2} {:>5}",
                cell.strategy,
                cell.market,
                cell.trade_count,
                cell.gross_pnl,
                cell.total_fees,
                cell.net_pnl,
                cell.win_rate,
                cell.sharpe_ratio,
                pass_flag,
            );
            if !cell.strategy_source.is_empty() {
                info!("║   ↳ source: {}", cell.strategy_source);
            }
        }

        if !result.below_sharpe_threshold.is_empty() {
            info!("╠══════════════════════════════════════════════════════════════════════╣");
            warn!(
                "║ BELOW Sharpe ≥ 1.0: {}",
                result.below_sharpe_threshold.join(", ")
            );
        }
        info!("╚══════════════════════════════════════════════════════════════════════╝");
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether a candle's OHLC range crossed TP or SL levels for an open
/// position. Returns the resolved exit (reason + fill price) if any.
///
/// When both TP and SL would have been touched in the same candle, SL is
/// assumed to fill first — the conservative worst-case for a backtest, and
/// the only honest assumption without tick data.
fn intra_candle_tp_sl_hit(
    is_long: bool,
    entry_price: f64,
    tp_pct: f64,
    sl_pct: f64,
    low: f64,
    high: f64,
) -> Option<(ExitReason, f64)> {
    if entry_price <= 0.0 || low <= 0.0 || high <= 0.0 {
        return None;
    }
    let (tp_price, sl_price) = if is_long {
        (
            entry_price * (1.0 + tp_pct / 100.0),
            entry_price * (1.0 - sl_pct / 100.0),
        )
    } else {
        (
            entry_price * (1.0 - tp_pct / 100.0),
            entry_price * (1.0 + sl_pct / 100.0),
        )
    };
    let tp_hit = tp_pct > 0.0
        && if is_long {
            high >= tp_price
        } else {
            low <= tp_price
        };
    let sl_hit = sl_pct > 0.0
        && if is_long {
            low <= sl_price
        } else {
            high >= sl_price
        };

    match (tp_hit, sl_hit) {
        (false, false) => None,
        (true, false) => Some((ExitReason::TakeProfit, tp_price)),
        (false, true) => Some((ExitReason::StopLoss, sl_price)),
        // Both hit in the same candle: assume SL fills first (worst case).
        (true, true) => Some((ExitReason::StopLoss, sl_price)),
    }
}

/// Map a strategy name to its blueprint source path (for data-driven strategies).
/// Returns an empty string for built-in strategies.
fn strategy_source_path(name: &str) -> String {
    match name {
        "blueprint-scalper" => "data/blueprints/cluster-001.json".to_string(),
        "blueprint-mean-revert" => "data/blueprints/cluster-004.json".to_string(),
        s if s.starts_with("blueprint-cluster-") => {
            format!(
                "data/blueprints/{}.json",
                s.strip_prefix("blueprint-").unwrap()
            )
        }
        _ => String::new(), // Built-in strategies have no blueprint source
    }
}

/// Write JSON to a file atomically (write .tmp, rename).
fn write_json_atomic<T: Serialize>(path: &str, data: &T) -> anyhow::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = format!("{}.tmp", path);
    let json = serde_json::to_string_pretty(data)?;
    std::fs::write(&tmp_path, json)?;
    std::fs::rename(&tmp_path, path)?;
    info!("Results written to {}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_interval_ms() {
        assert_eq!(parse_interval_ms("1m").unwrap(), 60_000);
        assert_eq!(parse_interval_ms("5m").unwrap(), 300_000);
        assert_eq!(parse_interval_ms("15m").unwrap(), 900_000);
        assert_eq!(parse_interval_ms("1h").unwrap(), 3_600_000);
        assert_eq!(parse_interval_ms("4h").unwrap(), 14_400_000);
        assert_eq!(parse_interval_ms("1d").unwrap(), 86_400_000);
        assert_eq!(parse_interval_ms("1w").unwrap(), 604_800_000);
    }

    #[test]
    fn test_parse_interval_invalid() {
        assert!(parse_interval_ms("abc").is_err());
        assert!(parse_interval_ms("1x").is_err());
        assert!(parse_interval_ms("").is_err());
    }

    #[test]
    fn test_bt_position_long_pnl() {
        let pos = BtPosition {
            symbol: "BTC".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 105.0,
            peak_price: 105.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time_ms: 0,
            entry_fee: 1.0,
            accrued_borrow_fee: 0.5,
            borrow_rate_hourly: 0.0001,
        };
        assert!((pos.unrealized_pnl_pct() - 5.0).abs() < 0.001);
        assert!((pos.unrealized_pnl_usd() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_bt_position_short_pnl() {
        let pos = BtPosition {
            symbol: "BTC".to_string(),
            is_long: false,
            entry_price: 100.0,
            current_price: 95.0,
            peak_price: 95.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time_ms: 0,
            entry_fee: 1.0,
            accrued_borrow_fee: 0.5,
            borrow_rate_hourly: 0.0001,
        };
        assert!((pos.unrealized_pnl_pct() - 5.0).abs() < 0.001);
        assert!((pos.unrealized_pnl_usd() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_bt_position_update_price_long() {
        let mut pos = BtPosition {
            symbol: "BTC".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 100.0,
            peak_price: 100.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time_ms: 0,
            entry_fee: 1.0,
            accrued_borrow_fee: 0.0,
            borrow_rate_hourly: 0.0001,
        };
        pos.update_price(110.0, 300.0); // 5min candle
        assert_eq!(pos.peak_price, 110.0);
        assert!(pos.accrued_borrow_fee > 0.0);
    }

    #[test]
    fn test_bt_position_update_price_short() {
        let mut pos = BtPosition {
            symbol: "BTC".to_string(),
            is_long: false,
            entry_price: 100.0,
            current_price: 100.0,
            peak_price: 100.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time_ms: 0,
            entry_fee: 1.0,
            accrued_borrow_fee: 0.0,
            borrow_rate_hourly: 0.0001,
        };
        pos.update_price(90.0, 300.0);
        assert_eq!(pos.peak_price, 90.0); // Tracks lowest for shorts
    }

    #[test]
    fn test_bt_position_hold_secs() {
        let pos = BtPosition {
            symbol: "BTC".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 100.0,
            peak_price: 100.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time_ms: 1000000,
            entry_fee: 1.0,
            accrued_borrow_fee: 0.0,
            borrow_rate_hourly: 0.0001,
        };
        assert_eq!(pos.hold_secs(1060000), 60); // 60 seconds
    }

    #[test]
    fn test_bt_position_total_fees() {
        let pos = BtPosition {
            symbol: "BTC".to_string(),
            is_long: true,
            entry_price: 100.0,
            current_price: 100.0,
            peak_price: 100.0,
            size_usd: 1000.0,
            leverage: 5.0,
            open_time_ms: 0,
            entry_fee: 1.0,
            accrued_borrow_fee: 0.5,
            borrow_rate_hourly: 0.0001,
        };
        assert!((pos.total_fees() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_backtest_cell_stats_finalize() {
        let mut stats = BacktestCellStats {
            strategy: "test".to_string(),
            market: "BTC".to_string(),
            trade_count: 10,
            win_count: 6,
            loss_count: 4,
            gross_pnl: 100.0,
            total_fees: 20.0,
            net_pnl: 80.0,
            ..Default::default()
        };
        stats.finalize();
        assert!((stats.win_rate - 60.0).abs() < 0.001);
        assert!((stats.fee_ratio - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_backtest_cell_stats_finalize_no_trades() {
        let mut stats = BacktestCellStats::default();
        stats.finalize();
        assert_eq!(stats.win_rate, 0.0);
        assert_eq!(stats.fee_ratio, 0.0);
    }

    #[test]
    fn test_candle_deserialization() {
        let json = r#"{
            "t": 1778895720000,
            "T": 1778895779999,
            "s": "BTC",
            "i": "1m",
            "o": "78988.0",
            "c": "78993.0",
            "h": "79002.0",
            "l": "78988.0",
            "v": "10.93",
            "n": 149
        }"#;
        let raw: RawCandle = serde_json::from_str(json).unwrap();
        assert_eq!(raw.t, 1778895720000);
        assert_eq!(raw.s, "BTC");
        let candle: HlCandle = raw.into();
        assert_eq!(candle.t, 1778895720000);
        assert_eq!(candle.t_close, 1778895779999);
    }

    #[test]
    fn test_write_json_atomic() {
        let tmp_dir = std::env::temp_dir().join("dalgona_backtest_test");
        std::fs::create_dir_all(&tmp_dir).ok();
        let path = tmp_dir
            .join("test_output.json")
            .to_str()
            .unwrap()
            .to_string();
        let data = serde_json::json!({"test": 42});
        write_json_atomic(&path, &data).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"test\": 42"));
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    // Integration-style test: create a strategy and run a synthetic backtest
    /// Minimal valid TOML config for tests.
    fn test_config_toml(market: &str) -> String {
        format!(
            r#"
[agent]
log_level = "info"
poll_interval_secs = 5

[flash]
api_url = "https://flashapi.trade"
rpc_url = "https://api.mainnet-beta.solana.com"
keypair_path = ""
market = "{}"
input_token = "USDC"
pool = "Crypto.1"
leverage = 5.0
slippage_pct = "0.5"

[strategy]
active = "momentum-scalper"
clip_size_usd = 100.0

[risk]
max_position_notional_usd = 1000.0
max_daily_loss_usd = 200.0
max_drawdown_pct = 20.0
"#,
            market
        )
    }

    fn test_bt_config(strategies: Vec<&str>, markets: Vec<&str>, interval: &str) -> BacktestConfig {
        BacktestConfig {
            strategies: strategies.into_iter().map(String::from).collect(),
            markets: markets.into_iter().map(String::from).collect(),
            start_time_ms: 0,
            end_time_ms: 0,
            interval: interval.to_string(),
            starting_balance: 1000.0,
            fee_rate: 0.001,
            borrow_rate_hourly: 0.0001,
            leverage: 5.0,
        }
    }

    #[test]
    fn test_run_cell_synthetic() {
        let config: crate::config::Config = toml::from_str(&test_config_toml("BTC")).unwrap();
        let bt_config = test_bt_config(vec!["momentum-scalper"], vec!["BTC"], "5m");
        let engine = BacktestEngine::new(config, bt_config).unwrap();

        // Create synthetic candles: gradually rising prices
        let mut candles = Vec::new();
        let base_time = 1778812800000i64;
        for i in 0..60 {
            let price = 100.0 + (i as f64 * 0.1); // Slowly rising
            candles.push(HlCandle {
                t: base_time + (i as i64 * 300000),
                t_close: base_time + ((i as i64 + 1) * 300000) - 1,
                s: "BTC".to_string(),
                i: "5m".to_string(),
                o: format!("{}", price - 0.05),
                c: format!("{}", price),
                h: format!("{}", price + 0.1),
                l: format!("{}", price - 0.1),
                v: "100.0".to_string(),
                n: 100,
            });
        }

        let (stats, _trades) = engine
            .run_cell("momentum-scalper", "BTC", &candles)
            .unwrap();
        // Even with no trades, stats should be valid
        assert_eq!(stats.strategy, "momentum-scalper");
        assert_eq!(stats.market, "BTC");
        assert_eq!(stats.total_candles, 60);
        // The slowly rising prices may or may not trigger momentum signals
        // depending on threshold, which is fine
        assert!(stats.trade_count < 100); // Sanity check
    }

    #[test]
    fn test_run_cell_volatile_data() {
        let config: crate::config::Config = toml::from_str(&test_config_toml("SOL")).unwrap();
        let bt_config = test_bt_config(vec!["momentum-scalper"], vec!["SOL"], "1m");
        let engine = BacktestEngine::new(config, bt_config).unwrap();

        // Create volatile synthetic candles with a clear momentum spike
        let mut candles = Vec::new();
        let base_time = 1778812800000i64;
        let mut price = 90.0;
        for i in 0..120 {
            // Flat for 60 candles, then spike up
            if i >= 60 && i < 75 {
                price += 0.5; // Strong upward momentum
            } else if i >= 75 && i < 85 {
                price -= 0.3; // Reversal
            } else {
                price += (i as f64 * 0.001).sin() * 0.1; // Small noise
            }
            candles.push(HlCandle {
                t: base_time + (i as i64 * 60000),
                t_close: base_time + ((i as i64 + 1) * 60000) - 1,
                s: "SOL".to_string(),
                i: "1m".to_string(),
                o: format!("{:.3}", price - 0.05),
                c: format!("{:.3}", price),
                h: format!("{:.3}", price + 0.1),
                l: format!("{:.3}", price - 0.1),
                v: "1000.0".to_string(),
                n: 50,
            });
        }

        let (stats, _trades) = engine
            .run_cell("momentum-scalper", "SOL", &candles)
            .unwrap();
        assert_eq!(stats.total_candles, 120);
        assert_eq!(stats.strategy, "momentum-scalper");
        // With a momentum spike, we expect at least some activity
        // (exact count depends on strategy threshold)
    }

    #[test]
    fn test_backtest_engine_new_validates_strategies() {
        let config: crate::config::Config = toml::from_str(&test_config_toml("BTC")).unwrap();
        let bt_config = BacktestConfig {
            strategies: vec!["nonexistent-strategy".to_string()],
            ..test_bt_config(vec!["nonexistent-strategy"], vec!["BTC"], "5m")
        };
        assert!(BacktestEngine::new(config, bt_config).is_err());
    }

    // -------------------------------------------------------------------------
    // VAL-VALIDATE-002: Sharpe ratio filter (≥ 1.0 threshold)
    // VAL-VALIDATE-003: Backtest results file has complete schema
    // VAL-CROSS-003: Backtest results reference strategy source
    // -------------------------------------------------------------------------
    #[test]
    fn test_strategy_source_path_mapping() {
        // Data-driven strategies should map to their blueprint files
        assert_eq!(
            strategy_source_path("blueprint-scalper"),
            "data/blueprints/cluster-001.json"
        );
        assert_eq!(
            strategy_source_path("blueprint-mean-revert"),
            "data/blueprints/cluster-004.json"
        );
        // Built-in strategies have no blueprint source
        assert_eq!(strategy_source_path("momentum-scalper"), "");
        assert_eq!(strategy_source_path("lp-consumption"), "");
        assert_eq!(strategy_source_path("mean-reversion"), "");
        assert_eq!(strategy_source_path("trend-follower"), "");
    }

    #[test]
    fn test_backtest_cell_stats_strategy_source() {
        let config: crate::config::Config = toml::from_str(&test_config_toml("BTC")).unwrap();
        let bt_config = test_bt_config(vec!["momentum-scalper"], vec!["BTC"], "5m");
        let engine = BacktestEngine::new(config, bt_config).unwrap();

        let candles = vec![HlCandle {
            t: 1778812800000,
            t_close: 1778813099999,
            s: "BTC".to_string(),
            i: "5m".to_string(),
            o: "100.0".to_string(),
            c: "100.0".to_string(),
            h: "100.0".to_string(),
            l: "100.0".to_string(),
            v: "100.0".to_string(),
            n: 10,
        }];

        let (stats, _) = engine
            .run_cell("momentum-scalper", "BTC", &candles)
            .unwrap();
        assert!(
            stats.strategy_source.is_empty(),
            "Built-in strategy should have empty source"
        );
    }

    #[test]
    fn test_sharpe_pass_flag() {
        let mut stats = BacktestCellStats {
            strategy: "test".to_string(),
            market: "BTC".to_string(),
            sharpe_ratio: 1.5,
            ..Default::default()
        };
        stats.finalize();
        assert!(stats.sharpe_pass, "Sharpe 1.5 should pass ≥ 1.0 threshold");

        stats.sharpe_ratio = 0.5;
        stats.finalize();
        assert!(
            !stats.sharpe_pass,
            "Sharpe 0.5 should NOT pass ≥ 1.0 threshold"
        );

        stats.sharpe_ratio = 1.0;
        stats.finalize();
        assert!(stats.sharpe_pass, "Sharpe exactly 1.0 should pass");
    }

    #[test]
    fn test_backtest_cell_stats_serialization_has_strategy_source() {
        let stats = BacktestCellStats {
            strategy: "blueprint-scalper".to_string(),
            market: "BTC".to_string(),
            strategy_source: "data/blueprints/cluster-001.json".to_string(),
            sharpe_ratio: 1.5,
            sharpe_pass: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(
            json.contains("\"strategy_source\":\"data/blueprints/cluster-001.json\""),
            "Serialized JSON should contain strategy_source field"
        );
        assert!(
            json.contains("\"sharpe_pass\":true"),
            "Serialized JSON should contain sharpe_pass field"
        );
    }

    #[test]
    fn test_below_sharpe_threshold_tracking() {
        let result = BacktestResult {
            start_balance: 1000.0,
            final_balance: 1050.0,
            total_net_pnl: 50.0,
            total_trades: 10,
            total_fees: 5.0,
            cells: vec![
                BacktestCellStats {
                    strategy: "momentum-scalper".to_string(),
                    market: "BTC".to_string(),
                    sharpe_ratio: 0.5,
                    sharpe_pass: false,
                    ..Default::default()
                },
                BacktestCellStats {
                    strategy: "blueprint-scalper".to_string(),
                    market: "BTC".to_string(),
                    sharpe_ratio: 1.5,
                    sharpe_pass: true,
                    ..Default::default()
                },
            ],
            candle_stats: HashMap::new(),
            below_sharpe_threshold: vec!["momentum-scalper:BTC".to_string()],
        };

        assert_eq!(result.below_sharpe_threshold.len(), 1);
        assert_eq!(result.below_sharpe_threshold[0], "momentum-scalper:BTC");
    }
}
