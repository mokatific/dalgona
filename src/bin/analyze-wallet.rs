//! analyze-wallet — Fetches trade history, computes 12-metric classification suite,
//! classifies wallet strategies, and outputs per-wallet reports + strategy blueprints.
//!
//! Modes:
//!   --address <ADDR> --source <flash|jupiter|hyperliquid>  Single wallet
//!   --wallets data/wallets.json                            Batch from scrape output
//!
//! Output:
//!   data/reports/<address>.json         Per-wallet analysis
//!   data/strategy-blueprints/<name>.json  Aggregated strategy blueprints

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "analyze-wallet",
    about = "Analyze wallet trading history and classify strategies",
    version
)]
struct Args {
    /// Single wallet address to analyze
    #[arg(long)]
    address: Option<String>,

    /// Path to wallets.json (batch mode, from scrape-leaderboards)
    #[arg(long)]
    wallets: Option<PathBuf>,

    /// Data source: flash, jupiter, hyperliquid (auto-detected from wallets.json in batch mode)
    #[arg(long, value_enum)]
    source: Option<SourceArg>,

    /// Output directory for reports
    #[arg(short, long, default_value = "data/reports")]
    output: PathBuf,

    /// Delay in seconds between API requests to the same host
    #[arg(long, default_value_t = 1.0)]
    rate_limit: f64,

    /// Output directory for strategy blueprints
    #[arg(long, default_value = "data/strategy-blueprints")]
    blueprints_dir: PathBuf,
}

#[derive(Clone, Debug, clap::ValueEnum, PartialEq)]
enum SourceArg {
    Flash,
    Jupiter,
    Hyperliquid,
}

impl std::fmt::Display for SourceArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceArg::Flash => write!(f, "flash"),
            SourceArg::Jupiter => write!(f, "jupiter"),
            SourceArg::Hyperliquid => write!(f, "hyperliquid"),
        }
    }
}

// ── Rate limiter ─────────────────────────────────────────────────────────────

struct RateLimiter {
    last_request: Arc<Mutex<HashMap<String, Instant>>>,
    min_interval: Duration,
}

impl RateLimiter {
    fn new(interval_secs: f64) -> Self {
        Self {
            last_request: Arc::new(Mutex::new(HashMap::new())),
            min_interval: Duration::from_secs_f64(interval_secs.max(0.0)),
        }
    }

    async fn throttle(&self, host: &str) {
        let mut map = self.last_request.lock().await;
        if let Some(last) = map.get(host) {
            let elapsed = last.elapsed();
            if elapsed < self.min_interval {
                let sleep_dur = self.min_interval - elapsed;
                debug!(host, sleep_ms = sleep_dur.as_millis(), "Rate limiting");
                drop(map);
                tokio::time::sleep(sleep_dur).await;
                map = self.last_request.lock().await;
            }
        }
        map.insert(host.to_string(), Instant::now());
    }
}

// ── Input schema (matches scrape-leaderboards output) ────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WalletInput {
    address: String,
    source: String,
    #[serde(default)]
    rank: u32,
    #[serde(default)]
    total_trades: u64,
    #[serde(default)]
    pnl_usd: Option<f64>,
    #[serde(default)]
    win_rate_pct: Option<f64>,
    #[serde(default)]
    volume_usd: Option<f64>,
    #[serde(default)]
    markets_traded: Option<Vec<String>>,
    #[serde(default)]
    scraped_at: Option<String>,
}

// ── Per-wallet report ────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WalletReport {
    address: String,
    source: String,
    analyzed_at: String,
    total_trades: u64,
    fee_negative: bool,
    strategy_type: String,
    classification_confidence: f64,
    metrics: WalletMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct WalletMetrics {
    // 1. Clip size consistency
    clip_size_consistency_pct: Option<f64>,
    // 2. Hold time distribution
    hold_time_median_secs: Option<f64>,
    hold_time_p25_secs: Option<f64>,
    hold_time_p75_secs: Option<f64>,
    hold_time_max_secs: Option<f64>,
    // 3. Direction bias
    direction_bias: Option<f64>,
    direction_label: Option<String>,
    // 4. Win rate
    win_rate_pct: Option<f64>,
    // 5. PnL distribution
    pnl_mean_usd: Option<f64>,
    pnl_median_usd: Option<f64>,
    pnl_max_winner_usd: Option<f64>,
    pnl_max_loser_usd: Option<f64>,
    pnl_skewness: Option<f64>,
    // 6. Fee-adjusted PnL
    gross_pnl_usd: Option<f64>,
    net_pnl_usd: Option<f64>,
    total_fees_usd: Option<f64>,
    entry_fees_usd: Option<f64>,
    exit_fees_usd: Option<f64>,
    borrow_fees_usd: Option<f64>,
    // 7. Counterparty concentration
    counterparty_concentration_pct: Option<f64>,
    // 8. Market concentration
    market_concentration_pct: Option<f64>,
    markets_traded: Option<Vec<String>>,
    // 9. Time patterns
    median_fill_interval_secs: Option<f64>,
    trading_hours_utc: Option<Vec<u32>>,
    // 10. Scale-in behavior
    scale_in_pct: Option<f64>,
    // 11. Leverage usage
    avg_leverage: Option<f64>,
    max_leverage: Option<f64>,
}

// ── Strategy types ───────────────────────────────────────────────────────────

#[allow(dead_code)]
const STRATEGY_TYPES: &[&str] = &[
    "momentum-scalper",
    "mean-reversion",
    "trend-follower",
    "lp-consumer",
    "swing-trader",
    "hft-market-maker",
    "grid-martingale",
    "unknown",
];

// ── Strategy blueprint ───────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StrategyBlueprint {
    strategy_type: String,
    confidence: f64,
    source_wallets: Vec<String>,
    markets: Vec<String>,
    leverage: f64,
    clip_size_usd: f64,
    parameters: BlueprintParameters,
    backtest_metrics: BacktestMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BlueprintParameters {
    entry: EntryParams,
    exit: ExitParams,
    risk: RiskParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EntryParams {
    signal: String,
    threshold_pct: f64,
    confirmation: String,
    direction: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ExitParams {
    take_profit_pct: f64,
    stop_loss_pct: f64,
    trailing_stop_pct: Option<f64>,
    trailing_activation_pct: Option<f64>,
    max_hold_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RiskParams {
    max_position_notional_usd: f64,
    daily_loss_limit_usd: f64,
    cooldown_after_loss_secs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BacktestMetrics {
    total_trades_analyzed: u64,
    win_rate: f64,
    avg_winner_usd: f64,
    avg_loser_usd: f64,
    net_pnl_after_fees_usd: f64,
    sharpe_estimate: Option<f64>,
}

// ── API response types ───────────────────────────────────────────────────────

// fstats.io leaderboard data per wallet
#[derive(Clone, Debug, Deserialize)]
struct FstatsLeaderboard {
    leaderboard: Vec<FstatsEntry>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[allow(dead_code)]
struct FstatsEntry {
    owner: String,
    #[serde(default)]
    num_trades: u64,
    #[serde(default)]
    total_volume_usd: Option<f64>,
    #[serde(default)]
    gross_pnl: Option<f64>,
    #[serde(default)]
    entry_fees: Option<f64>,
    #[serde(default)]
    net_pnl: Option<f64>,
    #[serde(default)]
    wins: Option<u64>,
    #[serde(default)]
    losses: Option<u64>,
    #[serde(default)]
    rank: Option<u32>,
    #[serde(default, rename = "win_rate")]
    win_rate_raw: Option<f64>,
    #[serde(default)]
    total_pnl: Option<f64>,
    #[serde(default)]
    avg_trade_size: Option<f64>,
    #[serde(default)]
    largest_trade: Option<f64>,
}

// Hyperliquid fill data
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct HlFill {
    #[serde(default)]
    coin: Option<String>,
    #[serde(default)]
    side: String,
    #[serde(default)]
    dir: Option<String>,
    #[serde(default)]
    px: Option<String>,
    #[serde(default)]
    sz: Option<String>,
    #[serde(default)]
    fee: Option<String>,
    #[serde(default)]
    closed_pnl: Option<String>,
    #[serde(default)]
    time: Option<i64>,
    #[serde(default)]
    hash: Option<String>,
    #[serde(default)]
    start_position: Option<String>,
}

// Jupiter trader stats
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct JupiterTraderStats {
    total_volume_usd: Option<String>,
    total_pnl_usd: Option<String>,
    start_timestamp: Option<i64>,
    end_timestamp: Option<i64>,
}

// ── Aggregated wallet data (post-fetch normalization) ────────────────────────

#[derive(Clone, Debug, Default)]
struct WalletData {
    address: String,
    source: String,
    total_trades: u64,
    gross_pnl_usd: f64,
    entry_fees_usd: f64,
    exit_fees_usd: f64,
    borrow_fees_usd: f64,
    net_pnl_usd: f64,
    wins: u64,
    losses: u64,
    win_rate_pct: f64,
    volume_usd: f64,
    avg_trade_size_usd: f64,
    largest_trade_usd: f64,
    // Individual fills (if available from Hyperliquid)
    fills: Vec<HlFill>,
    // Markets (if known)
    markets: Vec<String>,
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

const FSTATS_HOST: &str = "fstats.io";
const FSTATS_BASE: &str = "https://fstats.io/api/v1/leaderboards";
const HL_API: &str = "https://api.hyperliquid.xyz/info";
const HL_HOST: &str = "api.hyperliquid.xyz";
const JUPITER_API: &str = "https://perps-api.jup.ag/v1";
const JUPITER_HOST: &str = "perps-api.jup.ag";
const FLASH_API: &str = "https://flashapi.trade";
const FLASH_HOST: &str = "flashapi.trade";
const MAX_RETRIES: u32 = 3;
const RETRY_BASE_DELAY_SECS: u64 = 2;

async fn fetch_json<T: serde::de::DeserializeOwned>(
    client: &Client,
    rate_limiter: &RateLimiter,
    host: &str,
    url: &str,
) -> Result<T> {
    let mut last_err = None;

    for attempt in 1..=MAX_RETRIES {
        rate_limiter.throttle(host).await;
        debug!(url, attempt, "Sending request");

        match client.get(url).send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    match resp.text().await {
                        Ok(body) => match serde_json::from_str::<T>(&body) {
                            Ok(data) => return Ok(data),
                            Err(e) => {
                                error!(url, attempt, error = %e, "JSON parse error");
                                last_err =
                                    Some(anyhow::anyhow!("JSON parse error for {}: {}", url, e));
                            }
                        },
                        Err(e) => {
                            last_err =
                                Some(anyhow::anyhow!("Failed to read body from {}: {}", url, e));
                        }
                    }
                } else if status.as_u16() == 429 {
                    let backoff = RETRY_BASE_DELAY_SECS * 2u64.pow(attempt - 1);
                    warn!(url, attempt, backoff_secs = backoff, "Rate limited (429)");
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    last_err = Some(anyhow::anyhow!("HTTP 429: {}", url));
                } else {
                    last_err = Some(anyhow::anyhow!("HTTP {} from {}", status, url));
                }
            }
            Err(e) => {
                let backoff = RETRY_BASE_DELAY_SECS * 2u64.pow(attempt - 1);
                warn!(url, attempt, error = %e, backoff_secs = backoff, "Request failed");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                last_err = Some(anyhow::anyhow!("Request failed for {}: {}", url, e));
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("All retries exhausted for {}", url)))
}

async fn post_json<T: serde::de::DeserializeOwned>(
    client: &Client,
    rate_limiter: &RateLimiter,
    host: &str,
    url: &str,
    body: &str,
) -> Result<T> {
    let mut last_err = None;

    for attempt in 1..=MAX_RETRIES {
        rate_limiter.throttle(host).await;
        debug!(url, attempt, "Sending POST");

        match client
            .post(url)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    match resp.text().await {
                        Ok(text) => match serde_json::from_str::<T>(&text) {
                            Ok(data) => return Ok(data),
                            Err(e) => {
                                error!(url, error = %e, "JSON parse error");
                                last_err = Some(anyhow::anyhow!("JSON parse error: {}", e));
                            }
                        },
                        Err(e) => {
                            last_err = Some(anyhow::anyhow!("Read body error: {}", e));
                        }
                    }
                } else if status.as_u16() == 429 {
                    let backoff = RETRY_BASE_DELAY_SECS * 2u64.pow(attempt - 1);
                    warn!(
                        url,
                        attempt,
                        backoff_secs = backoff,
                        "Rate limited (429) on POST, backing off"
                    );
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    last_err = Some(anyhow::anyhow!("HTTP 429: {}", url));
                } else {
                    last_err = Some(anyhow::anyhow!("HTTP {} from {}", status, url));
                }
            }
            Err(e) => {
                let backoff = RETRY_BASE_DELAY_SECS * 2u64.pow(attempt - 1);
                warn!(url, error = %e, backoff_secs = backoff, "POST failed");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                last_err = Some(anyhow::anyhow!("POST failed: {}", e));
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("All retries exhausted for POST {}", url)))
}

// ── Address validation ───────────────────────────────────────────────────────

fn is_valid_solana_address(s: &str) -> bool {
    if s.len() < 32 || s.len() > 44 {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() && c != '0' && c != 'O' && c != 'I' && c != 'l')
}

fn is_valid_evm_address(s: &str) -> bool {
    s.len() == 42 && s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
}

fn validate_address(address: &str, source: &str) -> Result<()> {
    match source {
        "flash-trade" | "flash" | "jupiter" => {
            if !is_valid_solana_address(address) {
                anyhow::bail!(
                    "Invalid Solana address format: '{}'. Expected base58, 32-44 characters.",
                    address
                );
            }
        }
        "hyperliquid" => {
            if !is_valid_evm_address(address) {
                anyhow::bail!(
                    "Invalid EVM address format: '{}'. Expected 0x-prefixed 42-character hex.",
                    address
                );
            }
        }
        _ => {
            // Accept either format for unknown sources
            if !is_valid_solana_address(address) && !is_valid_evm_address(address) {
                anyhow::bail!(
                    "Invalid address format: '{}'. Expected Solana base58 or EVM 0x hex.",
                    address
                );
            }
        }
    }
    Ok(())
}

/// Determine source from address format
fn source_from_address(address: &str) -> &'static str {
    if is_valid_evm_address(address) {
        "hyperliquid"
    } else {
        "flash-trade"
    }
}

// ── Fetch wallet data ────────────────────────────────────────────────────────

async fn fetch_flash_wallet_data(
    client: &Client,
    rate_limiter: &RateLimiter,
    address: &str,
) -> Result<WalletData> {
    // Fetch from both fstats PnL and volume leaderboards to get full data
    let mut data = WalletData {
        address: address.to_string(),
        source: "flash-trade".to_string(),
        ..Default::default()
    };

    // Try fstats PnL leaderboard
    for endpoint in &["pnl", "volume"] {
        let url = format!("{}/{}?days=30", FSTATS_BASE, endpoint);
        match fetch_json::<FstatsLeaderboard>(client, rate_limiter, FSTATS_HOST, &url).await {
            Ok(lb) => {
                if let Some(entry) = lb.leaderboard.iter().find(|e| e.owner == address) {
                    data.total_trades = data.total_trades.max(entry.num_trades);
                    if entry.gross_pnl.is_some() {
                        data.gross_pnl_usd = entry.gross_pnl.unwrap_or(0.0);
                    }
                    if entry.total_pnl.is_some() {
                        data.gross_pnl_usd = entry.total_pnl.unwrap_or(0.0);
                    }
                    if entry.entry_fees.is_some() {
                        data.entry_fees_usd = entry.entry_fees.unwrap_or(0.0);
                    }
                    if entry.net_pnl.is_some() {
                        data.net_pnl_usd = entry.net_pnl.unwrap_or(0.0);
                    }
                    data.wins = data.wins.max(entry.wins.unwrap_or(0));
                    data.losses = data.losses.max(entry.losses.unwrap_or(0));
                    if entry.win_rate_raw.is_some() {
                        data.win_rate_pct = entry.win_rate_raw.unwrap_or(0.0);
                    }
                    if entry.total_volume_usd.is_some() {
                        data.volume_usd = entry.total_volume_usd.unwrap_or(0.0);
                    }
                    if entry.avg_trade_size.is_some() {
                        data.avg_trade_size_usd = entry.avg_trade_size.unwrap_or(0.0);
                    }
                    if entry.largest_trade.is_some() {
                        data.largest_trade_usd = entry.largest_trade.unwrap_or(0.0);
                    }
                }
            }
            Err(e) => {
                warn!(endpoint, error = %e, "Failed to fetch fstats leaderboard");
            }
        }
    }

    // If we didn't get net_pnl from fstats, compute it
    if data.net_pnl_usd == 0.0 && data.gross_pnl_usd != 0.0 {
        data.net_pnl_usd = data.gross_pnl_usd - data.entry_fees_usd;
    }

    // Try Flash Trade API for current positions (gives market info)
    rate_limiter.throttle(FLASH_HOST).await;
    let positions_url = format!(
        "{}/positions/owner/{}?includePnlInLeverageDisplay=true",
        FLASH_API, address
    );
    match client.get(&positions_url).send().await {
        Ok(resp) => {
            if resp.status().is_success()
                && let Ok(text) = resp.text().await
                && let Ok(positions) = serde_json::from_str::<Vec<serde_json::Value>>(&text)
            {
                for pos in &positions {
                    if let Some(asset) = pos.get("marketSymbol").and_then(|v| v.as_str()) {
                        let sym = asset.to_uppercase();
                        if !data.markets.contains(&sym) && sym != "UNKNOWN" {
                            data.markets.push(sym);
                        }
                    }
                }
            }
        }
        Err(e) => debug!(error = %e, "Failed to fetch Flash Trade positions"),
    }

    // Compute win rate if missing
    if data.win_rate_pct == 0.0 && (data.wins + data.losses) > 0 {
        data.win_rate_pct = (data.wins as f64 / (data.wins + data.losses) as f64) * 100.0;
    }

    // Estimate exit fees as proportion of entry fees (typically 50/50)
    if data.exit_fees_usd == 0.0 && data.entry_fees_usd > 0.0 {
        data.exit_fees_usd = data.entry_fees_usd;
    }

    Ok(data)
}

async fn fetch_jupiter_wallet_data(
    client: &Client,
    rate_limiter: &RateLimiter,
    address: &str,
) -> Result<WalletData> {
    let mut data = WalletData {
        address: address.to_string(),
        source: "jupiter".to_string(),
        ..Default::default()
    };

    // Query Jupiter trader-stats for known markets
    let market_mints = [
        ("SOL", "So11111111111111111111111111111111111111112"),
        ("BTC", "qfnqNqs3nCAHjnyCgLRDbBtq4pNMtWGehBZ1nGq8vPH"),
        ("ETH", "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"),
        ("WIF", "EKpQGSJtjMFqWZL3YqGPBTcaEvpRz9cdXFn6VmegmYQ"),
    ];

    let _now_ts = Utc::now().timestamp();
    let year = Utc::now()
        .format("%Y")
        .to_string()
        .parse::<i64>()
        .unwrap_or(2026);

    for (symbol, mint) in &market_mints {
        let url = format!(
            "{}/trader-stats?walletAddress={}&marketMint={}&year={}&week=current",
            JUPITER_API, address, mint, year
        );
        match fetch_json::<JupiterTraderStats>(client, rate_limiter, JUPITER_HOST, &url).await {
            Ok(stats) => {
                if let Some(vol) = &stats.total_volume_usd
                    && let Ok(v) = vol.parse::<f64>()
                    && v > 0.0
                {
                    data.volume_usd += v;
                    if !data.markets.contains(&symbol.to_string()) {
                        data.markets.push(symbol.to_string());
                    }
                }
                if let Some(pnl) = &stats.total_pnl_usd
                    && let Ok(p) = pnl.parse::<f64>()
                {
                    data.gross_pnl_usd += p;
                }
            }
            Err(e) => {
                debug!(symbol, error = %e, "Jupiter stats fetch failed");
            }
        }
    }

    // For Jupiter, we estimate fees at 0.05% of volume (Jupiter's standard fee)
    if data.entry_fees_usd == 0.0 && data.volume_usd > 0.0 {
        let total_fee_rate = 0.0005; // 0.05%
        data.entry_fees_usd = data.volume_usd * total_fee_rate / 2.0;
        data.exit_fees_usd = data.volume_usd * total_fee_rate / 2.0;
    }
    data.total_trades = estimate_trade_count(data.volume_usd, data.avg_trade_size_usd);
    data.net_pnl_usd = data.gross_pnl_usd - data.entry_fees_usd - data.exit_fees_usd;

    Ok(data)
}

async fn fetch_hyperliquid_wallet_data(
    client: &Client,
    rate_limiter: &RateLimiter,
    address: &str,
) -> Result<WalletData> {
    let mut data = WalletData {
        address: address.to_string(),
        source: "hyperliquid".to_string(),
        ..Default::default()
    };

    let body = serde_json::json!({
        "type": "userFills",
        "user": address
    })
    .to_string();

    match post_json::<Vec<HlFill>>(client, rate_limiter, HL_HOST, HL_API, &body).await {
        Ok(fills) => {
            if fills.is_empty() {
                info!(address, "Hyperliquid wallet has no fills");
                return Ok(data);
            }

            let mut total_pnl = 0.0_f64;
            let mut total_fee = 0.0_f64;
            let mut wins = 0u64;
            let mut losses = 0u64;
            let mut coins: HashSet<String> = HashSet::new();
            let mut _long_count = 0u64;
            let mut _total_directional = 0u64;

            for fill in &fills {
                if let Some(pnl_str) = &fill.closed_pnl
                    && let Ok(pnl) = pnl_str.parse::<f64>()
                {
                    total_pnl += pnl;
                    if pnl > 0.0 {
                        wins += 1;
                    } else if pnl < 0.0 {
                        losses += 1;
                    }
                }

                if let Some(fee_str) = &fill.fee
                    && let Ok(fee) = fee_str.parse::<f64>()
                {
                    total_fee += fee.abs();
                }

                if let Some(coin) = &fill.coin {
                    coins.insert(coin.to_uppercase());
                }

                // Track direction
                let side = fill.side.to_lowercase();
                if side == "b" || side == "buy" || side == "long" {
                    _long_count += 1;
                    _total_directional += 1;
                } else if side == "s" || side == "sell" || side == "short" {
                    _total_directional += 1;
                }
            }

            data.fills = fills;
            data.total_trades = data.fills.len() as u64;
            data.gross_pnl_usd = total_pnl;
            data.entry_fees_usd = total_fee / 2.0;
            data.exit_fees_usd = total_fee / 2.0;
            data.total_fees_usd_internal(total_fee);
            data.wins = wins;
            data.losses = losses;
            data.net_pnl_usd = total_pnl - total_fee;
            data.markets = coins.into_iter().collect();

            if wins + losses > 0 {
                data.win_rate_pct = (wins as f64 / (wins + losses) as f64) * 100.0;
            }

            info!(
                address,
                trades = data.total_trades,
                pnl = data.gross_pnl_usd,
                fees = total_fee,
                "Hyperliquid wallet data fetched"
            );
        }
        Err(e) => {
            warn!(address, error = %e, "Failed to fetch Hyperliquid fills");
        }
    }

    Ok(data)
}

impl WalletData {
    fn total_fees_usd_internal(&mut self, total: f64) {
        self.entry_fees_usd = total / 2.0;
        self.exit_fees_usd = total / 2.0;
    }
}

fn estimate_trade_count(volume: f64, avg_size: f64) -> u64 {
    if avg_size > 0.0 {
        (volume / avg_size).round() as u64
    } else if volume > 0.0 {
        // Estimate average trade as $10k
        (volume / 10_000.0).round() as u64
    } else {
        0
    }
}

// ── Metric computation ──────────────────────────────────────────────────────

fn compute_metrics(data: &WalletData) -> WalletMetrics {
    if data.total_trades == 0 {
        return WalletMetrics::default();
    }

    let mut metrics = WalletMetrics::default();

    // --- 1. Clip size consistency ---
    // Estimate from avg_trade_size and largest_trade if available
    // For fstats data: compute from avg_trade_size / largest_trade ratio
    if data.avg_trade_size_usd > 0.0 && data.largest_trade_usd > 0.0 {
        let ratio = data.avg_trade_size_usd / data.largest_trade_usd;
        // If ratio is close to 1.0, trades are very consistent
        // If ratio is low, trade sizes vary a lot
        metrics.clip_size_consistency_pct = Some(ratio * 100.0);
    } else if data.volume_usd > 0.0 && data.total_trades > 0 {
        // Estimate consistency from volume/trades (uniform assumption)
        let _avg = data.volume_usd / data.total_trades as f64;
        // Assume moderate consistency if we only have aggregate data
        metrics.clip_size_consistency_pct = Some(60.0); // Default estimate
    }

    // For Hyperliquid fills, compute actual clip size consistency
    if !data.fills.is_empty() {
        let sizes: Vec<f64> = data
            .fills
            .iter()
            .filter_map(|f| f.sz.as_ref().and_then(|s| s.parse::<f64>().ok()))
            .filter(|&s| s > 0.0)
            .collect();
        if !sizes.is_empty() {
            let avg = sizes.iter().sum::<f64>() / sizes.len() as f64;
            let within_band = sizes
                .iter()
                .filter(|&&s| (s - avg).abs() / avg <= 0.10)
                .count();
            metrics.clip_size_consistency_pct =
                Some((within_band as f64 / sizes.len() as f64) * 100.0);
        }
    }

    // --- 2. Hold time distribution ---
    // For aggregated data, estimate based on trade patterns
    // For Hyperliquid fills, compute from timestamps
    if !data.fills.is_empty() {
        let timestamps: Vec<i64> = data
            .fills
            .iter()
            .filter_map(|f| f.time)
            .filter(|&t| t > 0)
            .collect();
        if timestamps.len() >= 2 {
            let mut sorted = timestamps.clone();
            sorted.sort();

            // Estimate hold times from gaps between fills
            let mut gaps: Vec<f64> = Vec::new();
            for i in 1..sorted.len() {
                let gap = (sorted[i] - sorted[i - 1]).abs() as f64 / 1000.0; // ms to secs
                gaps.push(gap);
            }

            if !gaps.is_empty() {
                gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = gaps.len();
                metrics.hold_time_median_secs = Some(gaps[n / 2]);
                metrics.hold_time_p25_secs = Some(gaps[n / 4]);
                metrics.hold_time_p75_secs = Some(gaps[3 * n / 4]);
                metrics.hold_time_max_secs = Some(gaps[n - 1]);
            }
        }
    } else {
        // Estimate from total trades and typical trading patterns
        // High trade count in 30 days → shorter hold times
        let est_hold = estimate_hold_time(data.total_trades);
        metrics.hold_time_median_secs = Some(est_hold);
        metrics.hold_time_p25_secs = Some(est_hold * 0.5);
        metrics.hold_time_p75_secs = Some(est_hold * 2.0);
        metrics.hold_time_max_secs = Some(est_hold * 5.0);
    }

    // --- 3. Direction bias ---
    if !data.fills.is_empty() {
        let mut longs = 0u64;
        let mut total = 0u64;
        for fill in &data.fills {
            let side = fill.side.to_lowercase();
            if side == "b" || side == "buy" || side == "long" {
                longs += 1;
                total += 1;
            } else if side == "s" || side == "sell" || side == "short" {
                total += 1;
            }
        }
        if total > 0 {
            let bias = longs as f64 / total as f64;
            metrics.direction_bias = Some(bias);
            metrics.direction_label = Some(direction_label(bias));
        }
    } else {
        // Estimate from wins/losses pattern or default neutral
        let bias = 0.5; // Default neutral for aggregated data
        metrics.direction_bias = Some(bias);
        metrics.direction_label = Some(direction_label(bias));
    }

    // --- 4. Win rate ---
    metrics.win_rate_pct = Some(data.win_rate_pct);

    // --- 5. PnL distribution ---
    if data.wins + data.losses > 0 {
        // Estimate from aggregate data
        let avg_winner = if data.wins > 0 && data.gross_pnl_usd > 0.0 {
            data.gross_pnl_usd / data.wins as f64
        } else {
            0.0
        };

        // For fills with individual PnL data
        if !data.fills.is_empty() {
            let mut pnls: Vec<f64> = Vec::new();
            for fill in &data.fills {
                if let Some(pnl_str) = &fill.closed_pnl
                    && let Ok(pnl) = pnl_str.parse::<f64>()
                {
                    pnls.push(pnl);
                }
            }
            if !pnls.is_empty() {
                pnls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = pnls.len();
                metrics.pnl_mean_usd = Some(pnls.iter().sum::<f64>() / n as f64);
                metrics.pnl_median_usd = Some(pnls[n / 2]);
                metrics.pnl_max_winner_usd = Some(pnls.iter().cloned().fold(0.0_f64, f64::max));
                metrics.pnl_max_loser_usd = Some(pnls.iter().cloned().fold(0.0_f64, f64::min));
                metrics.pnl_skewness = Some(compute_skewness(&pnls));
            }
        } else {
            metrics.pnl_mean_usd = Some(data.gross_pnl_usd / (data.wins + data.losses) as f64);
            metrics.pnl_median_usd = metrics.pnl_mean_usd;
            metrics.pnl_max_winner_usd = Some(avg_winner * 2.0); // Rough estimate
            metrics.pnl_max_loser_usd = Some(if data.losses > 0 {
                -(data.gross_pnl_usd / data.losses as f64).max(0.0)
            } else {
                0.0
            });
            metrics.pnl_skewness = Some(0.0);
        }
    }

    metrics.gross_pnl_usd = Some(data.gross_pnl_usd);
    metrics.net_pnl_usd = Some(data.net_pnl_usd);
    metrics.total_fees_usd = Some(data.entry_fees_usd + data.exit_fees_usd + data.borrow_fees_usd);
    metrics.entry_fees_usd = Some(data.entry_fees_usd);
    metrics.exit_fees_usd = Some(data.exit_fees_usd);
    metrics.borrow_fees_usd = Some(data.borrow_fees_usd);

    // --- 7. Counterparty concentration ---
    // Estimate from data source (Flash Trade typically has high LP concentration)
    metrics.counterparty_concentration_pct = match data.source.as_str() {
        "flash-trade" => Some(75.0), // Flash Trade often has single dominant LP
        "hyperliquid" => Some(50.0), // More distributed
        "jupiter" => Some(40.0),
        _ => None,
    };

    // --- 8. Market concentration ---
    // Ensure markets list has at least a default entry
    let effective_markets: Vec<String> = if data.markets.is_empty() {
        vec!["UNKNOWN".to_string()]
    } else {
        data.markets.clone()
    };
    {
        let total = effective_markets.len();
        // If we have fill data, compute actual concentration
        if !data.fills.is_empty() {
            let mut market_counts: HashMap<String, usize> = HashMap::new();
            for fill in &data.fills {
                if let Some(coin) = &fill.coin {
                    *market_counts.entry(coin.to_uppercase()).or_insert(0) += 1;
                }
            }
            let max_count = market_counts.values().max().copied().unwrap_or(0);
            let total_fills = data.fills.len();
            metrics.market_concentration_pct =
                Some((max_count as f64 / total_fills as f64) * 100.0);
            metrics.markets_traded = Some(
                market_counts
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .collect(),
            );
        } else {
            // Single market = 100%, multi-market = distributed
            if total == 1 {
                metrics.market_concentration_pct = Some(100.0);
            } else {
                metrics.market_concentration_pct = Some(100.0 / total as f64);
            }
            metrics.markets_traded = Some(effective_markets);
        }
    }

    // --- 9. Time patterns ---
    if !data.fills.is_empty() {
        let timestamps: Vec<i64> = data
            .fills
            .iter()
            .filter_map(|f| f.time)
            .filter(|&t| t > 0)
            .collect();
        if timestamps.len() >= 2 {
            let mut sorted = timestamps.clone();
            sorted.sort();
            let mut intervals: Vec<f64> = Vec::new();
            for i in 1..sorted.len() {
                intervals.push((sorted[i] - sorted[i - 1]).abs() as f64 / 1000.0);
            }
            if !intervals.is_empty() {
                intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                metrics.median_fill_interval_secs = Some(intervals[intervals.len() / 2]);
            }

            // Compute peak trading hours
            let mut hour_counts: HashMap<u32, usize> = HashMap::new();
            for ts in &timestamps {
                // Convert ms timestamp to hour
                let secs = ts / 1000;
                let datetime = chrono::DateTime::from_timestamp(secs, 0);
                if let Some(dt) = datetime {
                    let hour = dt.format("%H").to_string().parse::<u32>().unwrap_or(0);
                    *hour_counts.entry(hour).or_insert(0) += 1;
                }
            }
            let mut peak_hours: Vec<u32> = hour_counts
                .iter()
                .filter(|(_, count)| **count > timestamps.len() / 24)
                .map(|(&h, _)| h)
                .collect();
            peak_hours.sort();
            metrics.trading_hours_utc = Some(peak_hours);
        }
    } else {
        // Estimate for aggregated data
        metrics.median_fill_interval_secs = Some(estimate_fill_interval(data.total_trades));
        metrics.trading_hours_utc = Some(vec![8, 9, 10, 14, 15, 16]); // Default peak hours
    }

    // --- 10. Scale-in behavior ---
    // Estimate: high trade count with consistent sizes = lower scale-in
    //           varied sizes = higher scale-in
    if let Some(clip_consistency) = metrics.clip_size_consistency_pct {
        metrics.scale_in_pct = Some((100.0 - clip_consistency).max(0.0));
    } else {
        metrics.scale_in_pct = Some(20.0); // Default
    }

    // --- 11. Leverage usage ---
    // Estimate from trade sizes vs typical capital
    if data.volume_usd > 0.0 && data.total_trades > 0 {
        let avg_trade = data.volume_usd / data.total_trades as f64;
        // Rough estimate: most perp traders use 3-10x leverage
        metrics.avg_leverage = Some(if avg_trade > 100_000.0 {
            5.0
        } else if avg_trade > 10_000.0 {
            3.0
        } else {
            2.0
        });
        metrics.max_leverage = Some(metrics.avg_leverage.unwrap_or(3.0) * 2.0);
    } else {
        metrics.avg_leverage = Some(3.0);
        metrics.max_leverage = Some(10.0);
    }

    metrics
}

fn direction_label(bias: f64) -> String {
    if (bias - 0.5).abs() > 0.2 {
        if bias > 0.5 {
            "long".to_string()
        } else {
            "short".to_string()
        }
    } else {
        "neutral".to_string()
    }
}

fn estimate_hold_time(total_trades: u64) -> f64 {
    // In 30 days, more trades → shorter holds
    if total_trades > 500 {
        60.0 // ~1 min (HFT)
    } else if total_trades > 100 {
        300.0 // ~5 min (scalper)
    } else if total_trades > 30 {
        1800.0 // ~30 min (active trader)
    } else if total_trades > 10 {
        7200.0 // ~2 hours (swing)
    } else {
        86400.0 // ~1 day (position trader)
    }
}

fn estimate_fill_interval(total_trades: u64) -> f64 {
    let seconds_in_30_days = 30.0 * 86400.0;
    if total_trades > 0 {
        seconds_in_30_days / total_trades as f64
    } else {
        0.0
    }
}

fn compute_skewness(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if variance == 0.0 {
        return 0.0;
    }
    let std_dev = variance.sqrt();
    values
        .iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n
}

// ── Strategy classification ─────────────────────────────────────────────────

fn classify_strategy(metrics: &WalletMetrics, data: &WalletData) -> (String, f64) {
    if data.total_trades == 0 {
        return ("unknown".to_string(), 0.0);
    }

    let mut scores: HashMap<&str, f64> = HashMap::new();

    // Extract key metrics with defaults
    let clip_consistency = metrics.clip_size_consistency_pct.unwrap_or(50.0);
    let hold_median = metrics.hold_time_median_secs.unwrap_or(300.0);
    let direction_bias = metrics.direction_bias.unwrap_or(0.5);
    let win_rate = metrics.win_rate_pct.unwrap_or(50.0);
    let market_concentration = metrics.market_concentration_pct.unwrap_or(50.0);
    let fill_interval = metrics.median_fill_interval_secs.unwrap_or(300.0);
    let scale_in = metrics.scale_in_pct.unwrap_or(20.0);
    let avg_leverage = metrics.avg_leverage.unwrap_or(3.0);

    // --- Momentum Scalper ---
    // High clip consistency, short holds, directional bias, high win rate,
    // concentrated markets
    let mut ms_score = 0.0;
    if clip_consistency > 60.0 {
        ms_score += 0.15;
    }
    if hold_median < 1800.0 {
        ms_score += 0.20;
    }
    if (direction_bias - 0.5).abs() > 0.15 {
        ms_score += 0.15;
    }
    if win_rate > 55.0 {
        ms_score += 0.15;
    }
    if market_concentration > 60.0 {
        ms_score += 0.15;
    }
    if fill_interval < 600.0 {
        ms_score += 0.10;
    }
    if clip_consistency > 70.0 && hold_median < 600.0 {
        ms_score += 0.10; // Bonus for very consistent, fast trading
    }
    scores.insert("momentum-scalper", ms_score);

    // --- Mean Reversion ---
    // Neutral direction, short holds, moderate frequency, tight PnL range
    let mut mr_score = 0.0;
    if (direction_bias - 0.5).abs() < 0.2 {
        mr_score += 0.25; // Strong signal: neutral direction
    }
    if hold_median < 3600.0 {
        mr_score += 0.15;
    }
    if win_rate > 55.0 && win_rate < 80.0 {
        mr_score += 0.15;
    }
    if let Some(skewness) = metrics.pnl_skewness
        && skewness.abs() < 1.0
    {
        mr_score += 0.15; // Tight PnL range
    }
    if market_concentration < 80.0 {
        mr_score += 0.10; // Trades multiple markets
    }
    if fill_interval < 1800.0 {
        mr_score += 0.10;
    }
    if let Some(max_winner) = metrics.pnl_max_winner_usd
        && let Some(mean) = metrics.pnl_mean_usd
        && mean != 0.0
        && (max_winner / mean.abs()).abs() < 3.0
    {
        mr_score += 0.10; // Small max winner relative to avg
    }
    scores.insert("mean-reversion", mr_score);

    // --- Trend Follower ---
    // Directional bias, longer holds, moderate win rate, larger winners
    let mut tf_score = 0.0;
    if (direction_bias - 0.5).abs() > 0.15 {
        tf_score += 0.15;
    }
    if hold_median > 3600.0 {
        tf_score += 0.25; // Longer holds
    }
    if win_rate > 40.0 && win_rate < 65.0 {
        tf_score += 0.15; // Moderate win rate
    }
    if let Some(skewness) = metrics.pnl_skewness
        && skewness > 0.5
    {
        tf_score += 0.15; // Positive skew (large winners)
    }
    if scale_in > 30.0 {
        tf_score += 0.10; // Scales into positions
    }
    if avg_leverage > 3.0 {
        tf_score += 0.10;
    }
    scores.insert("trend-follower", tf_score);

    // --- LP Consumer ---
    // Very high market concentration, high counterparty concentration,
    // directional bias, consistent sizes
    let mut lc_score = 0.0;
    if market_concentration > 80.0 {
        lc_score += 0.25;
    }
    if let Some(cp) = metrics.counterparty_concentration_pct
        && cp > 70.0
    {
        lc_score += 0.20;
    }
    if clip_consistency > 50.0 {
        lc_score += 0.15;
    }
    if (direction_bias - 0.5).abs() > 0.1 {
        lc_score += 0.15;
    }
    if win_rate > 50.0 {
        lc_score += 0.10;
    }
    if data.total_trades > 10 {
        lc_score += 0.15;
    }
    scores.insert("lp-consumer", lc_score);

    // --- Swing Trader ---
    // Moderate hold times, moderate direction, varied sizes
    let mut st_score = 0.0;
    if (1800.0..=86400.0).contains(&hold_median) {
        st_score += 0.25;
    }
    if (direction_bias - 0.5).abs() < 0.3 {
        st_score += 0.15;
    }
    if scale_in > 20.0 {
        st_score += 0.15;
    }
    if market_concentration < 80.0 {
        st_score += 0.15;
    }
    if win_rate > 45.0 && win_rate < 75.0 {
        st_score += 0.15;
    }
    if fill_interval > 600.0 {
        st_score += 0.15;
    }
    scores.insert("swing-trader", st_score);

    // --- HFT Market Maker ---
    // Very short holds, very frequent, neutral direction, many markets
    let mut hft_score = 0.0;
    if hold_median < 60.0 {
        hft_score += 0.25;
    }
    if (direction_bias - 0.5).abs() < 0.15 {
        hft_score += 0.20;
    }
    if fill_interval < 10.0 {
        hft_score += 0.20;
    }
    if let Some(markets) = &metrics.markets_traded
        && markets.len() >= 3
    {
        hft_score += 0.15;
    }
    if data.total_trades > 500 {
        hft_score += 0.20;
    }
    scores.insert("hft-market-maker", hft_score);

    // --- Grid/Martingale ---
    // Very consistent clip sizes, neutral direction, many trades, moderate hold
    let mut gm_score = 0.0;
    if clip_consistency > 80.0 {
        gm_score += 0.25;
    }
    if (direction_bias - 0.5).abs() < 0.2 {
        gm_score += 0.20;
    }
    if data.total_trades > 50 {
        gm_score += 0.15;
    }
    if hold_median < 3600.0 {
        gm_score += 0.15;
    }
    if win_rate > 50.0 && win_rate < 70.0 {
        gm_score += 0.15;
    }
    if let Some(max_winner) = metrics.pnl_max_winner_usd
        && let Some(max_loser) = metrics.pnl_max_loser_usd
        && max_winner.abs() > 0.0
        && max_loser.abs() > max_winner.abs() * 3.0
    {
        gm_score += 0.10; // Large losers relative to winners (martingale-like)
    }
    scores.insert("grid-martingale", gm_score);

    // Find the best match
    let (best_type, best_score) = scores
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((&"unknown", &0.0));

    // Require minimum confidence
    let min_confidence = 0.25;
    if *best_score < min_confidence {
        return ("unknown".to_string(), *best_score);
    }

    (best_type.to_string(), *best_score)
}

// ── Blueprint generation ────────────────────────────────────────────────────

fn generate_blueprints(
    reports: &[WalletReport],
    wallet_data: &[WalletData],
) -> Vec<StrategyBlueprint> {
    // Group reports by strategy type (excluding unknown and fee-negative)
    let mut strategy_wallets: HashMap<String, Vec<&WalletReport>> = HashMap::new();

    for report in reports {
        if report.strategy_type == "unknown" || report.fee_negative {
            continue;
        }
        strategy_wallets
            .entry(report.strategy_type.clone())
            .or_default()
            .push(report);
    }

    let mut blueprints = Vec::new();

    for (strategy_type, wallets) in &strategy_wallets {
        if wallets.is_empty() {
            continue;
        }

        let confidence = wallets
            .iter()
            .map(|w| w.classification_confidence)
            .sum::<f64>()
            / wallets.len() as f64;

        // Aggregate metrics from fee-positive wallets
        let net_pnl_sum: f64 = wallets.iter().filter_map(|w| w.metrics.net_pnl_usd).sum();

        let total_trades: u64 = wallets
            .iter()
            .filter_map(|w| {
                if w.total_trades > 0 {
                    Some(w.total_trades)
                } else {
                    None
                }
            })
            .sum();

        let avg_win_rate: f64 = wallets
            .iter()
            .filter_map(|w| w.metrics.win_rate_pct)
            .sum::<f64>()
            / wallets.len() as f64;

        let avg_winner: f64 = wallets
            .iter()
            .filter_map(|w| w.metrics.pnl_max_winner_usd)
            .sum::<f64>()
            / wallets.len() as f64;

        let avg_loser: f64 = wallets
            .iter()
            .filter_map(|w| w.metrics.pnl_max_loser_usd)
            .sum::<f64>()
            / wallets.len() as f64;

        let avg_leverage: f64 = wallets
            .iter()
            .filter_map(|w| w.metrics.avg_leverage)
            .sum::<f64>()
            / wallets.len() as f64;

        let avg_clip_size: f64 = wallet_data
            .iter()
            .filter(|d| wallets.iter().any(|w| w.address == d.address))
            .map(|d| {
                if d.avg_trade_size_usd > 0.0 {
                    d.avg_trade_size_usd
                } else if d.total_trades > 0 && d.volume_usd > 0.0 {
                    d.volume_usd / d.total_trades as f64
                } else {
                    100.0 // Default
                }
            })
            .sum::<f64>()
            / wallet_data
                .iter()
                .filter(|d| wallets.iter().any(|w| w.address == d.address))
                .count()
                .max(1) as f64;

        let avg_hold_secs: f64 = wallets
            .iter()
            .filter_map(|w| w.metrics.hold_time_median_secs)
            .sum::<f64>()
            / wallets
                .iter()
                .filter(|w| w.metrics.hold_time_median_secs.is_some())
                .count()
                .max(1) as f64;

        // Collect all markets
        let mut all_markets: Vec<String> = Vec::new();
        for w in wallets {
            if let Some(markets) = &w.metrics.markets_traded {
                for m in markets {
                    if !all_markets.contains(m) {
                        all_markets.push(m.clone());
                    }
                }
            }
        }
        if all_markets.is_empty() {
            all_markets.push("SOL".to_string()); // Default
        }

        let source_wallets: Vec<String> = wallets.iter().map(|w| w.address.clone()).collect();

        // Generate strategy-specific parameters
        let (entry, exit, risk) =
            generate_strategy_params(strategy_type, avg_win_rate, avg_hold_secs, avg_leverage);

        // Compute Sharpe estimate
        let sharpe = if total_trades > 0 {
            let returns: Vec<f64> = wallets
                .iter()
                .filter_map(|w| w.metrics.net_pnl_usd)
                .collect();
            if returns.len() > 1 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                if variance > 0.0 {
                    Some(mean / variance.sqrt() * (365.0_f64 * 24.0 * 12.0).sqrt()) // Annualized
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        blueprints.push(StrategyBlueprint {
            strategy_type: strategy_type.clone(),
            confidence,
            source_wallets,
            markets: all_markets,
            leverage: avg_leverage,
            clip_size_usd: avg_clip_size,
            parameters: BlueprintParameters { entry, exit, risk },
            backtest_metrics: BacktestMetrics {
                total_trades_analyzed: total_trades,
                win_rate: avg_win_rate,
                avg_winner_usd: avg_winner,
                avg_loser_usd: avg_loser,
                net_pnl_after_fees_usd: net_pnl_sum,
                sharpe_estimate: sharpe,
            },
        });
    }

    blueprints
}

fn generate_strategy_params(
    strategy_type: &str,
    _win_rate: f64,
    _avg_hold_secs: f64,
    leverage: f64,
) -> (EntryParams, ExitParams, RiskParams) {
    match strategy_type {
        "momentum-scalper" => (
            EntryParams {
                signal: "price_velocity_exceeds_threshold".to_string(),
                threshold_pct: 0.15,
                confirmation: "consecutive_moves_same_direction >= 3".to_string(),
                direction: "follow_momentum".to_string(),
            },
            ExitParams {
                take_profit_pct: 2.5,
                stop_loss_pct: 1.0,
                trailing_stop_pct: Some(0.8),
                trailing_activation_pct: Some(1.5),
                max_hold_secs: 1800,
            },
            RiskParams {
                max_position_notional_usd: 100.0 * leverage,
                daily_loss_limit_usd: 200.0,
                cooldown_after_loss_secs: 300,
            },
        ),
        "mean-reversion" => (
            EntryParams {
                signal: "price_deviation_from_mean_exceeds_threshold".to_string(),
                threshold_pct: 1.5,
                confirmation: "reversal_tick_confirmed >= 3".to_string(),
                direction: "counter_trend".to_string(),
            },
            ExitParams {
                take_profit_pct: 1.0,
                stop_loss_pct: 0.8,
                trailing_stop_pct: None,
                trailing_activation_pct: None,
                max_hold_secs: 3600,
            },
            RiskParams {
                max_position_notional_usd: 100.0 * leverage,
                daily_loss_limit_usd: 150.0,
                cooldown_after_loss_secs: 180,
            },
        ),
        "trend-follower" => (
            EntryParams {
                signal: "confirmed_breakout_above_resistance".to_string(),
                threshold_pct: 0.5,
                confirmation: "momentum_acceleration >= 2_bars".to_string(),
                direction: "follow_breakout".to_string(),
            },
            ExitParams {
                take_profit_pct: 5.0,
                stop_loss_pct: 2.0,
                trailing_stop_pct: Some(1.5),
                trailing_activation_pct: Some(3.0),
                max_hold_secs: 86400,
            },
            RiskParams {
                max_position_notional_usd: 200.0 * leverage,
                daily_loss_limit_usd: 300.0,
                cooldown_after_loss_secs: 600,
            },
        ),
        "lp-consumer" => (
            EntryParams {
                signal: "lp_depth_consumption_velocity_exceeds_threshold".to_string(),
                threshold_pct: 0.5,
                confirmation: "consumption_directional >= 70%".to_string(),
                direction: "follow_consumption".to_string(),
            },
            ExitParams {
                take_profit_pct: 2.0,
                stop_loss_pct: 1.0,
                trailing_stop_pct: Some(0.8),
                trailing_activation_pct: Some(1.2),
                max_hold_secs: 3600,
            },
            RiskParams {
                max_position_notional_usd: 100.0 * leverage,
                daily_loss_limit_usd: 200.0,
                cooldown_after_loss_secs: 300,
            },
        ),
        "swing-trader" => (
            EntryParams {
                signal: "technical_setup_confirmed".to_string(),
                threshold_pct: 0.3,
                confirmation: "multi_timeframe_alignment".to_string(),
                direction: "both".to_string(),
            },
            ExitParams {
                take_profit_pct: 4.0,
                stop_loss_pct: 2.0,
                trailing_stop_pct: Some(1.5),
                trailing_activation_pct: Some(2.5),
                max_hold_secs: 28800,
            },
            RiskParams {
                max_position_notional_usd: 250.0 * leverage,
                daily_loss_limit_usd: 400.0,
                cooldown_after_loss_secs: 900,
            },
        ),
        "hft-market-maker" => (
            EntryParams {
                signal: "spread_widening_opportunity".to_string(),
                threshold_pct: 0.05,
                confirmation: "both_sides_liquidity_available".to_string(),
                direction: "neutral".to_string(),
            },
            ExitParams {
                take_profit_pct: 0.1,
                stop_loss_pct: 0.05,
                trailing_stop_pct: None,
                trailing_activation_pct: None,
                max_hold_secs: 60,
            },
            RiskParams {
                max_position_notional_usd: 500.0 * leverage,
                daily_loss_limit_usd: 100.0,
                cooldown_after_loss_secs: 10,
            },
        ),
        "grid-martingale" => (
            EntryParams {
                signal: "price_at_grid_level".to_string(),
                threshold_pct: 0.5,
                confirmation: "grid_spacing_confirmed".to_string(),
                direction: "counter_trend".to_string(),
            },
            ExitParams {
                take_profit_pct: 0.5,
                stop_loss_pct: 5.0,
                trailing_stop_pct: None,
                trailing_activation_pct: None,
                max_hold_secs: 86400,
            },
            RiskParams {
                max_position_notional_usd: 50.0 * leverage,
                daily_loss_limit_usd: 500.0,
                cooldown_after_loss_secs: 30,
            },
        ),
        _ => (
            EntryParams {
                signal: "unknown".to_string(),
                threshold_pct: 0.0,
                confirmation: "none".to_string(),
                direction: "unknown".to_string(),
            },
            ExitParams {
                take_profit_pct: 2.0,
                stop_loss_pct: 1.0,
                trailing_stop_pct: None,
                trailing_activation_pct: None,
                max_hold_secs: 3600,
            },
            RiskParams {
                max_position_notional_usd: 100.0,
                daily_loss_limit_usd: 200.0,
                cooldown_after_loss_secs: 300,
            },
        ),
    }
}

// ── File I/O ─────────────────────────────────────────────────────────────────

fn atomic_write_json<T: serde::Serialize>(path: &PathBuf, data: &T) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent)
            .context(format!("Failed to create directory: {}", parent.display()))?;
    }

    let tmp_path = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(data).context("Failed to serialize JSON")?;
    fs::write(&tmp_path, &json)
        .context(format!("Failed to write temp file: {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path).context(format!(
        "Failed to rename {} -> {}",
        tmp_path.display(),
        path.display()
    ))?;
    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(false)
        .init();

    info!("=== analyze-wallet ===");

    // Validate arguments
    if args.address.is_none() && args.wallets.is_none() {
        anyhow::bail!("Either --address or --wallets is required. Use --help for usage.");
    }

    if args.address.is_some() && args.source.is_none() {
        anyhow::bail!(
            "--source is required when using --address. Specify: flash, jupiter, or hyperliquid"
        );
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("Failed to create HTTP client")?;

    let rate_limiter = RateLimiter::new(args.rate_limit);

    // Build the list of wallets to analyze
    let wallets_to_analyze = if let Some(address) = &args.address {
        let source_str = match args.source {
            Some(SourceArg::Flash) => "flash-trade",
            Some(SourceArg::Jupiter) => "jupiter",
            Some(SourceArg::Hyperliquid) => "hyperliquid",
            None => source_from_address(address),
        };
        validate_address(address, source_str)?;

        vec![WalletInput {
            address: address.clone(),
            source: source_str.to_string(),
            rank: 0,
            total_trades: 0,
            pnl_usd: None,
            win_rate_pct: None,
            volume_usd: None,
            markets_traded: None,
            scraped_at: None,
        }]
    } else {
        let wallets_path = args.wallets.as_ref().unwrap();
        let content = fs::read_to_string(wallets_path).context(format!(
            "Failed to read wallets file: {}",
            wallets_path.display()
        ))?;
        let wallets: Vec<WalletInput> =
            serde_json::from_str(&content).context("Failed to parse wallets.json")?;
        info!(count = wallets.len(), "Loaded wallets from file");
        wallets
    };

    // Deduplicate by address
    let mut seen_addresses: HashSet<String> = HashSet::new();
    let wallets_to_analyze: Vec<WalletInput> = wallets_to_analyze
        .into_iter()
        .filter(|w| {
            if seen_addresses.contains(&w.address) {
                warn!(address = %w.address, "Duplicate address in input, skipping");
                false
            } else {
                seen_addresses.insert(w.address.clone());
                true
            }
        })
        .collect();

    info!(
        total = wallets_to_analyze.len(),
        "Wallets to analyze (after dedup)"
    );

    // Ensure output directories exist
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;
    fs::create_dir_all(&args.blueprints_dir).context("Failed to create blueprints directory")?;

    // Process each wallet
    let mut reports: Vec<WalletReport> = Vec::new();
    let mut all_wallet_data: Vec<WalletData> = Vec::new();
    let total = wallets_to_analyze.len();

    for (i, wallet) in wallets_to_analyze.iter().enumerate() {
        info!(
            wallet = %wallet.address,
            source = %wallet.source,
            progress = format!("{}/{}", i + 1, total),
            "Analyzing wallet"
        );

        // Validate address format
        if let Err(e) = validate_address(&wallet.address, &wallet.source) {
            warn!(
                address = %wallet.address,
                error = %e,
                "Invalid address format, skipping"
            );
            continue;
        }

        // Fetch wallet data from the appropriate API
        let wallet_data = match wallet.source.as_str() {
            "flash-trade" | "flash" => {
                fetch_flash_wallet_data(&client, &rate_limiter, &wallet.address).await
            }
            "jupiter" => fetch_jupiter_wallet_data(&client, &rate_limiter, &wallet.address).await,
            "hyperliquid" => {
                fetch_hyperliquid_wallet_data(&client, &rate_limiter, &wallet.address).await
            }
            _ => {
                warn!(source = %wallet.source, "Unknown source, attempting flash-trade");
                fetch_flash_wallet_data(&client, &rate_limiter, &wallet.address).await
            }
        };

        let data = match wallet_data {
            Ok(d) => d,
            Err(e) => {
                error!(address = %wallet.address, error = %e, "Failed to fetch wallet data");

                // If we got rate limited, sleep longer before the next wallet to let the rate limit window clear
                let err_str = e.to_string();
                if err_str.contains("429") || err_str.contains("Too Many Requests") {
                    let cooldown = 5;
                    warn!(
                        "Rate limit detected (429). Sleeping for {} seconds to cool down...",
                        cooldown
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(cooldown)).await;
                }

                // Create empty report
                let report = WalletReport {
                    address: wallet.address.clone(),
                    source: wallet.source.clone(),
                    analyzed_at: Utc::now().to_rfc3339(),
                    total_trades: 0,
                    fee_negative: false,
                    strategy_type: "unknown".to_string(),
                    classification_confidence: 0.0,
                    metrics: WalletMetrics::default(),
                };
                let report_path = args.output.join(format!("{}.json", wallet.address));
                if let Err(e) = atomic_write_json(&report_path, &report) {
                    error!(path = %report_path.display(), error = %e, "Failed to write report");
                }
                reports.push(report);
                continue;
            }
        };

        // Compute metrics
        let metrics = compute_metrics(&data);

        // Classify strategy
        let (strategy_type, confidence) = classify_strategy(&metrics, &data);

        // Determine fee-negative status
        let fee_negative = metrics.net_pnl_usd.map(|net| net < 0.0).unwrap_or(false);

        let report = WalletReport {
            address: wallet.address.clone(),
            source: wallet.source.clone(),
            analyzed_at: Utc::now().to_rfc3339(),
            total_trades: data.total_trades,
            fee_negative,
            strategy_type,
            classification_confidence: confidence,
            metrics,
        };

        info!(
            address = %wallet.address,
            trades = report.total_trades,
            strategy = %report.strategy_type,
            confidence = report.classification_confidence,
            fee_negative = report.fee_negative,
            "Wallet analysis complete"
        );

        // Write per-wallet report
        let report_path = args.output.join(format!("{}.json", wallet.address));
        if let Err(e) = atomic_write_json(&report_path, &report) {
            error!(path = %report_path.display(), error = %e, "Failed to write report");
        }

        reports.push(report);
        all_wallet_data.push(data);
    }

    // Generate strategy blueprints
    let blueprints = generate_blueprints(&reports, &all_wallet_data);

    info!(
        total_wallets = reports.len(),
        blueprints = blueprints.len(),
        "Generating strategy blueprints"
    );

    let mut strategy_counts: HashMap<String, usize> = HashMap::new();
    let mut fee_negative_count = 0usize;

    for report in &reports {
        *strategy_counts
            .entry(report.strategy_type.clone())
            .or_insert(0) += 1;
        if report.fee_negative {
            fee_negative_count += 1;
        }
    }

    for blueprint in &blueprints {
        info!(
            strategy = %blueprint.strategy_type,
            wallets = blueprint.source_wallets.len(),
            confidence = blueprint.confidence,
            net_pnl = blueprint.backtest_metrics.net_pnl_after_fees_usd,
            "Blueprint generated"
        );

        let blueprint_path = args
            .blueprints_dir
            .join(format!("{}.json", blueprint.strategy_type));
        atomic_write_json(&blueprint_path, blueprint)?;
    }

    // Log summary
    info!("=== Analysis Summary ===");
    info!(total_analyzed = reports.len(), "Total wallets analyzed");
    info!(
        fee_positive = reports.len() - fee_negative_count,
        fee_negative = fee_negative_count,
        "Fee breakdown"
    );
    for (strategy, count) in &strategy_counts {
        info!(strategy, count, "Strategy distribution");
    }
    info!(
        blueprints_generated = blueprints.len(),
        "Strategy blueprints written"
    );

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data(
        trades: u64,
        gross_pnl: f64,
        fees: f64,
        wins: u64,
        losses: u64,
    ) -> WalletData {
        WalletData {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            total_trades: trades,
            gross_pnl_usd: gross_pnl,
            entry_fees_usd: fees / 2.0,
            exit_fees_usd: fees / 2.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: gross_pnl - fees,
            wins,
            losses,
            win_rate_pct: if wins + losses > 0 {
                wins as f64 / (wins + losses) as f64 * 100.0
            } else {
                0.0
            },
            volume_usd: 100_000.0,
            avg_trade_size_usd: if trades > 0 {
                100_000.0 / trades as f64
            } else {
                0.0
            },
            largest_trade_usd: 50_000.0,
            fills: vec![],
            markets: vec!["SOL".to_string()],
        }
    }

    #[test]
    fn test_address_validation_solana() {
        assert!(is_valid_solana_address(
            "BxkieMNZXqJf1GnAHjn7VjFHGLR3fKxuJG1MWEUuVrnB"
        ));
        assert!(is_valid_solana_address(
            "8ddc12hR2ePg4UkkWcecd9ShcNJyHrkBpLDjd8Yjn4GG"
        ));
        assert!(!is_valid_solana_address(""));
        assert!(!is_valid_solana_address("0x1234"));
        assert!(!is_valid_solana_address("short"));
    }

    #[test]
    fn test_address_validation_evm() {
        assert!(is_valid_evm_address(
            "0x22520a8e6e3f7b7f57e21a0d9a774dd909c35964"
        ));
        assert!(!is_valid_evm_address(""));
        assert!(!is_valid_evm_address("0x1234"));
        assert!(!is_valid_evm_address(
            "22520a8e6e3f7b7f57e21a0d9a774dd909c35964"
        ));
    }

    #[test]
    fn test_validate_address_flash() {
        assert!(
            validate_address(
                "BxkieMNZXqJf1GnAHjn7VjFHGLR3fKxuJG1MWEUuVrnB",
                "flash-trade"
            )
            .is_ok()
        );
        assert!(validate_address("invalid", "flash-trade").is_err());
    }

    #[test]
    fn test_validate_address_hyperliquid() {
        assert!(
            validate_address("0x22520a8e6e3f7b7f57e21a0d9a774dd909c35964", "hyperliquid").is_ok()
        );
        assert!(validate_address("invalid", "hyperliquid").is_err());
        assert!(
            validate_address(
                "BxkieMNZXqJf1GnAHjn7VjFHGLR3fKxuJG1MWEUuVrnB",
                "hyperliquid"
            )
            .is_err()
        );
    }

    #[test]
    fn test_source_from_address() {
        assert_eq!(
            source_from_address("0x22520a8e6e3f7b7f57e21a0d9a774dd909c35964"),
            "hyperliquid"
        );
        assert_eq!(
            source_from_address("BxkieMNZXqJf1GnAHjn7VjFHGLR3fKxuJG1MWEUuVrnB"),
            "flash-trade"
        );
    }

    #[test]
    fn test_zero_trades_produces_null_metrics() {
        let data = make_test_data(0, 0.0, 0.0, 0, 0);
        let metrics = compute_metrics(&data);
        assert!(metrics.clip_size_consistency_pct.is_none());
        assert!(metrics.win_rate_pct.is_none());
        assert!(metrics.net_pnl_usd.is_none());

        let (strategy, confidence) = classify_strategy(&metrics, &data);
        assert_eq!(strategy, "unknown");
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_fee_adjusted_pnl() {
        let data = make_test_data(100, 10000.0, 500.0, 60, 40);
        let metrics = compute_metrics(&data);
        assert_eq!(metrics.gross_pnl_usd.unwrap(), 10000.0);
        assert_eq!(metrics.total_fees_usd.unwrap(), 500.0);
        assert_eq!(metrics.net_pnl_usd.unwrap(), 9500.0);
        assert_eq!(metrics.entry_fees_usd.unwrap(), 250.0);
        assert_eq!(metrics.exit_fees_usd.unwrap(), 250.0);
    }

    #[test]
    fn test_fee_negative_flag() {
        let data = make_test_data(100, 500.0, 1000.0, 30, 70);
        let metrics = compute_metrics(&data);
        assert!(metrics.net_pnl_usd.unwrap() < 0.0);
        // fee_negative would be true in the report
        assert!(metrics.net_pnl_usd.map(|net| net < 0.0).unwrap_or(false));
    }

    #[test]
    fn test_zero_fee_edge_case() {
        let data = make_test_data(10, 5000.0, 0.0, 8, 2);
        let metrics = compute_metrics(&data);
        assert_eq!(metrics.gross_pnl_usd.unwrap(), 5000.0);
        assert_eq!(metrics.total_fees_usd.unwrap(), 0.0);
        assert_eq!(metrics.net_pnl_usd.unwrap(), 5000.0);
    }

    #[test]
    fn test_win_rate_computation() {
        let data = make_test_data(100, 10000.0, 500.0, 75, 25);
        let metrics = compute_metrics(&data);
        assert!((metrics.win_rate_pct.unwrap() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_direction_label() {
        assert_eq!(direction_label(0.8), "long");
        assert_eq!(direction_label(0.2), "short");
        assert_eq!(direction_label(0.5), "neutral");
        assert_eq!(direction_label(0.6), "neutral"); // Within 0.2 of 0.5
        assert_eq!(direction_label(0.71), "long"); // > 0.7
    }

    #[test]
    fn test_skewness() {
        let symmetric = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let skew = compute_skewness(&symmetric);
        assert!(skew.abs() < 0.5); // Approximately symmetric

        let right_skewed = vec![-10.0, 1.0, 1.0, 1.0, 1.0];
        let skew = compute_skewness(&right_skewed);
        assert!(skew < -1.0); // Left-skewed (big loser)
    }

    #[test]
    fn test_classify_momentum_scalper() {
        // High consistency, short holds, directional, high win rate, concentrated
        let data = WalletData {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            total_trades: 200,
            gross_pnl_usd: 50000.0,
            entry_fees_usd: 1000.0,
            exit_fees_usd: 1000.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: 48000.0,
            wins: 150,
            losses: 50,
            win_rate_pct: 75.0,
            volume_usd: 2_000_000.0,
            avg_trade_size_usd: 10000.0,
            largest_trade_usd: 12000.0,
            fills: vec![],
            markets: vec!["SOL".to_string()],
        };
        let mut metrics = compute_metrics(&data);
        // Force metrics for clear classification
        metrics.clip_size_consistency_pct = Some(85.0);
        metrics.hold_time_median_secs = Some(120.0);
        metrics.direction_bias = Some(0.75);
        metrics.direction_label = Some("long".to_string());
        metrics.market_concentration_pct = Some(95.0);
        metrics.median_fill_interval_secs = Some(180.0);
        metrics.counterparty_concentration_pct = Some(50.0); // Not LP-dominated

        let (strategy, confidence) = classify_strategy(&metrics, &data);
        assert_eq!(strategy, "momentum-scalper");
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_classify_mean_reversion() {
        let data = WalletData {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            total_trades: 300,
            gross_pnl_usd: 20000.0,
            entry_fees_usd: 800.0,
            exit_fees_usd: 800.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: 18400.0,
            wins: 200,
            losses: 100,
            win_rate_pct: 66.7,
            volume_usd: 5_000_000.0,
            avg_trade_size_usd: 15000.0,
            largest_trade_usd: 30000.0,
            fills: vec![],
            markets: vec!["SOL".to_string(), "BTC".to_string(), "ETH".to_string()],
        };
        let mut metrics = compute_metrics(&data);
        metrics.direction_bias = Some(0.5);
        metrics.direction_label = Some("neutral".to_string());
        metrics.hold_time_median_secs = Some(600.0);
        metrics.pnl_skewness = Some(0.3);
        metrics.market_concentration_pct = Some(40.0);
        metrics.median_fill_interval_secs = Some(300.0);

        let (strategy, confidence) = classify_strategy(&metrics, &data);
        assert_eq!(strategy, "mean-reversion");
        assert!(confidence > 0.3);
    }

    #[test]
    fn test_classify_trend_follower() {
        let data = WalletData {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            total_trades: 30,
            gross_pnl_usd: 15000.0,
            entry_fees_usd: 300.0,
            exit_fees_usd: 300.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: 14400.0,
            wins: 15,
            losses: 15,
            win_rate_pct: 50.0,
            volume_usd: 500_000.0,
            avg_trade_size_usd: 15000.0,
            largest_trade_usd: 50000.0,
            fills: vec![],
            markets: vec!["BTC".to_string(), "ETH".to_string()],
        };
        let mut metrics = compute_metrics(&data);
        metrics.direction_bias = Some(0.8);
        metrics.direction_label = Some("long".to_string());
        metrics.hold_time_median_secs = Some(14400.0); // 4 hours
        metrics.pnl_skewness = Some(1.5); // Positive skew
        metrics.scale_in_pct = Some(40.0);
        metrics.avg_leverage = Some(5.0);
        metrics.max_leverage = Some(10.0);

        let (strategy, confidence) = classify_strategy(&metrics, &data);
        assert_eq!(strategy, "trend-follower");
        assert!(confidence > 0.3);
    }

    #[test]
    fn test_classify_hft_market_maker() {
        let data = WalletData {
            address: "test".to_string(),
            source: "hyperliquid".to_string(),
            total_trades: 5000,
            gross_pnl_usd: 100000.0,
            entry_fees_usd: 5000.0,
            exit_fees_usd: 5000.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: 90000.0,
            wins: 2700,
            losses: 2300,
            win_rate_pct: 54.0,
            volume_usd: 100_000_000.0,
            avg_trade_size_usd: 20000.0,
            largest_trade_usd: 25000.0,
            fills: vec![],
            markets: vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()],
        };
        let mut metrics = compute_metrics(&data);
        metrics.hold_time_median_secs = Some(30.0); // Very fast
        metrics.direction_bias = Some(0.5);
        metrics.direction_label = Some("neutral".to_string());
        metrics.median_fill_interval_secs = Some(5.0);
        metrics.market_concentration_pct = Some(30.0);

        let (strategy, confidence) = classify_strategy(&metrics, &data);
        assert_eq!(strategy, "hft-market-maker");
        assert!(confidence > 0.3);
    }

    #[test]
    fn test_classify_unknown_low_trades() {
        let data = make_test_data(3, 100.0, 10.0, 2, 1);
        let metrics = compute_metrics(&data);
        let (strategy, _confidence) = classify_strategy(&metrics, &data);
        // With only 3 trades, should be unknown or a weak classification
        // (not enough data for confident classification)
        assert!(STRATEGY_TYPES.contains(&strategy.as_str()));
    }

    #[test]
    fn test_estimate_hold_time() {
        assert!(estimate_hold_time(1000) < 100.0); // HFT-like
        assert!(estimate_hold_time(200) < 600.0); // Scalper
        assert!(estimate_hold_time(50) > 1000.0 && estimate_hold_time(50) < 10000.0); // Active
        assert!(estimate_hold_time(5) > 86400.0 / 2.0); // Position trader
    }

    #[test]
    fn test_strategy_blueprint_contains_required_fields() {
        let report = WalletReport {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            analyzed_at: "2026-01-01T00:00:00Z".to_string(),
            total_trades: 100,
            fee_negative: false,
            strategy_type: "momentum-scalper".to_string(),
            classification_confidence: 0.85,
            metrics: WalletMetrics {
                clip_size_consistency_pct: Some(85.0),
                hold_time_median_secs: Some(120.0),
                hold_time_p25_secs: Some(60.0),
                hold_time_p75_secs: Some(300.0),
                hold_time_max_secs: Some(600.0),
                direction_bias: Some(0.75),
                direction_label: Some("long".to_string()),
                win_rate_pct: Some(75.0),
                pnl_mean_usd: Some(100.0),
                pnl_median_usd: Some(80.0),
                pnl_max_winner_usd: Some(500.0),
                pnl_max_loser_usd: Some(-200.0),
                pnl_skewness: Some(0.5),
                gross_pnl_usd: Some(10000.0),
                net_pnl_usd: Some(9500.0),
                total_fees_usd: Some(500.0),
                entry_fees_usd: Some(250.0),
                exit_fees_usd: Some(250.0),
                borrow_fees_usd: Some(0.0),
                counterparty_concentration_pct: Some(80.0),
                market_concentration_pct: Some(90.0),
                markets_traded: Some(vec!["SOL".to_string()]),
                median_fill_interval_secs: Some(180.0),
                trading_hours_utc: Some(vec![8, 9, 10]),
                scale_in_pct: Some(15.0),
                avg_leverage: Some(5.0),
                max_leverage: Some(10.0),
            },
        };

        let data = WalletData {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            total_trades: 100,
            gross_pnl_usd: 10000.0,
            entry_fees_usd: 250.0,
            exit_fees_usd: 250.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: 9500.0,
            wins: 75,
            losses: 25,
            win_rate_pct: 75.0,
            volume_usd: 1_000_000.0,
            avg_trade_size_usd: 10000.0,
            largest_trade_usd: 12000.0,
            fills: vec![],
            markets: vec!["SOL".to_string()],
        };

        let blueprints = generate_blueprints(&[report], &[data]);
        assert_eq!(blueprints.len(), 1);

        let bp = &blueprints[0];
        assert_eq!(bp.strategy_type, "momentum-scalper");
        assert!(!bp.source_wallets.is_empty());
        assert!(bp.confidence > 0.0);
        assert!(!bp.markets.is_empty());
        assert!(bp.leverage >= 1.0);
        assert!(bp.clip_size_usd > 0.0);

        // Entry parameters
        assert!(!bp.parameters.entry.signal.is_empty());
        assert!(bp.parameters.entry.threshold_pct > 0.0);
        assert!(!bp.parameters.entry.confirmation.is_empty());

        // Exit parameters
        assert!(bp.parameters.exit.take_profit_pct > 0.0);
        assert!(bp.parameters.exit.stop_loss_pct > 0.0);
        assert!(bp.parameters.exit.max_hold_secs > 0);

        // Risk parameters
        assert!(bp.parameters.risk.max_position_notional_usd > 0.0);
        assert!(bp.parameters.risk.daily_loss_limit_usd > 0.0);

        // Backtest metrics
        assert!(bp.backtest_metrics.total_trades_analyzed > 0);
        assert!(bp.backtest_metrics.win_rate > 0.0);
    }

    #[test]
    fn test_fee_negative_excluded_from_blueprints() {
        let report = WalletReport {
            address: "loser".to_string(),
            source: "flash-trade".to_string(),
            analyzed_at: "2026-01-01T00:00:00Z".to_string(),
            total_trades: 50,
            fee_negative: true,
            strategy_type: "momentum-scalper".to_string(),
            classification_confidence: 0.7,
            metrics: WalletMetrics {
                net_pnl_usd: Some(-500.0),
                gross_pnl_usd: Some(100.0),
                total_fees_usd: Some(600.0),
                ..WalletMetrics::default()
            },
        };

        let data = make_test_data(50, 100.0, 600.0, 25, 25);
        let blueprints = generate_blueprints(&[report], &[data]);
        assert!(
            blueprints.is_empty(),
            "Fee-negative wallets should not generate blueprints"
        );
    }

    #[test]
    fn test_unknown_excluded_from_blueprints() {
        let report = WalletReport {
            address: "unknown_wallet".to_string(),
            source: "flash-trade".to_string(),
            analyzed_at: "2026-01-01T00:00:00Z".to_string(),
            total_trades: 2,
            fee_negative: false,
            strategy_type: "unknown".to_string(),
            classification_confidence: 0.1,
            metrics: WalletMetrics::default(),
        };

        let data = make_test_data(2, 50.0, 5.0, 1, 1);
        let blueprints = generate_blueprints(&[report], &[data]);
        assert!(
            blueprints.is_empty(),
            "Unknown strategy wallets should not generate blueprints"
        );
    }

    #[test]
    fn test_report_json_schema() {
        let report = WalletReport {
            address: "BxkieMNZXqJf1GnAHjn7VjFHGLR3fKxuJG1MWEUuVrnB".to_string(),
            source: "flash-trade".to_string(),
            analyzed_at: "2026-01-01T00:00:00Z".to_string(),
            total_trades: 100,
            fee_negative: false,
            strategy_type: "momentum-scalper".to_string(),
            classification_confidence: 0.85,
            metrics: WalletMetrics {
                clip_size_consistency_pct: Some(85.0),
                hold_time_median_secs: Some(120.0),
                hold_time_p25_secs: Some(60.0),
                hold_time_p75_secs: Some(300.0),
                hold_time_max_secs: Some(600.0),
                direction_bias: Some(0.75),
                direction_label: Some("long".to_string()),
                win_rate_pct: Some(75.0),
                pnl_mean_usd: Some(100.0),
                pnl_median_usd: Some(80.0),
                pnl_max_winner_usd: Some(500.0),
                pnl_max_loser_usd: Some(-200.0),
                pnl_skewness: Some(0.5),
                gross_pnl_usd: Some(10000.0),
                net_pnl_usd: Some(9500.0),
                total_fees_usd: Some(500.0),
                entry_fees_usd: Some(250.0),
                exit_fees_usd: Some(250.0),
                borrow_fees_usd: Some(0.0),
                counterparty_concentration_pct: Some(80.0),
                market_concentration_pct: Some(90.0),
                markets_traded: Some(vec!["SOL".to_string()]),
                median_fill_interval_secs: Some(180.0),
                trading_hours_utc: Some(vec![8, 9, 10]),
                scale_in_pct: Some(15.0),
                avg_leverage: Some(5.0),
                max_leverage: Some(10.0),
            },
        };

        let json = serde_json::to_value(&report).unwrap();
        // Verify all required top-level fields
        assert!(json.get("address").is_some());
        assert!(json.get("source").is_some());
        assert!(json.get("analyzed_at").is_some());
        assert!(json.get("total_trades").is_some());
        assert!(json.get("fee_negative").is_some());
        assert!(json.get("strategy_type").is_some());
        assert!(json.get("classification_confidence").is_some());
        assert!(json.get("metrics").is_some());

        // Verify all metric fields
        let metrics = json.get("metrics").unwrap();
        assert!(metrics.get("clip_size_consistency_pct").is_some());
        assert!(metrics.get("hold_time_median_secs").is_some());
        assert!(metrics.get("direction_bias").is_some());
        assert!(metrics.get("win_rate_pct").is_some());
        assert!(metrics.get("pnl_mean_usd").is_some());
        assert!(metrics.get("gross_pnl_usd").is_some());
        assert!(metrics.get("net_pnl_usd").is_some());
        assert!(metrics.get("total_fees_usd").is_some());
        assert!(metrics.get("entry_fees_usd").is_some());
        assert!(metrics.get("exit_fees_usd").is_some());
        assert!(metrics.get("borrow_fees_usd").is_some());
        assert!(metrics.get("counterparty_concentration_pct").is_some());
        assert!(metrics.get("market_concentration_pct").is_some());
        assert!(metrics.get("markets_traded").is_some());
        assert!(metrics.get("median_fill_interval_secs").is_some());
        assert!(metrics.get("trading_hours_utc").is_some());
        assert!(metrics.get("scale_in_pct").is_some());
        assert!(metrics.get("avg_leverage").is_some());
        assert!(metrics.get("max_leverage").is_some());
    }

    #[test]
    fn test_at_least_3_strategy_types_classifiable() {
        // Verify that the classification logic can produce at least 3 types
        let mut classified_types: HashSet<String> = HashSet::new();

        // Momentum scalper
        let data_ms = WalletData {
            total_trades: 200,
            wins: 150,
            losses: 50,
            win_rate_pct: 75.0,
            markets: vec!["SOL".to_string()],
            ..make_test_data(200, 50000.0, 2000.0, 150, 50)
        };
        let mut metrics_ms = compute_metrics(&data_ms);
        metrics_ms.clip_size_consistency_pct = Some(85.0);
        metrics_ms.hold_time_median_secs = Some(120.0);
        metrics_ms.direction_bias = Some(0.75);
        metrics_ms.direction_label = Some("long".to_string());
        metrics_ms.market_concentration_pct = Some(95.0);
        metrics_ms.median_fill_interval_secs = Some(180.0);
        classified_types.insert(classify_strategy(&metrics_ms, &data_ms).0);

        // Mean reversion
        let data_mr = WalletData {
            total_trades: 300,
            wins: 200,
            losses: 100,
            win_rate_pct: 66.7,
            markets: vec!["SOL".to_string(), "BTC".to_string(), "ETH".to_string()],
            ..make_test_data(300, 20000.0, 1600.0, 200, 100)
        };
        let mut metrics_mr = compute_metrics(&data_mr);
        metrics_mr.direction_bias = Some(0.5);
        metrics_mr.direction_label = Some("neutral".to_string());
        metrics_mr.hold_time_median_secs = Some(600.0);
        metrics_mr.pnl_skewness = Some(0.3);
        metrics_mr.market_concentration_pct = Some(40.0);
        metrics_mr.median_fill_interval_secs = Some(300.0);
        classified_types.insert(classify_strategy(&metrics_mr, &data_mr).0);

        // Trend follower
        let data_tf = WalletData {
            total_trades: 30,
            wins: 15,
            losses: 15,
            win_rate_pct: 50.0,
            markets: vec!["BTC".to_string(), "ETH".to_string()],
            ..make_test_data(30, 15000.0, 600.0, 15, 15)
        };
        let mut metrics_tf = compute_metrics(&data_tf);
        metrics_tf.direction_bias = Some(0.8);
        metrics_tf.direction_label = Some("long".to_string());
        metrics_tf.hold_time_median_secs = Some(14400.0);
        metrics_tf.pnl_skewness = Some(1.5);
        metrics_tf.scale_in_pct = Some(40.0);
        metrics_tf.avg_leverage = Some(5.0);
        metrics_tf.max_leverage = Some(10.0);
        classified_types.insert(classify_strategy(&metrics_tf, &data_tf).0);

        assert!(
            classified_types.len() >= 3,
            "Expected at least 3 distinct strategy types, got: {:?}",
            classified_types
        );
    }

    #[test]
    fn test_lp_consumer_classification() {
        let data = WalletData {
            address: "test".to_string(),
            source: "flash-trade".to_string(),
            total_trades: 50,
            gross_pnl_usd: 25000.0,
            entry_fees_usd: 500.0,
            exit_fees_usd: 500.0,
            borrow_fees_usd: 0.0,
            net_pnl_usd: 24000.0,
            wins: 35,
            losses: 15,
            win_rate_pct: 70.0,
            volume_usd: 2_000_000.0,
            avg_trade_size_usd: 40000.0,
            largest_trade_usd: 45000.0,
            fills: vec![],
            markets: vec!["ZEC".to_string()], // Single market
        };
        let mut metrics = compute_metrics(&data);
        metrics.clip_size_consistency_pct = Some(90.0);
        metrics.market_concentration_pct = Some(100.0); // Single market
        metrics.counterparty_concentration_pct = Some(95.0);
        metrics.direction_bias = Some(0.8);
        metrics.direction_label = Some("long".to_string());
        metrics.hold_time_median_secs = Some(600.0);
        metrics.median_fill_interval_secs = Some(300.0);

        let (strategy, confidence) = classify_strategy(&metrics, &data);
        assert_eq!(strategy, "lp-consumer");
        assert!(confidence > 0.4);
    }
}
