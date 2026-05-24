//! scan-markets — Ranks Flash Trade markets by attractiveness for LP consumption strategies.
//!
//! Fetches market data from Flash Trade API (pool-data, prices, raw/markets)
//! and Hyperliquid meta endpoint to produce a ranked list of markets.
//!
//! Scoring criteria:
//!   - LP concentration / AUM (higher TVL = more liquidity to consume)
//!   - Utilization (moderate is ideal; too high = crowded, too low = no edge)
//!   - Max leverage (higher = more capital efficiency)
//!   - Fee rate (lower = cheaper to trade)
//!   - Available capacity (more room for our positions)
//!   - Flash-only bonus (markets not on HL = less competition)
//!
//! Output: data/market-rankings.json with market symbol, rank, score, metrics,
//!         HL→Flash Trade asset mapping, and Flash-only market flags.

use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{info, warn};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "scan-markets",
    about = "Rank Flash Trade markets by attractiveness for LP consumption",
    version
)]
struct Args {
    /// Output file path (JSON). Defaults to data/market-rankings.json
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Flash Trade API base URL
    #[arg(long, default_value = "https://flashapi.trade")]
    flash_url: String,

    /// Hyperliquid Info API base URL
    #[arg(long, default_value = "https://api.hyperliquid.xyz")]
    hl_url: String,

    /// Weight for AUM/TVL in scoring (0.0–1.0)
    #[arg(long, default_value_t = 0.25)]
    weight_aum: f64,

    /// Weight for utilization in scoring (0.0–1.0)
    #[arg(long, default_value_t = 0.20)]
    weight_utilization: f64,

    /// Weight for max leverage in scoring (0.0–1.0)
    #[arg(long, default_value_t = 0.15)]
    weight_leverage: f64,

    /// Weight for fee rate in scoring (0.0–1.0, lower fee = higher score)
    #[arg(long, default_value_t = 0.15)]
    weight_fee: f64,

    /// Weight for available capacity in scoring (0.0–1.0)
    #[arg(long, default_value_t = 0.15)]
    weight_capacity: f64,

    /// Bonus for Flash-only markets (added to total score)
    #[arg(long, default_value_t = 0.10)]
    flash_only_bonus: f64,

    /// Minimum AUM (USD) to include a market
    #[arg(long, default_value_t = 0.0)]
    min_aum_usd: f64,
}

// ── Data Types ───────────────────────────────────────────────────────────────

/// Per-market metrics collected from Flash Trade API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMetrics {
    /// Flash Trade symbol (e.g. "SOL", "BTC", "FARTCOIN")
    pub symbol: String,
    /// Pool name the asset belongs to (e.g. "Crypto.1")
    pub pool_name: String,
    /// Total pool AUM in USD (from lpStats.totalPoolValueUsd)
    pub pool_aum_usd: f64,
    /// Max AUM allowed for the pool
    pub pool_max_aum_usd: f64,
    /// Utilization percentage for the asset (custody-level)
    pub utilization_pct: f64,
    /// Max leverage available for the asset
    pub max_leverage: f64,
    /// Open position fee rate (e.g. 1500000 = 0.15%)
    pub open_fee_rate: f64,
    /// Close position fee rate
    pub close_fee_rate: f64,
    /// Available capacity to add in USD
    pub available_capacity_usd: f64,
    /// Total USD owned by the custody
    pub total_usd_owned: f64,
    /// Current price from oracle
    pub price_usd: Option<f64>,
    /// Whether this market exists on Hyperliquid
    pub on_hyperliquid: bool,
    /// HL symbol if different from Flash Trade symbol
    pub hl_symbol: Option<String>,
    /// Whether this is a Flash-only market (not on HL)
    pub flash_only: bool,
}

/// Ranked market entry in the output JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedMarket {
    /// Rank (1 = best)
    pub rank: usize,
    /// Market symbol on Flash Trade
    pub symbol: String,
    /// Pool name
    pub pool_name: String,
    /// Overall attractiveness score (0–1 scale)
    pub score: f64,
    /// Breakdown of individual component scores
    pub score_breakdown: ScoreBreakdown,
    /// Raw metrics
    pub metrics: MarketMetrics,
}

/// Individual scoring components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub aum_score: f64,
    pub utilization_score: f64,
    pub leverage_score: f64,
    pub fee_score: f64,
    pub capacity_score: f64,
    pub flash_only_bonus: f64,
}

/// The complete output file schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRankings {
    /// Timestamp when the scan was run
    pub scanned_at: String,
    /// Number of markets ranked
    pub total_markets: usize,
    /// Ranked market list
    pub markets: Vec<RankedMarket>,
    /// HL→Flash Trade symbol mapping
    pub asset_mapping: AssetMapping,
}

/// HL→Flash Trade symbol correspondence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMapping {
    /// Markets on both HL and Flash Trade with symbol correspondence
    pub both_platforms: Vec<SymbolMapping>,
    /// Markets only on Flash Trade (LP consumption edge)
    pub flash_only: Vec<FlashOnlyMarket>,
    /// Markets on HL but not on Flash Trade (for reference)
    pub hl_only: Vec<String>,
}

/// Symbol mapping for a market on both platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMapping {
    pub flash_symbol: String,
    pub hl_symbol: String,
    pub note: Option<String>,
}

/// Flash-only market entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashOnlyMarket {
    pub symbol: String,
    pub pool_name: String,
    pub pool_aum_usd: f64,
}

// ── API Response Types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct PoolDataResponse {
    #[serde(default)]
    pools: Vec<PoolEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct PoolEntry {
    #[serde(default)]
    custody_stats: Vec<CustodyStats>,
    #[serde(default)]
    lp_stats: Option<LpStats>,
    #[serde(default, rename = "poolName")]
    pool_name: String,
    #[serde(default, rename = "poolAddress")]
    pool_address: String, // used for API key, not read by scoring
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CustodyStats {
    symbol: String,
    #[serde(default)]
    utilization_ui: String,
    #[serde(default)]
    max_leverage: String,
    #[serde(default)]
    open_position_fee_rate: String,
    #[serde(default)]
    close_position_fee_rate: String,
    #[serde(default)]
    available_to_add_usd_ui: String,
    #[serde(default)]
    total_usd_owned_amount_ui: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LpStats {
    #[serde(default)]
    total_pool_value_usd: String,
    #[serde(default)]
    max_aum_usd: String,
}

#[derive(Debug, Deserialize)]
struct HlMetaResponse {
    universe: Vec<HlMarketInfo>,
}

#[derive(Debug, Deserialize)]
struct HlMarketInfo {
    name: String,
}

// ── Scoring Logic ────────────────────────────────────────────────────────────

/// Known symbol mappings between Hyperliquid and Flash Trade.
/// Some symbols differ between platforms.
fn known_symbol_mappings() -> HashMap<String, String> {
    let mut m = HashMap::new();
    // Hyperliquid uses kPEPE, Flash Trade uses PEPE
    m.insert("kPEPE".to_string(), "PEPE".to_string());
    // These are typically the same but we list them for completeness
    m
}

/// Build HL→Flash Trade asset mapping.
pub fn build_asset_mapping(flash_symbols: &[String], hl_symbols: &[String]) -> AssetMapping {
    let hl_set: HashSet<&str> = hl_symbols.iter().map(|s| s.as_str()).collect();
    let flash_set: HashSet<&str> = flash_symbols.iter().map(|s| s.as_str()).collect();
    let known = known_symbol_mappings();

    let mut both_platforms = Vec::new();
    let mut flash_only_list = Vec::new();
    let mut hl_only_list = Vec::new();

    // Flash symbols: check which are on HL
    let mut flash_matched = HashSet::new();
    for sym in flash_symbols {
        // Direct match
        if hl_set.contains(sym.as_str()) {
            both_platforms.push(SymbolMapping {
                flash_symbol: sym.clone(),
                hl_symbol: sym.clone(),
                note: None,
            });
            flash_matched.insert(sym.as_str());
        } else if let Some(hl_sym) = known.get(sym.as_str()) {
            // Known mapping (Flash→HL)
            both_platforms.push(SymbolMapping {
                flash_symbol: sym.clone(),
                hl_symbol: hl_sym.clone(),
                note: Some(format!("Flash Trade {} corresponds to HL {}", sym, hl_sym)),
            });
            flash_matched.insert(sym.as_str());
        } else if let Some(hl_sym) = known.iter().find(|(_, v)| *v == sym) {
            // Reverse mapping (HL→Flash)
            both_platforms.push(SymbolMapping {
                flash_symbol: sym.clone(),
                hl_symbol: hl_sym.0.clone(),
                note: Some(format!(
                    "Flash Trade {} corresponds to HL {}",
                    sym, hl_sym.0
                )),
            });
            flash_matched.insert(sym.as_str());
        }
    }

    // Check reverse: known HL symbols that map to Flash symbols
    for (hl_sym, flash_sym) in &known {
        if flash_set.contains(flash_sym.as_str()) && !flash_matched.contains(flash_sym.as_str()) {
            both_platforms.push(SymbolMapping {
                flash_symbol: flash_sym.clone(),
                hl_symbol: hl_sym.clone(),
                note: Some(format!("HL {} → Flash Trade {}", hl_sym, flash_sym)),
            });
            flash_matched.insert(flash_sym.as_str());
        }
    }

    // Flash-only markets (not on HL and not in known mappings)
    for sym in flash_symbols {
        if !flash_matched.contains(sym.as_str()) && sym != "USDC" {
            flash_only_list.push(FlashOnlyMarket {
                symbol: sym.clone(),
                pool_name: String::new(),
                pool_aum_usd: 0.0,
            });
        }
    }

    // HL-only markets (not on Flash)
    for hl_sym in hl_symbols {
        let hl_str = hl_sym.as_str();
        let is_mapped_to_flash = both_platforms.iter().any(|m| m.hl_symbol == hl_str);
        if !flash_set.contains(hl_str) && !is_mapped_to_flash {
            // Also check if there's a known mapping from HL→Flash
            if !known.contains_key(hl_str) {
                hl_only_list.push(hl_str.to_string());
            }
        }
    }

    AssetMapping {
        both_platforms,
        flash_only: flash_only_list,
        hl_only: hl_only_list,
    }
}

/// Normalize a value to 0–1 range using min-max scaling.
fn normalize(value: f64, min_val: f64, max_val: f64) -> f64 {
    if (max_val - min_val).abs() < f64::EPSILON {
        return 0.5; // no variation → neutral score
    }
    ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
}

/// Score a single market based on its metrics.
///
/// Returns (total_score, breakdown).
pub fn score_market(
    metrics: &MarketMetrics,
    all_metrics: &[MarketMetrics],
    weights: &ScoreWeights,
    flash_only_bonus: f64,
) -> (f64, ScoreBreakdown) {
    // Collect ranges for normalization
    let aum_values: Vec<f64> = all_metrics.iter().map(|m| m.pool_aum_usd).collect();
    let leverage_values: Vec<f64> = all_metrics.iter().map(|m| m.max_leverage).collect();
    let fee_values: Vec<f64> = all_metrics
        .iter()
        .map(|m| m.open_fee_rate + m.close_fee_rate)
        .collect();
    let capacity_values: Vec<f64> = all_metrics
        .iter()
        .map(|m| m.available_capacity_usd)
        .collect();

    let aum_min = aum_values.iter().copied().fold(f64::MAX, f64::min);
    let aum_max = aum_values.iter().copied().fold(f64::MIN, f64::max);
    let lev_min = leverage_values.iter().copied().fold(f64::MAX, f64::min);
    let lev_max = leverage_values.iter().copied().fold(f64::MIN, f64::max);
    let fee_min = fee_values.iter().copied().fold(f64::MAX, f64::min);
    let fee_max = fee_values.iter().copied().fold(f64::MIN, f64::max);
    let cap_min = capacity_values.iter().copied().fold(f64::MAX, f64::min);
    let cap_max = capacity_values.iter().copied().fold(f64::MIN, f64::max);

    // AUM: higher is better
    let aum_score = normalize(metrics.pool_aum_usd, aum_min, aum_max);

    // Utilization: we want moderate utilization (30-70% is ideal for LP consumption)
    // Too low = no activity, too high = saturated
    let util_score = score_utilization(metrics.utilization_pct);

    // Leverage: higher is better (more capital efficiency)
    let leverage_score = normalize(metrics.max_leverage, lev_min, lev_max);

    // Fee: lower is better → invert the normalization
    let total_fee = metrics.open_fee_rate + metrics.close_fee_rate;
    let fee_score = 1.0 - normalize(total_fee, fee_min, fee_max);

    // Capacity: more available is better
    let capacity_score = normalize(metrics.available_capacity_usd, cap_min, cap_max);

    // Flash-only bonus
    let flash_bonus = if metrics.flash_only {
        flash_only_bonus
    } else {
        0.0
    };

    let total = weights.aum * aum_score
        + weights.utilization * util_score
        + weights.leverage * leverage_score
        + weights.fee * fee_score
        + weights.capacity * capacity_score
        + flash_bonus;

    (
        total,
        ScoreBreakdown {
            aum_score,
            utilization_score: util_score,
            leverage_score,
            fee_score,
            capacity_score,
            flash_only_bonus: flash_bonus,
        },
    )
}

/// Score utilization: ideal range is 20-60%, penalize extremes.
fn score_utilization(utilization_pct: f64) -> f64 {
    // Bell curve centered around 40% utilization
    // Peak score at 40%, declining toward 0% and 100%
    let ideal = 40.0;
    let spread = 30.0;
    let distance = (utilization_pct - ideal).abs();
    (1.0 - (distance / spread).min(1.0)).max(0.0)
}

/// Scoring weights.
pub struct ScoreWeights {
    aum: f64,
    utilization: f64,
    leverage: f64,
    fee: f64,
    capacity: f64,
}

// ── Data Fetching ────────────────────────────────────────────────────────────

async fn fetch_flash_pool_data(client: &Client, base_url: &str) -> Result<Vec<PoolEntry>> {
    let url = format!("{}/pool-data", base_url);
    info!("Fetching Flash Trade pool data from {}", url);
    let resp = client.get(&url).send().await.context("pool-data request")?;
    let data: PoolDataResponse = resp.json().await.context("pool-data parse")?;
    info!("Found {} pools", data.pools.len());
    Ok(data.pools)
}

async fn fetch_flash_prices(client: &Client, base_url: &str) -> Result<HashMap<String, f64>> {
    let url = format!("{}/prices", base_url);
    info!("Fetching Flash Trade prices from {}", url);
    let resp = client.get(&url).send().await.context("prices request")?;
    let data: HashMap<String, serde_json::Value> = resp.json().await.context("prices parse")?;

    let mut prices = HashMap::new();
    for (symbol, val) in &data {
        if let Some(price_ui) = val.get("priceUi").and_then(|v| v.as_f64()) {
            prices.insert(symbol.clone(), price_ui);
        }
    }
    info!("Found {} price entries", prices.len());
    Ok(prices)
}

async fn fetch_hl_meta(client: &Client, hl_url: &str) -> Result<Vec<String>> {
    let url = format!("{}/info", hl_url);
    info!("Fetching Hyperliquid meta from {}", url);
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"type": "meta"}))
        .send()
        .await
        .context("HL meta request")?;

    let data: HlMetaResponse = resp.json().await.context("HL meta parse")?;
    let symbols: Vec<String> = data.universe.iter().map(|m| m.name.clone()).collect();
    info!("Found {} HL markets", symbols.len());
    Ok(symbols)
}

// ── Market Extraction ────────────────────────────────────────────────────────

/// Extract individual market metrics from pool data entries.
pub fn extract_market_metrics(
    pools: &[PoolEntry],
    prices: &HashMap<String, f64>,
    hl_symbols: &[String],
) -> Vec<MarketMetrics> {
    let hl_set: HashSet<&str> = hl_symbols.iter().map(|s| s.as_str()).collect();
    let known = known_symbol_mappings();
    let mut all_metrics = Vec::new();

    for pool in pools {
        let pool_aum = pool
            .lp_stats
            .as_ref()
            .and_then(|s| s.total_pool_value_usd.parse::<f64>().ok())
            .unwrap_or(0.0);
        let pool_max_aum = pool
            .lp_stats
            .as_ref()
            .and_then(|s| s.max_aum_usd.parse::<f64>().ok())
            .unwrap_or(0.0);

        for custody in &pool.custody_stats {
            // Skip USDC collateral entries
            if custody.symbol == "USDC" {
                continue;
            }

            let util_pct = custody.utilization_ui.parse::<f64>().unwrap_or(0.0);
            let max_lev = custody.max_leverage.parse::<f64>().unwrap_or(1.0);
            let open_fee = custody.open_position_fee_rate.parse::<f64>().unwrap_or(0.0);
            let close_fee = custody
                .close_position_fee_rate
                .parse::<f64>()
                .unwrap_or(0.0);
            let available = custody
                .available_to_add_usd_ui
                .parse::<f64>()
                .unwrap_or(0.0);
            let total_owned = custody
                .total_usd_owned_amount_ui
                .parse::<f64>()
                .unwrap_or(0.0);

            let symbol = custody.symbol.clone();
            let price = prices.get(&symbol).copied();

            // Check if on HL (direct match or known mapping)
            let on_hl = hl_set.contains(symbol.as_str()) || known.contains_key(&symbol);
            let hl_symbol = if on_hl {
                if hl_set.contains(symbol.as_str()) {
                    Some(symbol.clone())
                } else {
                    known.get(&symbol).cloned()
                }
            } else {
                None
            };

            all_metrics.push(MarketMetrics {
                symbol,
                pool_name: pool.pool_name.clone(),
                pool_aum_usd: pool_aum,
                pool_max_aum_usd: pool_max_aum,
                utilization_pct: util_pct,
                max_leverage: max_lev,
                open_fee_rate: open_fee,
                close_fee_rate: close_fee,
                available_capacity_usd: available,
                total_usd_owned: total_owned,
                price_usd: price,
                on_hyperliquid: on_hl,
                hl_symbol,
                flash_only: !on_hl,
            });
        }
    }

    all_metrics
}

// ── Main Logic ───────────────────────────────────────────────────────────────

async fn run(args: Args) -> Result<()> {
    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/market-rankings.json"));

    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("HTTP client")?;

    // Fetch data from APIs
    let pools = fetch_flash_pool_data(&client, &args.flash_url).await?;
    let prices = fetch_flash_prices(&client, &args.flash_url).await?;
    let hl_symbols = fetch_hl_meta(&client, &args.hl_url).await?;

    // Extract market metrics
    let all_metrics = extract_market_metrics(&pools, &prices, &hl_symbols);

    // Filter by minimum AUM
    let filtered: Vec<MarketMetrics> = all_metrics
        .into_iter()
        .filter(|m| m.pool_aum_usd >= args.min_aum_usd)
        .collect();

    info!(
        "After filtering: {} markets (min_aum=${})",
        filtered.len(),
        args.min_aum_usd
    );

    if filtered.is_empty() {
        warn!("No markets found matching criteria");
        // Still produce output with empty rankings
        let flash_syms: Vec<String> = vec![];
        let rankings = MarketRankings {
            scanned_at: chrono::Utc::now().to_rfc3339(),
            total_markets: 0,
            markets: vec![],
            asset_mapping: build_asset_mapping(&flash_syms, &hl_symbols),
        };
        write_output(&rankings, &output_path)?;
        return Ok(());
    }

    let weights = ScoreWeights {
        aum: args.weight_aum,
        utilization: args.weight_utilization,
        leverage: args.weight_leverage,
        fee: args.weight_fee,
        capacity: args.weight_capacity,
    };

    // Score and rank markets
    let mut scored: Vec<(f64, ScoreBreakdown, MarketMetrics)> = filtered
        .iter()
        .map(|m| {
            let (score, breakdown) = score_market(m, &filtered, &weights, args.flash_only_bonus);
            (score, breakdown, m.clone())
        })
        .collect();

    // Sort by score descending
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build ranked output
    let ranked_markets: Vec<RankedMarket> = scored
        .into_iter()
        .enumerate()
        .map(|(i, (score, breakdown, metrics))| RankedMarket {
            rank: i + 1,
            symbol: metrics.symbol.clone(),
            pool_name: metrics.pool_name.clone(),
            score,
            score_breakdown: breakdown,
            metrics,
        })
        .collect();

    // Build asset mapping
    let flash_symbols: Vec<String> = ranked_markets.iter().map(|m| m.symbol.clone()).collect();
    let asset_mapping = build_asset_mapping(&flash_symbols, &hl_symbols);

    // Enrich flash_only markets in mapping with pool data
    let mut enriched_mapping = asset_mapping;
    for fo in &mut enriched_mapping.flash_only {
        if let Some(market) = ranked_markets.iter().find(|m| m.symbol == fo.symbol) {
            fo.pool_name = market.pool_name.clone();
            fo.pool_aum_usd = market.metrics.pool_aum_usd;
        }
    }

    let rankings = MarketRankings {
        scanned_at: chrono::Utc::now().to_rfc3339(),
        total_markets: ranked_markets.len(),
        markets: ranked_markets,
        asset_mapping: enriched_mapping,
    };

    // Print summary
    info!("=== Market Rankings ===");
    for market in &rankings.markets {
        info!(
            "  #{:<3} {:<12} score={:.3}  AUM=${:.0}  util={:.1}%  lev={:.0}x  fee={:.0}  cap=${:.0}  {}{}",
            market.rank,
            market.symbol,
            market.score,
            market.metrics.pool_aum_usd,
            market.metrics.utilization_pct,
            market.metrics.max_leverage,
            market.metrics.open_fee_rate + market.metrics.close_fee_rate,
            market.metrics.available_capacity_usd,
            if market.metrics.flash_only {
                "FLASH-ONLY "
            } else {
                ""
            },
            market.pool_name,
        );
    }

    info!(
        "Asset mapping: {} on both platforms, {} Flash-only, {} HL-only",
        rankings.asset_mapping.both_platforms.len(),
        rankings.asset_mapping.flash_only.len(),
        rankings.asset_mapping.hl_only.len(),
    );

    write_output(&rankings, &output_path)?;

    Ok(())
}

/// Write rankings to JSON file using atomic write pattern.
fn write_output(rankings: &MarketRankings, path: &PathBuf) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating directory {:?}", parent))?;
    }

    let json = serde_json::to_string_pretty(rankings).context("serializing rankings")?;

    // Atomic write: write to .tmp then rename
    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, &json).with_context(|| format!("writing {:?}", tmp_path))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("renaming {:?} to {:?}", tmp_path, path))?;

    info!("Wrote market rankings to {:?}", path);
    Ok(())
}

// ── Entry Point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    info!("scan-markets starting");
    run(args).await
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_metrics(
        symbol: &str,
        aum: f64,
        util: f64,
        leverage: f64,
        fee: f64,
        capacity: f64,
        flash_only: bool,
    ) -> MarketMetrics {
        MarketMetrics {
            symbol: symbol.to_string(),
            pool_name: "TestPool".to_string(),
            pool_aum_usd: aum,
            pool_max_aum_usd: aum * 2.0,
            utilization_pct: util,
            max_leverage: leverage,
            open_fee_rate: fee / 2.0,
            close_fee_rate: fee / 2.0,
            available_capacity_usd: capacity,
            total_usd_owned: aum,
            price_usd: Some(100.0),
            on_hyperliquid: !flash_only,
            hl_symbol: if flash_only {
                None
            } else {
                Some(symbol.to_string())
            },
            flash_only,
        }
    }

    #[test]
    fn test_utilization_score_ideal_range() {
        // Ideal is around 40%
        let score_ideal = score_utilization(40.0);
        let score_low = score_utilization(5.0);
        let score_high = score_utilization(90.0);
        let score_zero = score_utilization(0.0);

        assert!(
            score_ideal > 0.9,
            "Ideal utilization should score high: got {}",
            score_ideal
        );
        assert!(
            score_low < score_ideal,
            "Low utilization should score less than ideal"
        );
        assert!(
            score_high < score_ideal,
            "High utilization should score less than ideal"
        );
        assert!(
            score_zero < score_ideal,
            "Zero utilization should score less than ideal"
        );
    }

    #[test]
    fn test_utilization_score_bell_curve() {
        // Scores should decrease as we move away from ideal
        let s40 = score_utilization(40.0);
        let s30 = score_utilization(30.0);
        let s20 = score_utilization(20.0);
        let s10 = score_utilization(10.0);

        assert!(s40 >= s30, "40% >= 30%");
        assert!(s30 >= s20, "30% >= 20%");
        assert!(s20 >= s10, "20% >= 10%");
    }

    #[test]
    fn test_normalize_basic() {
        assert!((normalize(5.0, 0.0, 10.0) - 0.5).abs() < 0.001);
        assert!((normalize(0.0, 0.0, 10.0) - 0.0).abs() < 0.001);
        assert!((normalize(10.0, 0.0, 10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_clamps() {
        assert!((normalize(-5.0, 0.0, 10.0) - 0.0).abs() < 0.001);
        assert!((normalize(15.0, 0.0, 10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_no_variation() {
        // All same values → neutral 0.5
        assert!((normalize(5.0, 5.0, 5.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_score_market_high_aum_scores_higher() {
        let metrics_low = make_test_metrics("LOW", 1000.0, 40.0, 50.0, 3000000.0, 50000.0, false);
        let metrics_high =
            make_test_metrics("HIGH", 5000000.0, 40.0, 50.0, 3000000.0, 50000.0, false);
        let all = vec![metrics_low.clone(), metrics_high.clone()];

        let weights = ScoreWeights {
            aum: 0.25,
            utilization: 0.20,
            leverage: 0.15,
            fee: 0.15,
            capacity: 0.15,
        };

        let (score_low, _) = score_market(&metrics_low, &all, &weights, 0.1);
        let (score_high, _) = score_market(&metrics_high, &all, &weights, 0.1);

        assert!(
            score_high > score_low,
            "Higher AUM should score higher: {} vs {}",
            score_high,
            score_low
        );
    }

    #[test]
    fn test_score_market_flash_only_bonus() {
        let metrics_normal =
            make_test_metrics("SOL", 100000.0, 40.0, 50.0, 3000000.0, 50000.0, false);
        let metrics_flash =
            make_test_metrics("MEME", 100000.0, 40.0, 50.0, 3000000.0, 50000.0, true);
        let all = vec![metrics_normal.clone(), metrics_flash.clone()];

        let weights = ScoreWeights {
            aum: 0.25,
            utilization: 0.20,
            leverage: 0.15,
            fee: 0.15,
            capacity: 0.15,
        };

        let (score_normal, _) = score_market(&metrics_normal, &all, &weights, 0.1);
        let (score_flash, breakdown) = score_market(&metrics_flash, &all, &weights, 0.1);

        assert!(
            score_flash > score_normal,
            "Flash-only market should score higher with bonus: {} vs {}",
            score_flash,
            score_normal
        );
        assert!(
            (breakdown.flash_only_bonus - 0.1).abs() < 0.001,
            "Flash-only bonus should be 0.1"
        );
    }

    #[test]
    fn test_score_market_lower_fee_better() {
        let metrics_high_fee =
            make_test_metrics("A", 100000.0, 40.0, 50.0, 5000000.0, 50000.0, false);
        let metrics_low_fee =
            make_test_metrics("B", 100000.0, 40.0, 50.0, 1000000.0, 50000.0, false);
        let all = vec![metrics_high_fee.clone(), metrics_low_fee.clone()];

        let weights = ScoreWeights {
            aum: 0.25,
            utilization: 0.20,
            leverage: 0.15,
            fee: 0.15,
            capacity: 0.15,
        };

        let (score_high_fee, _) = score_market(&metrics_high_fee, &all, &weights, 0.1);
        let (score_low_fee, _) = score_market(&metrics_low_fee, &all, &weights, 0.1);

        assert!(
            score_low_fee > score_high_fee,
            "Lower fee should score higher: {} vs {}",
            score_low_fee,
            score_high_fee
        );
    }

    #[test]
    fn test_build_asset_mapping_basic() {
        let flash = vec![
            "SOL".to_string(),
            "BTC".to_string(),
            "ETH".to_string(),
            "FARTCOIN".to_string(),
        ];
        let hl = vec![
            "BTC".to_string(),
            "ETH".to_string(),
            "SOL".to_string(),
            "DOGE".to_string(),
        ];

        let mapping = build_asset_mapping(&flash, &hl);

        assert_eq!(mapping.both_platforms.len(), 3, "SOL, BTC, ETH on both");
        assert!(
            mapping.flash_only.iter().any(|m| m.symbol == "FARTCOIN"),
            "FARTCOIN should be Flash-only"
        );
        assert!(
            mapping.hl_only.contains(&"DOGE".to_string()),
            "DOGE should be HL-only"
        );
    }

    #[test]
    fn test_build_asset_mapping_known_symbol_differences() {
        let flash = vec!["PEPE".to_string()];
        let hl = vec!["kPEPE".to_string()];

        let mapping = build_asset_mapping(&flash, &hl);

        // PEPE ↔ kPEPE should be in both_platforms via known mapping
        assert!(
            mapping
                .both_platforms
                .iter()
                .any(|m| m.flash_symbol == "PEPE" && m.hl_symbol == "kPEPE"),
            "PEPE should map to kPEPE"
        );
    }

    #[test]
    fn test_build_asset_mapping_empty_inputs() {
        let mapping = build_asset_mapping(&[], &[]);
        assert!(mapping.both_platforms.is_empty());
        assert!(mapping.flash_only.is_empty());
        assert!(mapping.hl_only.is_empty());
    }

    #[test]
    fn test_extract_market_metrics_skips_usdc() {
        let pool_json = serde_json::json!({
            "custodyStats": [
                {
                    "symbol": "USDC",
                    "utilizationUi": "0.00",
                    "maxLeverage": "0.00",
                    "openPositionFeeRate": "1500000",
                    "closePositionFeeRate": "1500000",
                    "availableToAddUsdUi": "1000.00",
                    "totalUsdOwnedAmountUi": "5000.00"
                },
                {
                    "symbol": "SOL",
                    "utilizationUi": "25.00",
                    "maxLeverage": "50.00",
                    "openPositionFeeRate": "1500000",
                    "closePositionFeeRate": "1500000",
                    "availableToAddUsdUi": "20000.00",
                    "totalUsdOwnedAmountUi": "30000.00"
                }
            ],
            "lpStats": {
                "totalPoolValueUsd": "100000.00",
                "maxAumUsd": "1000000.00"
            },
            "poolName": "Crypto.1",
            "poolAddress": "test123"
        });

        let pool: PoolEntry = serde_json::from_value(pool_json).unwrap();
        let metrics = extract_market_metrics(&[pool], &HashMap::new(), &[]);

        assert_eq!(metrics.len(), 1, "Should only have SOL, not USDC");
        assert_eq!(metrics[0].symbol, "SOL");
        assert!((metrics[0].pool_aum_usd - 100000.0).abs() < 0.01);
        assert!((metrics[0].utilization_pct - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_market_metrics_detects_flash_only() {
        let pool_json = serde_json::json!({
            "custodyStats": [
                {
                    "symbol": "FARTCOIN",
                    "utilizationUi": "14.23",
                    "maxLeverage": "50.00",
                    "openPositionFeeRate": "1200000",
                    "closePositionFeeRate": "1200000",
                    "availableToAddUsdUi": "22000.00",
                    "totalUsdOwnedAmountUi": "35000.00"
                }
            ],
            "lpStats": {
                "totalPoolValueUsd": "60000.00",
                "maxAumUsd": "1000000.00"
            },
            "poolName": "Trump.1",
            "poolAddress": "test456"
        });

        let pool: PoolEntry = serde_json::from_value(pool_json).unwrap();
        let hl_symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];
        let metrics = extract_market_metrics(&[pool], &HashMap::new(), &hl_symbols);

        assert_eq!(metrics.len(), 1);
        assert!(metrics[0].flash_only, "FARTCOIN should be Flash-only");
        assert!(!metrics[0].on_hyperliquid);
    }

    #[test]
    fn test_extract_market_metrics_with_prices() {
        let pool_json = serde_json::json!({
            "custodyStats": [
                {
                    "symbol": "BTC",
                    "utilizationUi": "50.00",
                    "maxLeverage": "100.00",
                    "openPositionFeeRate": "1500000",
                    "closePositionFeeRate": "1500000",
                    "availableToAddUsdUi": "50000.00",
                    "totalUsdOwnedAmountUi": "100000.00"
                }
            ],
            "lpStats": {
                "totalPoolValueUsd": "500000.00",
                "maxAumUsd": "5000000.00"
            },
            "poolName": "Crypto.1",
            "poolAddress": "test789"
        });

        let pool: PoolEntry = serde_json::from_value(pool_json).unwrap();
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 104500.0);

        let metrics = extract_market_metrics(&[pool], &prices, &["BTC".to_string()]);

        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].price_usd, Some(104500.0));
        assert!(metrics[0].on_hyperliquid);
        assert!(!metrics[0].flash_only);
    }

    #[test]
    fn test_ranking_output_order() {
        // Higher score should get lower rank number
        let metrics = vec![
            make_test_metrics("LOW", 1000.0, 40.0, 50.0, 3000000.0, 50000.0, false),
            make_test_metrics("HIGH", 5000000.0, 40.0, 50.0, 3000000.0, 50000.0, false),
        ];

        let weights = ScoreWeights {
            aum: 0.25,
            utilization: 0.20,
            leverage: 0.15,
            fee: 0.15,
            capacity: 0.15,
        };

        let mut scored: Vec<(f64, ScoreBreakdown, MarketMetrics)> = metrics
            .iter()
            .map(|m| {
                let (score, breakdown) = score_market(m, &metrics, &weights, 0.1);
                (score, breakdown, m.clone())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // HIGH should be first (higher AUM)
        assert_eq!(scored[0].2.symbol, "HIGH");
        assert_eq!(scored[1].2.symbol, "LOW");
    }

    #[test]
    fn test_market_rankings_json_schema() {
        // Verify the output structure serializes correctly
        let rankings = MarketRankings {
            scanned_at: "2026-05-22T00:00:00Z".to_string(),
            total_markets: 1,
            markets: vec![RankedMarket {
                rank: 1,
                symbol: "SOL".to_string(),
                pool_name: "Crypto.1".to_string(),
                score: 0.75,
                score_breakdown: ScoreBreakdown {
                    aum_score: 0.8,
                    utilization_score: 0.9,
                    leverage_score: 1.0,
                    fee_score: 0.5,
                    capacity_score: 0.6,
                    flash_only_bonus: 0.0,
                },
                metrics: make_test_metrics(
                    "SOL", 4440120.0, 30.0, 50.0, 3000000.0, 100000.0, false,
                ),
            }],
            asset_mapping: AssetMapping {
                both_platforms: vec![SymbolMapping {
                    flash_symbol: "SOL".to_string(),
                    hl_symbol: "SOL".to_string(),
                    note: None,
                }],
                flash_only: vec![],
                hl_only: vec!["DOGE".to_string()],
            },
        };

        let json = serde_json::to_string_pretty(&rankings).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Verify top-level fields
        assert!(parsed.get("scanned_at").is_some());
        assert!(parsed.get("total_markets").is_some());
        assert!(parsed.get("markets").is_some());
        assert!(parsed.get("asset_mapping").is_some());

        // Verify market fields
        let market = &parsed["markets"][0];
        assert!(market.get("rank").is_some());
        assert!(market.get("symbol").is_some());
        assert!(market.get("score").is_some());
        assert!(market.get("score_breakdown").is_some());
        assert!(market.get("metrics").is_some());

        // Verify metrics fields
        let metrics = &market["metrics"];
        assert!(metrics.get("symbol").is_some());
        assert!(metrics.get("pool_name").is_some());
        assert!(metrics.get("pool_aum_usd").is_some());
        assert!(metrics.get("utilization_pct").is_some());
        assert!(metrics.get("max_leverage").is_some());
        assert!(metrics.get("flash_only").is_some());

        // Verify asset mapping fields
        let mapping = &parsed["asset_mapping"];
        assert!(mapping.get("both_platforms").is_some());
        assert!(mapping.get("flash_only").is_some());
        assert!(mapping.get("hl_only").is_some());
    }

    #[test]
    fn test_atomic_write() {
        let tmp_dir = std::env::temp_dir().join("scan_markets_test");
        let _ = fs::create_dir_all(&tmp_dir);
        let output_path = tmp_dir.join("test-rankings.json");

        let rankings = MarketRankings {
            scanned_at: "2026-05-22T00:00:00Z".to_string(),
            total_markets: 0,
            markets: vec![],
            asset_mapping: AssetMapping {
                both_platforms: vec![],
                flash_only: vec![],
                hl_only: vec![],
            },
        };

        write_output(&rankings, &output_path).unwrap();
        assert!(output_path.exists(), "Output file should exist");

        // Verify no .tmp file remains
        let tmp_path = output_path.with_extension("json.tmp");
        assert!(!tmp_path.exists(), "Temp file should be cleaned up");

        // Verify valid JSON
        let content = fs::read_to_string(&output_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["total_markets"], 0);

        // Cleanup
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_score_market_all_same_metrics_equal_scores() {
        let m1 = make_test_metrics("A", 1000.0, 40.0, 50.0, 3000000.0, 50000.0, false);
        let m2 = make_test_metrics("B", 1000.0, 40.0, 50.0, 3000000.0, 50000.0, false);
        let all = vec![m1.clone(), m2.clone()];

        let weights = ScoreWeights {
            aum: 0.25,
            utilization: 0.20,
            leverage: 0.15,
            fee: 0.15,
            capacity: 0.15,
        };

        let (s1, _) = score_market(&m1, &all, &weights, 0.1);
        let (s2, _) = score_market(&m2, &all, &weights, 0.1);

        assert!(
            (s1 - s2).abs() < 0.001,
            "Identical metrics should produce same scores: {} vs {}",
            s1,
            s2
        );
    }
}
