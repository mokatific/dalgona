use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

#[allow(dead_code)]
const FLASH_API: &str = "https://flashapi.trade";

#[derive(Debug, Clone)]
pub struct FlashClient {
    client: Client,
    base_url: String,
}

// --- Price ---

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PriceData {
    pub price: u64,
    pub exponent: i32,
    #[serde(rename = "priceUi")]
    pub price_ui: f64,
    #[serde(rename = "timestampUs")]
    pub timestamp_us: u64,
}

// --- Position ---

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct FlashPosition {
    pub position_key: String,
    pub owner: String,
    pub pool: String,
    pub custody: String,
    pub collateral: String,
    pub side: String,
    pub asset: String,
    pub size: String,
    pub size_usd: String,
    pub collateral_usd: String,
    pub leverage: String,
    pub entry_price: String,
    pub mark_price: String,
    pub liquidation_price: Option<String>,
    pub unrealized_pnl_usd: Option<String>,
    pub unrealized_pnl_pct: Option<String>,
    pub borrow_fee: Option<String>,
    pub open_time: Option<i64>,
}

// --- Market ---

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct FlashMarket {
    pub pool: String,
    pub name: String,
    pub symbol: String,
    pub asset: String,
    pub side: String,
    pub market_account: String,
    pub custody_account: String,
    pub token_mint: String,
    pub token_vault: String,
    pub oracle: String,
    pub max_leverage: Option<f64>,
    pub fee_pct: Option<f64>,
}

// --- Open Position ---

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenPositionRequest {
    pub input_token_symbol: String,
    pub output_token_symbol: String,
    pub input_amount_ui: String,
    pub leverage: f64,
    pub trade_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take_profit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_loss: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slippage_percentage: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct OpenPositionResponse {
    pub new_leverage: Option<String>,
    pub new_entry_price: Option<String>,
    pub new_liquidation_price: Option<String>,
    pub entry_fee: Option<String>,
    pub you_pay_usd_ui: Option<String>,
    pub you_recieve_usd_ui: Option<String>,
    pub output_amount_ui: Option<String>,
    pub transaction_base64: Option<String>,
    pub take_profit_quote: Option<TriggerQuote>,
    pub stop_loss_quote: Option<TriggerQuote>,
    pub err: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct TriggerQuote {
    pub exit_price_ui: Option<String>,
    pub profit_usd_ui: Option<String>,
    pub loss_usd_ui: Option<String>,
    pub pnl_percentage: Option<String>,
}

// --- Close Position ---

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClosePositionRequest {
    pub position_key: String,
    pub input_usd_ui: String,
    pub withdraw_token_symbol: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slippage_percentage: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct ClosePositionResponse {
    pub receive_token_amount_ui: Option<String>,
    pub mark_price: Option<String>,
    pub entry_price: Option<String>,
    pub settled_pnl: Option<String>,
    pub fees: Option<String>,
    pub transaction_base64: Option<String>,
    pub err: Option<String>,
}

// --- Place Trigger Order ---

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct PlaceTriggerRequest {
    pub owner: String,
    pub position_key: String,
    pub order_type: String,
    pub price_ui: String,
    pub slippage_percentage: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct PlaceTriggerResponse {
    pub transaction_base64: Option<String>,
    pub err: Option<String>,
}

// --- Pool Data ---

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct PoolData {
    pub pool_pubkey: Option<String>,
    pub aum_usd: Option<String>,
    pub utilization: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PoolDataWrapper {
    #[serde(default)]
    pools: Vec<serde_json::Value>,
}

/// Parsed pool info for a single market, extracted from the raw `/pool-data` response.
/// Contains utilization for both long and short sides plus total AUM.
#[derive(Debug, Clone)]
pub struct RawPoolInfo {
    pub aum_usd: f64,
    pub long_utilization: f64,
    pub short_utilization: f64,
}

impl FlashClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    // --- Prices ---

    pub async fn get_price(&self, symbol: &str) -> Result<f64> {
        let url = format!("{}/prices/{}", self.base_url, symbol);
        debug!("GET {}", url);
        let resp = self.client.get(&url).send().await?;
        let data: PriceData = resp.json().await?;
        Ok(data.price_ui)
    }

    #[allow(dead_code)]
    pub async fn get_prices(&self) -> Result<Vec<PriceData>> {
        let url = format!("{}/prices", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let data: Vec<PriceData> = resp.json().await?;
        Ok(data)
    }

    // --- Positions ---

    pub async fn get_positions(&self, owner: &str) -> Result<Vec<FlashPosition>> {
        let url = format!(
            "{}/positions/owner/{}?includePnlInLeverageDisplay=true",
            self.base_url, owner
        );
        debug!("GET {}", url);
        let resp = self.client.get(&url).send().await?;
        if resp.status() == 404 {
            return Ok(vec![]);
        }
        let data: Vec<FlashPosition> = resp.json().await?;
        Ok(data)
    }

    #[allow(dead_code)]
    pub async fn get_position_for_market(
        &self,
        owner: &str,
        asset: &str,
        side: &str,
    ) -> Result<Option<FlashPosition>> {
        let positions = self.get_positions(owner).await?;
        Ok(positions
            .into_iter()
            .find(|p| p.asset == asset && p.side.to_uppercase() == side.to_uppercase()))
    }

    // --- Markets ---

    #[allow(dead_code)]
    pub async fn get_markets(&self) -> Result<Vec<FlashMarket>> {
        let url = format!("{}/raw/markets", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let data: Vec<FlashMarket> = resp.json().await?;
        Ok(data)
    }

    // --- Open Position ---

    pub async fn preview_open_position(
        &self,
        input_token: &str,
        output_token: &str,
        amount_usd: f64,
        leverage: f64,
        trade_type: &str,
    ) -> Result<OpenPositionResponse> {
        self.open_position_inner(
            input_token,
            output_token,
            amount_usd,
            leverage,
            trade_type,
            None,
            None,
            None,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn build_open_position(
        &self,
        input_token: &str,
        output_token: &str,
        amount_usd: f64,
        leverage: f64,
        trade_type: &str,
        owner: &str,
        slippage: &str,
        tp: Option<f64>,
        sl: Option<f64>,
    ) -> Result<OpenPositionResponse> {
        self.open_position_inner(
            input_token,
            output_token,
            amount_usd,
            leverage,
            trade_type,
            Some(owner.to_string()),
            Some(slippage.to_string()),
            tp.map(|p| format!("{:.2}", p)),
            sl.map(|p| format!("{:.2}", p)),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn open_position_inner(
        &self,
        input_token: &str,
        output_token: &str,
        amount_usd: f64,
        leverage: f64,
        trade_type: &str,
        owner: Option<String>,
        slippage: Option<String>,
        tp: Option<String>,
        sl: Option<String>,
    ) -> Result<OpenPositionResponse> {
        let url = format!("{}/transaction-builder/open-position", self.base_url);
        let body = OpenPositionRequest {
            input_token_symbol: input_token.to_string(),
            output_token_symbol: output_token.to_string(),
            input_amount_ui: format!("{:.2}", amount_usd),
            leverage,
            trade_type: trade_type.to_string(),
            owner,
            take_profit: tp,
            stop_loss: sl,
            slippage_percentage: slippage,
        };
        debug!("POST {} {:?}", url, body);
        let resp = self.client.post(&url).json(&body).send().await?;
        let data: OpenPositionResponse = resp.json().await?;
        if let Some(ref err) = data.err {
            warn!("Open position error: {}", err);
        }
        Ok(data)
    }

    // --- Close Position ---

    pub async fn build_close_position(
        &self,
        position_key: &str,
        close_usd: f64,
        withdraw_token: &str,
        slippage: &str,
    ) -> Result<ClosePositionResponse> {
        let url = format!("{}/transaction-builder/close-position", self.base_url);
        let body = ClosePositionRequest {
            position_key: position_key.to_string(),
            input_usd_ui: format!("{:.2}", close_usd),
            withdraw_token_symbol: withdraw_token.to_string(),
            slippage_percentage: Some(slippage.to_string()),
        };
        debug!("POST {} {:?}", url, body);
        let resp = self.client.post(&url).json(&body).send().await?;
        let data: ClosePositionResponse = resp.json().await?;
        if let Some(ref err) = data.err {
            warn!("Close position error: {}", err);
        }
        Ok(data)
    }

    // --- Trigger Orders ---

    #[allow(dead_code)]
    pub async fn build_trigger_order(
        &self,
        owner: &str,
        position_key: &str,
        order_type: &str,
        price: f64,
        slippage: &str,
    ) -> Result<PlaceTriggerResponse> {
        let url = format!("{}/transaction-builder/place-trigger-order", self.base_url);
        let body = PlaceTriggerRequest {
            owner: owner.to_string(),
            position_key: position_key.to_string(),
            order_type: order_type.to_string(),
            price_ui: format!("{:.2}", price),
            slippage_percentage: Some(slippage.to_string()),
        };
        let resp = self.client.post(&url).json(&body).send().await?;
        let data: PlaceTriggerResponse = resp.json().await?;
        if let Some(ref err) = data.err {
            warn!("Trigger order error: {}", err);
        }
        Ok(data)
    }

    // --- Pool Data ---

    pub async fn get_pool_data(&self) -> Result<Vec<serde_json::Value>> {
        let url = format!("{}/pool-data", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let wrapper: PoolDataWrapper = resp.json().await?;
        Ok(wrapper.pools)
    }

    /// Fetch pool data and extract parsed info for a specific market.
    ///
    /// Parses the raw `/pool-data` response to find long/short utilization and AUM
    /// for the given market symbol. Returns `None` if no matching data is found.
    pub async fn get_pool_snapshot_for_market(&self, market: &str) -> Result<Option<RawPoolInfo>> {
        let pools = self.get_pool_data().await?;
        Ok(parse_pool_data_for_market(&pools, market))
    }

    // --- Preview Endpoints ---

    /// Preview exit fee for closing a position. Returns the fee amount in USD.
    /// For paper trading we can use a fake position_key since we just want the fee estimate.
    pub async fn preview_exit_fee(&self, position_key: &str, close_usd: f64) -> Result<f64> {
        let url = format!("{}/preview/exit-fee", self.base_url);
        let body = serde_json::json!({
            "positionKey": position_key,
            "closeAmountUsd": format!("{:.2}", close_usd),
        });
        debug!("POST {}", url);
        let resp = self.client.post(&url).json(&body).send().await?;
        let data: serde_json::Value = resp.json().await?;
        // Response typically has a "fee" or "fees" field in USD
        data.get("fee")
            .or_else(|| data.get("fees"))
            .and_then(|v| v.as_str().or_else(|| v.as_f64().map(|_| "ok")))
            .and_then(|s| {
                if let Ok(f) = s.parse::<f64>() {
                    return Some(f);
                }
                // might be nested
                None
            })
            .or_else(|| data.get("fee").and_then(|v| v.as_f64()))
            .or_else(|| data.get("fees").and_then(|v| v.as_f64()))
            .ok_or_else(|| {
                anyhow::anyhow!("could not parse exit fee from preview response: {:?}", data)
            })
    }
}

/// Parse raw pool data entries to extract utilization for a specific market.
///
/// Handles two common Flash Trade API formats:
/// 1. Flat entries with `asset`/`symbol` + `side` + `utilization` fields (per-custody)
/// 2. Pool entries with nested `custodies` array
///
/// Returns `None` if no matching entries are found.
fn parse_pool_data_for_market(pools: &[serde_json::Value], market: &str) -> Option<RawPoolInfo> {
    let market_lower = market.to_lowercase();

    let mut aum_usd = 0.0_f64;
    let mut long_utilization = 0.0_f64;
    let mut short_utilization = 0.0_f64;
    let mut found_long = false;
    let mut found_short = false;
    let mut found_aum = false;

    // Pass 1: flat entries with side field (per-custody format)
    for pool in pools {
        let asset = pool
            .get("asset")
            .or_else(|| pool.get("symbol"))
            .or_else(|| pool.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if asset.to_lowercase() != market_lower {
            continue;
        }

        // Extract AUM
        if !found_aum
            && let Some(aum_val) = pool.get("aumUsd").or_else(|| pool.get("aum_usd"))
            && let Some(v) = aum_val
                .as_str()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| aum_val.as_f64())
        {
            aum_usd = v;
            found_aum = true;
        }

        let side = pool
            .get("side")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        let util = pool
            .get("utilization")
            .and_then(|v| {
                v.as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| v.as_f64())
            })
            .unwrap_or(0.0);

        if side == "long" {
            long_utilization = util;
            found_long = true;
        } else if side == "short" {
            short_utilization = util;
            found_short = true;
        }
    }

    // Pass 2: nested custodies format
    if !found_long || !found_short {
        for pool in pools {
            let asset = pool
                .get("asset")
                .or_else(|| pool.get("symbol"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if asset.to_lowercase() != market_lower {
                continue;
            }

            if let Some(custodies) = pool.get("custodies").and_then(|v| v.as_array()) {
                for custody in custodies {
                    let side = custody
                        .get("side")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_lowercase();
                    let util = custody
                        .get("utilization")
                        .and_then(|v| {
                            v.as_str()
                                .and_then(|s| s.parse::<f64>().ok())
                                .or_else(|| v.as_f64())
                        })
                        .unwrap_or(0.0);

                    if side == "long" {
                        long_utilization = util;
                        found_long = true;
                    } else if side == "short" {
                        short_utilization = util;
                        found_short = true;
                    }
                }
            }

            if !found_aum
                && let Some(aum_val) = pool.get("aumUsd").or_else(|| pool.get("aum_usd"))
                && let Some(v) = aum_val
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| aum_val.as_f64())
            {
                aum_usd = v;
                found_aum = true;
            }
        }
    }

    if found_long || found_short {
        Some(RawPoolInfo {
            aum_usd,
            long_utilization,
            short_utilization,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod pool_parser_tests {
    use super::*;

    #[test]
    fn test_parse_flat_per_custody_format() {
        let raw = vec![
            serde_json::json!({
                "asset": "SOL",
                "side": "Long",
                "utilization": "0.45",
                "aumUsd": "5000000.0"
            }),
            serde_json::json!({
                "asset": "SOL",
                "side": "Short",
                "utilization": "0.30",
            }),
            serde_json::json!({
                "asset": "BTC",
                "side": "Long",
                "utilization": "0.60",
            }),
        ];

        let result = parse_pool_data_for_market(&raw, "SOL");
        assert!(result.is_some(), "Should find pool data for SOL");
        let info = result.unwrap();
        assert!((info.aum_usd - 5_000_000.0).abs() < 0.01);
        assert!((info.long_utilization - 0.45).abs() < 0.001);
        assert!((info.short_utilization - 0.30).abs() < 0.001);
    }

    #[test]
    fn test_parse_nested_custodies_format() {
        let raw = vec![serde_json::json!({
            "asset": "SOL",
            "aumUsd": "3000000.0",
            "custodies": [
                {"side": "Long", "utilization": "0.55"},
                {"side": "Short", "utilization": "0.25"}
            ]
        })];

        let result = parse_pool_data_for_market(&raw, "SOL");
        assert!(result.is_some());
        let info = result.unwrap();
        assert!((info.aum_usd - 3_000_000.0).abs() < 0.01);
        assert!((info.long_utilization - 0.55).abs() < 0.001);
        assert!((info.short_utilization - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_parse_returns_none_for_unknown_market() {
        let raw = vec![serde_json::json!({
            "asset": "SOL",
            "side": "Long",
            "utilization": "0.45",
        })];

        let result = parse_pool_data_for_market(&raw, "ETH");
        assert!(
            result.is_none(),
            "Should return None for market not in data"
        );
    }

    #[test]
    fn test_parse_case_insensitive_market_match() {
        let raw = vec![
            serde_json::json!({
                "asset": "sol",
                "side": "Long",
                "utilization": "0.40",
            }),
            serde_json::json!({
                "asset": "sol",
                "side": "Short",
                "utilization": "0.20",
            }),
        ];

        let result = parse_pool_data_for_market(&raw, "SOL");
        assert!(result.is_some());
        let info = result.unwrap();
        assert!((info.long_utilization - 0.40).abs() < 0.001);
        assert!((info.short_utilization - 0.20).abs() < 0.001);
    }

    #[test]
    fn test_parse_handles_numeric_utilization() {
        let raw = vec![
            serde_json::json!({
                "asset": "BTC",
                "side": "Long",
                "utilization": 0.65,
                "aumUsd": "10000000.0"
            }),
            serde_json::json!({
                "asset": "BTC",
                "side": "Short",
                "utilization": 0.35,
            }),
        ];

        let result = parse_pool_data_for_market(&raw, "BTC");
        assert!(result.is_some());
        let info = result.unwrap();
        assert!((info.long_utilization - 0.65).abs() < 0.001);
        assert!((info.short_utilization - 0.35).abs() < 0.001);
        assert!((info.aum_usd - 10_000_000.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_empty_pools() {
        let raw: Vec<serde_json::Value> = vec![];
        let result = parse_pool_data_for_market(&raw, "SOL");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_partial_data_only_long() {
        let raw = vec![serde_json::json!({
            "asset": "SOL",
            "side": "Long",
            "utilization": "0.45",
        })];

        let result = parse_pool_data_for_market(&raw, "SOL");
        assert!(
            result.is_some(),
            "Should return data even with only long side"
        );
        let info = result.unwrap();
        assert!((info.long_utilization - 0.45).abs() < 0.001);
        assert!(
            (info.short_utilization - 0.0).abs() < 0.001,
            "Missing short should default to 0.0"
        );
    }
}
