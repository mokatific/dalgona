use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub agent: AgentConfig,
    pub flash: FlashConfig,
    pub strategy: StrategyConfig,
    pub risk: RiskConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub poll_interval_secs: u64,
    pub log_level: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlashConfig {
    pub api_url: String,
    pub rpc_url: String,
    pub keypair_path: String,
    pub market: String,
    pub input_token: String,
    pub pool: String,
    pub leverage: f64,
    pub slippage_pct: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategyConfig {
    pub direction_bias: String,
    pub momentum_threshold_pct: f64,
    pub lookback_count: usize,
    pub scale_in_clips: u32,
    pub clip_size_usd: f64,
    pub max_hold_secs: u64,
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub trailing_stop_pct: f64,
    pub trailing_activation_pct: f64,
    pub cooldown_after_loss_secs: u64,
    pub use_native_tp_sl: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RiskConfig {
    pub max_position_notional_usd: f64,
    pub max_daily_loss_usd: f64,
    pub max_drawdown_pct: f64,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.agent.poll_interval_secs)
    }
}
