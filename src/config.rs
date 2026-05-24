use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub agent: AgentConfig,
    pub flash: FlashConfig,
    pub strategy: StrategySection,
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

/// Top-level strategy configuration section.
///
/// Supports two config formats:
///
/// **New format** (preferred):
/// ```toml
/// [strategy]
/// active = "momentum-scalper"
///
/// [strategy.momentum-scalper]
/// direction_bias = "neutral"
/// momentum_threshold_pct = 0.15
/// # ... all fields
/// ```
///
/// **Old format** (backward compatible):
/// ```toml
/// [strategy]
/// direction_bias = "neutral"
/// momentum_threshold_pct = 0.15
/// # ... all fields directly
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategySection {
    /// The name of the active strategy (e.g., "momentum-scalper").
    /// If absent, defaults to "momentum-scalper" for backward compatibility.
    #[serde(default)]
    pub active: Option<String>,

    /// Strategy-specific parameter sub-tables.
    /// Key is the strategy name (e.g., "momentum-scalper"), value is a TOML table.
    #[serde(default)]
    pub strategies: HashMap<String, toml::Value>,

    // --- Legacy flat fields (for backward compatibility) ---
    #[serde(default = "default_direction_bias")]
    pub direction_bias: String,
    #[serde(default = "default_momentum_threshold_pct")]
    pub momentum_threshold_pct: f64,
    #[serde(default = "default_lookback_count")]
    pub lookback_count: usize,
    #[serde(default = "default_scale_in_clips")]
    pub scale_in_clips: u32,
    #[serde(default = "default_clip_size_usd")]
    pub clip_size_usd: f64,
    #[serde(default = "default_max_hold_secs")]
    pub max_hold_secs: u64,
    #[serde(default = "default_take_profit_pct")]
    pub take_profit_pct: f64,
    #[serde(default = "default_stop_loss_pct")]
    pub stop_loss_pct: f64,
    #[serde(default = "default_trailing_stop_pct")]
    pub trailing_stop_pct: f64,
    #[serde(default = "default_trailing_activation_pct")]
    pub trailing_activation_pct: f64,
    #[serde(default = "default_cooldown_after_loss_secs")]
    pub cooldown_after_loss_secs: u64,
    #[serde(default = "default_use_native_tp_sl")]
    pub use_native_tp_sl: bool,
}

fn default_direction_bias() -> String {
    "neutral".to_string()
}
fn default_momentum_threshold_pct() -> f64 {
    0.15
}
fn default_lookback_count() -> usize {
    60
}
fn default_scale_in_clips() -> u32 {
    1
}
fn default_clip_size_usd() -> f64 {
    100.0
}
fn default_max_hold_secs() -> u64 {
    1800
}
fn default_take_profit_pct() -> f64 {
    2.5
}
fn default_stop_loss_pct() -> f64 {
    1.0
}
fn default_trailing_stop_pct() -> f64 {
    0.8
}
fn default_trailing_activation_pct() -> f64 {
    1.5
}
fn default_cooldown_after_loss_secs() -> u64 {
    300
}
fn default_use_native_tp_sl() -> bool {
    true
}

impl StrategySection {
    /// Resolve the active strategy name.
    /// Priority: explicit `active` field → default "momentum-scalper".
    pub fn resolve_active(&self, cli_override: Option<&str>) -> String {
        cli_override
            .map(|s| s.to_string())
            .or_else(|| self.active.clone())
            .unwrap_or_else(|| "momentum-scalper".to_string())
    }

    /// Get the parameters for a specific strategy.
    ///
    /// If a sub-table exists for the strategy name, parse it.
    /// Otherwise, fall back to the flat legacy fields (backward compatibility).
    pub fn get_params(
        &self,
        strategy_name: &str,
    ) -> anyhow::Result<crate::strategy::StrategyParams> {
        if let Some(sub_table) = self.strategies.get(strategy_name) {
            // New format: parse the sub-table
            let params: StrategyFlatParams = sub_table.clone().try_into().map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse [strategy.{}] sub-table: {}. \
                     Config format updated. Please restructure [strategy] to include \
                     active = \"{}\" and [strategy.{}] sub-table with all strategy fields.",
                    strategy_name,
                    e,
                    strategy_name,
                    strategy_name
                )
            })?;
            Ok(params.into_strategy_params())
        } else if strategy_name == "momentum-scalper" {
            // Old format: use flat fields directly
            Ok(crate::strategy::StrategyParams {
                direction_bias: self.direction_bias.clone(),
                momentum_threshold_pct: self.momentum_threshold_pct,
                lookback_count: self.lookback_count,
                scale_in_clips: self.scale_in_clips,
                clip_size_usd: self.clip_size_usd,
                max_hold_secs: self.max_hold_secs,
                take_profit_pct: self.take_profit_pct,
                stop_loss_pct: self.stop_loss_pct,
                trailing_stop_pct: self.trailing_stop_pct,
                trailing_activation_pct: self.trailing_activation_pct,
                cooldown_after_loss_secs: self.cooldown_after_loss_secs,
                use_native_tp_sl: self.use_native_tp_sl,
            })
        } else {
            anyhow::bail!(
                "No configuration found for strategy '{}'. Available sub-tables: [{}]. \
                 Add a [strategy.{}] section to your config file.",
                strategy_name,
                self.strategies
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                strategy_name
            )
        }
    }

    /// Get the TOML sub-table for a strategy, if one exists.
    /// Returns None if the strategy uses the flat legacy config format.
    pub fn get_sub_table(&self, strategy_name: &str) -> Option<&toml::Value> {
        self.strategies.get(strategy_name)
    }
}

/// Intermediate struct for parsing strategy sub-tables from TOML.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct StrategyFlatParams {
    #[serde(default = "default_direction_bias")]
    pub direction_bias: String,
    #[serde(default = "default_momentum_threshold_pct")]
    pub momentum_threshold_pct: f64,
    #[serde(default = "default_lookback_count")]
    pub lookback_count: usize,
    #[serde(default = "default_scale_in_clips")]
    pub scale_in_clips: u32,
    #[serde(default = "default_clip_size_usd")]
    pub clip_size_usd: f64,
    #[serde(default = "default_max_hold_secs")]
    pub max_hold_secs: u64,
    #[serde(default = "default_take_profit_pct")]
    pub take_profit_pct: f64,
    #[serde(default = "default_stop_loss_pct")]
    pub stop_loss_pct: f64,
    #[serde(default = "default_trailing_stop_pct")]
    pub trailing_stop_pct: f64,
    #[serde(default = "default_trailing_activation_pct")]
    pub trailing_activation_pct: f64,
    #[serde(default = "default_cooldown_after_loss_secs")]
    pub cooldown_after_loss_secs: u64,
    #[serde(default = "default_use_native_tp_sl")]
    pub use_native_tp_sl: bool,
}

impl StrategyFlatParams {
    fn into_strategy_params(self) -> crate::strategy::StrategyParams {
        crate::strategy::StrategyParams {
            direction_bias: self.direction_bias,
            momentum_threshold_pct: self.momentum_threshold_pct,
            lookback_count: self.lookback_count,
            scale_in_clips: self.scale_in_clips,
            clip_size_usd: self.clip_size_usd,
            max_hold_secs: self.max_hold_secs,
            take_profit_pct: self.take_profit_pct,
            stop_loss_pct: self.stop_loss_pct,
            trailing_stop_pct: self.trailing_stop_pct,
            trailing_activation_pct: self.trailing_activation_pct,
            cooldown_after_loss_secs: self.cooldown_after_loss_secs,
            use_native_tp_sl: self.use_native_tp_sl,
        }
    }
}

/// Legacy type alias for backward compatibility with code that expects `StrategyConfig`.
/// This is the old flat struct; new code should use `StrategySection` + `StrategyParams`.
#[allow(dead_code)]
pub type StrategyConfig = StrategySection;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RiskConfig {
    pub max_position_notional_usd: f64,
    pub max_daily_loss_usd: f64,
    pub max_drawdown_pct: f64,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;

        // Post-process: extract strategy sub-tables from the raw TOML.
        // TOML sub-tables like [strategy.lp-consumption] are not automatically
        // picked up by serde's HashMap<String, Value> when the struct also has
        // named fields. We need to manually extract them.
        let raw: toml::Value = toml::from_str(&content)?;
        if let Some(strategy_table) = raw.get("strategy").and_then(|v| v.as_table()) {
            let known_flat_fields = [
                "active",
                "direction_bias",
                "momentum_threshold_pct",
                "lookback_count",
                "scale_in_clips",
                "clip_size_usd",
                "max_hold_secs",
                "take_profit_pct",
                "stop_loss_pct",
                "trailing_stop_pct",
                "trailing_activation_pct",
                "cooldown_after_loss_secs",
                "use_native_tp_sl",
            ];
            for (key, value) in strategy_table {
                if !known_flat_fields.contains(&key.as_str()) && value.is_table() {
                    config
                        .strategy
                        .strategies
                        .insert(key.clone(), value.clone());
                }
            }
        }

        Ok(config)
    }

    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.agent.poll_interval_secs)
    }
}
