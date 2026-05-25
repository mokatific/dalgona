mod backtest;
mod config;
mod engine;
mod executor;
mod flash_api;
mod monitor;
mod paper;
mod risk;
mod signal;
mod strategy;

use clap::Parser;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

#[derive(Parser, Debug)]
#[command(name = "dalgona", about = "Flash Trade momentum scalper bot")]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config/perps.toml")]
    config: PathBuf,

    /// Solana keypair path (overrides config, not needed for --paper)
    #[arg(short, long)]
    keypair: Option<String>,

    /// Market to trade (overrides config). For multi-market paper trading, use --markets instead.
    #[arg(short, long)]
    market: Option<String>,

    /// Strategy to use (overrides config [strategy] active field). For multi-strategy, use --strategies.
    #[arg(short, long)]
    strategy: Option<String>,

    /// Comma-separated list of strategies for multi-strategy paper trading
    /// (e.g., --strategies momentum-scalper,lp-consumption)
    #[arg(long)]
    strategies: Option<String>,

    /// Comma-separated list of markets for multi-market paper trading
    /// (e.g., --markets SOL,BTC,ETH)
    #[arg(long)]
    markets: Option<String>,

    /// Dry run mode -- single preview, no signing, then exit
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Paper trading -- full open/monitor/close loop against live prices, simulated PnL, no real transactions
    #[arg(long, default_value_t = false)]
    paper: bool,

    /// Starting balance for paper trading (USD), defaults to 1000
    #[arg(long, default_value_t = 1000.0)]
    paper_balance: f64,

    /// Output directory for paper trading results (default: data/paper-results)
    #[arg(long, default_value = "data/paper-results")]
    paper_output: String,

    /// Backtest mode -- replay Hyperliquid historical candles through strategies
    #[arg(long, default_value_t = false)]
    backtest: bool,

    /// Backtest start time (ISO 8601, e.g., "2025-05-01T00:00:00Z" or "2025-05-01")
    #[arg(long)]
    backtest_start: Option<String>,

    /// Backtest end time (ISO 8601, defaults to now)
    #[arg(long)]
    backtest_end: Option<String>,

    /// Backtest candle interval (e.g., "1m", "5m", "15m", "1h", "4h")
    #[arg(long, default_value = "5m")]
    backtest_interval: String,

    /// Backtest fee rate as decimal per side (default: 0.001 = 0.1%, matching Flash Trade base taker fee)
    #[arg(long, default_value_t = 0.001)]
    backtest_fee_rate: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config_path = if args.config.exists() {
        args.config
    } else {
        PathBuf::from("config/perps.toml")
    };

    let mut config = config::Config::load(&config_path)?;

    if let Some(market) = args.market {
        config.flash.market = market;
    }
    if let Some(keypair) = args.keypair {
        config.flash.keypair_path = keypair;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.agent.log_level)),
        )
        .with_target(false)
        .with_thread_ids(false)
        .init();

    tracing::info!("=== Dalgona Autonomous Strategy Poacher v0.2.0 (Flash Trade) ===");
    tracing::info!("Config: {}", config_path.display());
    tracing::info!("Market: {}", config.flash.market);

    // Resolve strategy name from CLI flag or config, validate early
    let strategy_name = config.strategy.resolve_active(args.strategy.as_deref());
    tracing::info!("Strategy: {}", strategy_name);

    // Validate strategy name first
    if !crate::strategy::available_strategies().contains(&strategy_name.as_str()) {
        let available = crate::strategy::available_strategies().join(", ");
        tracing::error!(
            "Unknown strategy '{}'. Available strategies: {}",
            strategy_name,
            available
        );
        std::process::exit(1);
    }

    // Validate parameters by attempting to create the strategy (dry run)
    let sub_table = config.strategy.get_sub_table(&strategy_name);
    // For strategies that have a sub-table (e.g., lp-consumption), get_params may fail
    // because they have different parameter schemas. Use flat legacy params as fallback.
    let fallback_params = config
        .strategy
        .get_params(&strategy_name)
        .unwrap_or_else(|_| {
            // Create minimal fallback params for validation
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
    if let Err(e) =
        crate::strategy::create_strategy_from_config(&strategy_name, sub_table, fallback_params)
    {
        tracing::error!("Invalid strategy parameters: {}", e);
        std::process::exit(1);
    }

    match (args.dry_run, args.paper, args.backtest) {
        (true, true, _) => {
            anyhow::bail!("Cannot use --dry-run and --paper together");
        }
        (true, _, true) | (_, true, true) => {
            anyhow::bail!("Cannot use --backtest with --dry-run or --paper");
        }
        (true, false, false) => {
            tracing::warn!("DRY RUN -- single preview, then exit");
            run_dry(config).await
        }
        (false, true, false) => {
            // Determine if this is multi-strategy multi-market mode
            let strategies_list = args
                .strategies
                .as_deref()
                .map(|s| s.split(',').map(|s| s.trim()).collect::<Vec<_>>());
            let markets_list = args.markets.as_deref().map(|s| {
                s.split(',')
                    .map(|s| s.trim().to_uppercase())
                    .collect::<Vec<_>>()
            });

            if strategies_list.is_some() || markets_list.is_some() {
                tracing::warn!("PAPER TRADING (MULTI) -- multi-strategy multi-market mode");
                let resolved_strategies =
                    strategies_list.unwrap_or_else(|| vec![strategy_name.as_str()]);
                let resolved_markets =
                    markets_list.unwrap_or_else(|| vec![config.flash.market.clone()]);
                run_multi_paper(
                    config,
                    args.paper_balance,
                    resolved_strategies,
                    resolved_markets,
                    &args.paper_output,
                )
                .await
            } else {
                tracing::warn!("PAPER TRADING -- full loop, simulated PnL, NO real transactions");
                run_paper(config, args.paper_balance, args.strategy.as_deref()).await
            }
        }
        (false, false, true) => {
            run_backtest(
                config,
                args.strategies.as_deref(),
                args.markets.as_deref(),
                args.backtest_start.as_deref(),
                args.backtest_end.as_deref(),
                &args.backtest_interval,
                args.paper_balance,
                args.backtest_fee_rate,
            )
            .await
        }
        (false, false, false) => {
            tracing::warn!("LIVE TRADING -- real transactions with real funds");
            run_live(config, args.strategy.as_deref()).await
        }
    }
}

async fn run_live(config: config::Config, strategy_name: Option<&str>) -> anyhow::Result<()> {
    let executor = executor::Executor::new(&config.flash.rpc_url, &config.flash.keypair_path)?;
    let mut engine = engine::ScalperEngine::new(config, executor, strategy_name)?;

    let running = engine.shutdown_handle();
    let _ = ctrlc::set_handler(move || {
        tracing::info!("Received shutdown signal, finishing current tick...");
        running.store(false, Ordering::Relaxed);
    });

    engine.run().await
}

async fn run_paper(
    config: config::Config,
    starting_balance: f64,
    strategy_name: Option<&str>,
) -> anyhow::Result<()> {
    let mut engine = paper::PaperEngine::new(config, starting_balance, strategy_name)?;

    let running = engine.shutdown_handle();
    let _ = ctrlc::set_handler(move || {
        tracing::info!("Received shutdown signal, finishing current tick...");
        running.store(false, Ordering::Relaxed);
    });

    engine.run().await
}

async fn run_multi_paper(
    config: config::Config,
    starting_balance: f64,
    strategy_names: Vec<&str>,
    markets: Vec<String>,
    output_dir: &str,
) -> anyhow::Result<()> {
    let mut engine = paper::MultiPaperEngine::new(
        config,
        starting_balance,
        strategy_names,
        markets,
        output_dir,
    )?;

    let running = engine.shutdown_handle();
    let _ = ctrlc::set_handler(move || {
        tracing::info!("Received shutdown signal, finishing current tick...");
        running.store(false, Ordering::Relaxed);
    });

    engine.run().await
}

async fn run_dry(config: config::Config) -> anyhow::Result<()> {
    use flash_api::FlashClient;

    let flash = FlashClient::new(&config.flash.api_url);
    let market = &config.flash.market;

    tracing::info!("--- Dry Run: Fetching {} price ---", market);
    let price = flash.get_price(market).await?;
    tracing::info!("{} price: ${:.2}", market, price);

    tracing::info!(
        "--- Dry Run: Previewing {} LONG ${:.0} @ {}x ---",
        market,
        config.strategy.clip_size_usd,
        config.flash.leverage
    );
    let preview = flash
        .preview_open_position(
            &config.flash.input_token,
            market,
            config.strategy.clip_size_usd,
            config.flash.leverage,
            "LONG",
        )
        .await?;

    if let Some(ref err) = preview.err {
        tracing::warn!("Preview error: {}", err);
    } else {
        tracing::info!("Preview result:");
        tracing::info!(
            "  Entry price: {}",
            preview.new_entry_price.as_deref().unwrap_or("N/A")
        );
        tracing::info!(
            "  Liquidation: {}",
            preview.new_liquidation_price.as_deref().unwrap_or("N/A")
        );
        tracing::info!(
            "  Entry fee: ${}",
            preview.entry_fee.as_deref().unwrap_or("N/A")
        );
        tracing::info!(
            "  Notional: ${}",
            preview.you_recieve_usd_ui.as_deref().unwrap_or("N/A")
        );
        tracing::info!(
            "  Output: {} {}",
            preview.output_amount_ui.as_deref().unwrap_or("N/A"),
            market
        );
    }

    tracing::info!("--- Dry Run: Checking pool data ---");
    match flash.get_pool_data().await {
        Ok(pools) => {
            tracing::info!("Pool data: {} pools loaded", pools.len());
        }
        Err(e) => tracing::warn!("Pool data error: {:#}", e),
    }

    tracing::info!("--- Dry run complete ---");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_backtest(
    config: config::Config,
    strategies: Option<&str>,
    markets: Option<&str>,
    start: Option<&str>,
    end: Option<&str>,
    interval: &str,
    starting_balance: f64,
    fee_rate: f64,
) -> anyhow::Result<()> {
    use chrono::Utc;

    tracing::warn!("BACKTEST -- replaying Hyperliquid historical data through strategies");

    // Parse strategies
    let strategy_names: Vec<String> = strategies
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| vec![config.strategy.resolve_active(None)]);

    // Parse markets
    let market_names: Vec<String> = markets
        .map(|s| s.split(',').map(|s| s.trim().to_uppercase()).collect())
        .unwrap_or_else(|| vec![config.flash.market.clone()]);

    // Parse start time
    let start_time_ms = parse_backtest_time(start, "start")?;

    // Parse end time (default: now)
    let end_time_ms = if let Some(end_str) = end {
        parse_backtest_time(Some(end_str), "end")?
    } else {
        Utc::now().timestamp_millis()
    };

    if start_time_ms >= end_time_ms {
        anyhow::bail!(
            "Start time ({}) must be before end time ({})",
            chrono::DateTime::from_timestamp_millis(start_time_ms)
                .map(|t| t.to_rfc3339())
                .unwrap_or_default(),
            chrono::DateTime::from_timestamp_millis(end_time_ms)
                .map(|t| t.to_rfc3339())
                .unwrap_or_default(),
        );
    }

    let leverage = config.flash.leverage;

    tracing::info!("Strategies: {}", strategy_names.join(", "));
    tracing::info!("Markets: {}", market_names.join(", "));
    tracing::info!(
        "Period: {} → {}",
        chrono::DateTime::from_timestamp_millis(start_time_ms)
            .map(|t| t.to_rfc3339())
            .unwrap_or_default(),
        chrono::DateTime::from_timestamp_millis(end_time_ms)
            .map(|t| t.to_rfc3339())
            .unwrap_or_default(),
    );
    tracing::info!("Interval: {}", interval);
    tracing::info!(
        "Fee rate: {:.2}%  |  Leverage: {}x  |  Balance: ${:.0}",
        fee_rate * 100.0,
        leverage,
        starting_balance
    );

    let bt_config = backtest::BacktestConfig {
        strategies: strategy_names,
        markets: market_names,
        start_time_ms,
        end_time_ms,
        interval: interval.to_string(),
        starting_balance,
        fee_rate,
        borrow_rate_hourly: 0.0001, // 0.01%/hr default
        leverage,
    };

    let engine = backtest::BacktestEngine::new(config, bt_config)?;
    let result = engine.run().await?;

    tracing::info!("=== Backtest complete ===");
    tracing::info!(
        "Final balance: ${:.2} (net PnL: ${:.2})",
        result.final_balance,
        result.total_net_pnl
    );

    Ok(())
}

/// Parse a backtest time string (ISO 8601 full or date-only) to milliseconds.
fn parse_backtest_time(input: Option<&str>, label: &str) -> anyhow::Result<i64> {
    let s = input
        .ok_or_else(|| anyhow::anyhow!("--backtest-{} is required in backtest mode", label))?;

    // Try full ISO 8601 datetime first
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp_millis());
    }

    // Try date-only format (e.g., "2025-05-01")
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_local_timezone(chrono::Utc)
            .single()
            .ok_or_else(|| anyhow::anyhow!("Invalid date: {}", s))?;
        return Ok(dt.timestamp_millis());
    }

    anyhow::bail!(
        "Invalid --backtest-{} time: '{}'. Use ISO 8601 (2025-05-01T00:00:00Z) or date-only (2025-05-01)",
        label,
        s
    )
}
