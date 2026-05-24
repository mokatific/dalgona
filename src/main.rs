mod config;
mod engine;
mod executor;
mod flash_api;
mod risk;
mod signal;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "perps-scalper", about = "Flash Trade momentum scalper bot")]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config/perps.toml")]
    config: PathBuf,

    /// Solana keypair path (overrides config)
    #[arg(short, long)]
    keypair: Option<String>,

    /// Market to trade (overrides config)
    #[arg(short, long)]
    market: Option<String>,

    /// Dry run mode — preview only, no signing
    #[arg(long, default_value_t = false)]
    dry_run: bool,
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

    tracing::info!("=== Perps Momentum Scalper v0.2 (Flash Trade) ===");
    tracing::info!("Config: {}", config_path.display());
    tracing::info!("Market: {}", config.flash.market);
    tracing::info!("Mode: {}", if args.dry_run { "DRY RUN" } else { "LIVE" });

    if args.dry_run {
        tracing::warn!("Dry run mode — no transactions will be signed");
        run_dry(config).await
    } else {
        run_live(config).await
    }
}

async fn run_live(config: config::Config) -> anyhow::Result<()> {
    let executor = executor::Executor::new(&config.flash.rpc_url, &config.flash.keypair_path)?;
    let mut engine = engine::ScalperEngine::new(config, executor);
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
