# Dalgona — Autonomous Strategy Poacher ⚡️

An opinionated, semi-autonomous system that discovers promising perp strategies from Hyperliquid wallets, reconstructs them from fill-level data, and tests/replicates them on Flash Trade (Solana perps). Research and analysis happen on Hyperliquid; execution and paper/live trading happen on Flash Trade. Human approval is required before any live deployment.

✨ Highlights

- Discover profitable wallet behavior from Hyperliquid leaderboards
- Reconstruct entry/exit logic from fill-level data
- Generate data-driven blueprints (Python) and implement them as Rust strategies
- Backtest, paper trade, then (with human approval) go live

---

## Pipeline: Discover → Analyze → Implement → Validate 🔁

1. Discovery (Intelligence, Hyperliquid) — Scrape leaderboards and wallet fill data via QuickNode HyperCore.
2. Analysis (Python) — Cluster fills into positions, compute wallet metrics, classify strategy types, generate blueprints.
3. Implementation (Rust / Flash Trade) — Convert blueprints into pluggable `Strategy` modules and run them against Flash Trade prices.
4. Validation — Backtest on HL candles, paper trade live prices, require human sign-off before live trading.

---

## Why split Intelligence ↔ Execution? 🔍 → ⚙️

Hyperliquid provides rich per-wallet fill data (`userFills` / `userFillsByTime`) that we use to reverse-engineer strategy behavior. Flash Trade provides execution primitives and price/position state — the best place to test and execute the resulting strategies.

| Layer        |                Platform | Role                                                        |
| ------------ | ----------------------: | ----------------------------------------------------------- |
| Intelligence | Hyperliquid / QuickNode | Wallet discovery, fill-level analysis, blueprint generation |
| Execution    |    Flash Trade (Solana) | Market scanning, paper trading, live execution              |

---

## Strategies (pluggable Rust modules)

Currently implemented strategy types (each is a pluggable `Strategy`):

- momentum-scalper — edge: short bursts of velocity in thin markets; hold: minutes ⏱️
- lp-consumption — edge: liquidity provider exhaustion; hold: minutes→hours 🔁
- mean-reversion — edge: fade quick deviations from SMA; hold: minutes 🔄
- trend-follower — edge: confirmed breakouts with wider stops; hold: hours 📈

Core loop: price → detect entry → open position → monitor exits → close → record trade.

Exit rules include TP/SL, trailing stops, time stops, momentum loss, and reversal detection.

---

## Quick Start 🚀

Build and run locally:

```bash
# Build
cargo build --release

# Run Rust tests
cargo test
```

Backtest example (replay Hyperliquid candles, no wallet needed):

```bash
./target/release/dalgona --backtest \
  --strategies momentum-scalper,mean-reversion \
  --markets BTC,SOL,ETH \
  --backtest-start 2026-05-01 --backtest-end 2026-05-15 \
  --backtest-interval 5m --paper-balance 1000
```

Paper trading (live prices, simulated PnL):

```bash
./target/release/dalgona --paper \
  --strategies momentum-scalper,lp-consumption \
  --markets SOL,BTC,ETH --paper-balance 1000
```

Dry run (preview one cycle):

```bash
./target/release/dalgona --dry-run
```

Live (requires funded Solana keypair and human approval):

```bash
./target/release/dalgona --keypair ~/.config/solana/id.json --market SOL
```

Wallet discovery (QuickNode HyperCore required):

```bash
export QUICKNODE_HL_URL="https://your-endpoint.quiknode.pro/your-token/"
cargo run --bin scrape-leaderboards -- --quicknode-url $QUICKNODE_HL_URL --output data/wallets-hl.json
```

Analyze discovered wallets (Rust):

```bash
cargo run --bin analyze-wallet -- --wallets data/wallets-hl.json --output data/reports/
```

Run Python analysis tests:

```bash
python -m pytest analysis/tests/ -v
```

---

## Backtesting & Output 📊

- Uses Hyperliquid `candleSnapshot` as data source. Intervals: 1m, 5m, 15m, 1h, 4h, 1d, 1w.
- Outputs: `data/backtest-results/summary.json` and `data/backtest-trades.json` (detailed simulated trades).
- Metrics: net/gross PnL, fees, win rate, Sharpe, max drawdown, avg hold time, best/worst trade.

---

## Analysis Pipeline (Python)

Key analysis modules:

- `analysis/position_clustering.py` — cluster fills into position cycles
- `analysis/wallet_metrics.py` — per-wallet metrics (consistency, hold time, PnL distribution)
- `analysis/strategy_classifier.py` — infer strategy types with supporting evidence
- `analysis/entry_reconstruction.py` — reconstruct entry triggers from candles
- `analysis/cluster_analysis.py` — group wallets running similar strategies
- `analysis/blueprint_generator.py` — produce data-derived strategy blueprints

Run the analysis tests with `python -m pytest analysis/tests/ -v`.

---

## Configuration

Edit `config/perps.toml` to set agent, flash, strategy, and risk parameters. Example highlights:

```toml
[agent]
poll_interval_secs = 300
log_level = "info"

[flash]
market = "SOL"
leverage = 10.0
input_token = "USDC"
pool = "Crypto.1"
slippage_pct = "0.5"
```

Per-strategy and risk overrides live in the same file — see `config/perps.toml` for full schema.

---

## Architecture (quick map)

src/ — Rust core: `main.rs`, `strategy.rs`, `backtest.rs`, `flash_api.rs`, `executor.rs`, `risk.rs`, `engine.rs`, `paper.rs`.

analysis/ — Python analysis code and unit tests.

CLI bins:

- `scrape-leaderboards` — discover wallets (QuickNode HyperCore)
- `analyze-wallet` — analyze wallets & generate blueprints

---

## Modes

- Backtest (`--backtest`) — replay historical candles (safe)
- Paper (`--paper`) — live prices, simulated PnL (safe)
- Dry run (`--dry-run`) — single preview cycle
- Live (default) — real trading (requires funding and approval)

Start with Backtest → Paper → Live (human sign-off required).

---

## Risk Controls ⚠️

- Daily loss limits, max drawdown circuit breaker
- Cross-cell exposure caps (`max_total_notional_usd`)
- Position sizing, cooldowns after losses, leverage limits
- TP/SL/trailing/time stops, and optional on-chain trigger orders
- Real USDC balance checks prior to sending transactions

---

## Testing

- Rust: `cargo test` (≈140 unit tests)
- Python: `python -m pytest analysis/tests/ -v`

---

## External APIs

- QuickNode HyperCore — wallet/fill data (requires endpoint)
- Hyperliquid Info — fills & candle snapshots
- Flash Trade — prices, positions, tx builder
- Dune / Solana RPC — analytics and transaction submission

---

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See the `LICENSE` file for the full text.

Copyright (C) 2026 Mokatific Studio
