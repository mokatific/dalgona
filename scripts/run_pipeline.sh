#!/usr/bin/env bash
# run_pipeline.sh — Dalgona Full Pipeline Orchestrator
#
# Runs all 6 stages of the strategy poaching pipeline:
#   1. Discover   → Scrape Hyperliquid leaderboards via QuickNode
#   2. Analyze    → Cluster fills, classify strategies, generate blueprints
#   3. Implement  → Inject blueprint parameters into Rust config
#   4. Backtest   → Validate against historical Hyperliquid candles
#   5. Paper Trade → Confirm against live Flash Trade prices
#   6. Live       → Execute with real capital (REQUIRES HUMAN APPROVAL)
#
# Usage:
#   ./scripts/run_pipeline.sh                    # Run stages 1-4 (safe)
#   ./scripts/run_pipeline.sh --from 3           # Start from stage 3
#   ./scripts/run_pipeline.sh --to 5             # Run up to stage 5
#   ./scripts/run_pipeline.sh --stage 4          # Run only stage 4
#   ./scripts/run_pipeline.sh --paper            # Run stages 1-5
#   ./scripts/run_pipeline.sh --live             # Run all 6 (needs approval)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
FROM_STAGE=1
TO_STAGE=4
BACKTEST_START=""
BACKTEST_END=""
BACKTEST_INTERVAL="5m"
BACKTEST_BALANCE=1000
PAPER_BALANCE=1000
PAPER_DURATION=""
MARKETS="BTC,SOL,ETH"
DISCOVER_MARKETS=230
QUICKNODE_URL="${QUICKNODE_HL_URL:-}"
MIN_CONFIDENCE=0.5
MIN_TRADES=100
KEYPAIR=""
LIVE_MARKET="SOL"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Dalgona Pipeline — Autonomous Strategy Poacher${NC}  ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

stage_header() {
    local num=$1 name=$2
    echo ""
    echo -e "${BLUE}━━━ Stage ${num}: ${name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
info() { echo -e "  ${CYAN}→${NC} $1"; }

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --from)        FROM_STAGE="$2"; shift 2 ;;
        --to)          TO_STAGE="$2"; shift 2 ;;
        --stage)       FROM_STAGE="$2"; TO_STAGE="$2"; shift 2 ;;
        --paper)       TO_STAGE=5; shift ;;
        --live)        TO_STAGE=6; shift ;;
        --markets)     MARKETS="$2"; shift 2 ;;
        --start)       BACKTEST_START="$2"; shift 2 ;;
        --end)         BACKTEST_END="$2"; shift 2 ;;
        --interval)    BACKTEST_INTERVAL="$2"; shift 2 ;;
        --balance)     BACKTEST_BALANCE="$2"; PAPER_BALANCE="$2"; shift 2 ;;
        --keypair)     KEYPAIR="$2"; shift 2 ;;
        --quicknode)   QUICKNODE_URL="$2"; shift 2 ;;
        --min-conf)    MIN_CONFIDENCE="$2"; shift 2 ;;
        --min-trades)  MIN_TRADES="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "  --from N         Start from stage N (1-6)"
            echo "  --to N           Stop after stage N (1-6)"
            echo "  --stage N        Run only stage N"
            echo "  --paper          Run stages 1-5"
            echo "  --live           Run stages 1-6 (needs approval)"
            echo "  --markets M      Comma-separated markets (default: BTC,SOL,ETH)"
            echo "  --start DATE     Backtest start (ISO 8601 or YYYY-MM-DD)"
            echo "  --end DATE       Backtest end (default: now)"
            echo "  --interval I     Candle interval (default: 5m)"
            echo "  --balance N      Starting balance (default: 1000)"
            echo "  --keypair PATH   Solana keypair for live trading"
            echo "  --quicknode URL  QuickNode HyperCore endpoint"
            echo "  --min-conf N     Min blueprint confidence (default: 0.5)"
            echo "  --min-trades N   Min trades for blueprint (default: 100)"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

banner
info "Pipeline: Stage ${FROM_STAGE} → Stage ${TO_STAGE}"
info "Markets: ${MARKETS}"

# Track which strategies pass each gate
STRATEGIES=""

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: DISCOVER
# ═══════════════════════════════════════════════════════════════════
if [[ $FROM_STAGE -le 1 && $TO_STAGE -ge 1 ]]; then
    stage_header 1 "Discover (Scrape Hyperliquid Leaderboards)"

    if [[ -z "$QUICKNODE_URL" ]]; then
        # Try .env file
        if [[ -f .env ]]; then
            QUICKNODE_URL=$(grep QUICKNODE_HL_URL .env | cut -d'"' -f2 || true)
        fi
    fi

    if [[ -z "$QUICKNODE_URL" ]]; then
        warn "No QuickNode URL found. Set QUICKNODE_HL_URL or --quicknode"
        warn "Using existing data/wallets-hl.json if available"
    else
        info "QuickNode endpoint configured"
        info "Discovering wallets from ${DISCOVER_MARKETS} HL markets..."

        cargo run --release --bin scrape-leaderboards -- \
            --source hyperliquid \
            --quicknode-url "$QUICKNODE_URL" \
            --output data/wallets-hl.json \
            --discover-markets "$DISCOVER_MARKETS" \
            --batch-size 5

        ok "Discovery complete"
    fi

    if [[ -f data/wallets-hl.json ]]; then
        WALLET_COUNT=$(python3 -c "import json; print(len(json.load(open('data/wallets-hl.json'))))" 2>/dev/null || echo "?")
        ok "Wallet data: data/wallets-hl.json (${WALLET_COUNT} wallets)"
    else
        fail "data/wallets-hl.json not found"
        exit 1
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: ANALYZE
# ═══════════════════════════════════════════════════════════════════
if [[ $FROM_STAGE -le 2 && $TO_STAGE -ge 2 ]]; then
    stage_header 2 "Analyze (Cluster, Classify, Generate Blueprints)"

    info "Running Python analysis pipeline..."
    python3 scripts/run_analysis.py

    BP_COUNT=$(ls -1 data/blueprints/cluster-*.json 2>/dev/null | wc -l || echo 0)
    ok "Generated ${BP_COUNT} cluster blueprints in data/blueprints/"

    if [[ -f data/reports/clusters-report.json ]]; then
        ok "Cluster report: data/reports/clusters-report.json"
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 3: IMPLEMENT (Blueprint → Config)
# ═══════════════════════════════════════════════════════════════════
if [[ $FROM_STAGE -le 3 && $TO_STAGE -ge 3 ]]; then
    stage_header 3 "Implement (Inject Blueprints into Config)"

    info "Injecting qualified blueprints into config/perps.toml..."
    python3 scripts/inject_blueprints.py \
        --blueprints-dir data/blueprints \
        --config config/perps.toml \
        --min-confidence "$MIN_CONFIDENCE" \
        --min-trades "$MIN_TRADES"

    # Extract injected strategy names
    STRATEGIES=$(grep '^\[strategy\.blueprint-cluster' config/perps.toml \
        | sed 's/\[strategy\.\(.*\)\]/\1/' | tr '\n' ',' | sed 's/,$//')

    if [[ -z "$STRATEGIES" ]]; then
        fail "No strategies injected. Check blueprint quality."
        exit 1
    fi

    ok "Strategies injected: ${STRATEGIES}"
    info "Building Rust binary..."
    cargo build --release 2>&1 | tail -1
    ok "Build complete"
fi

# Resolve strategies if starting from stage 4+
if [[ -z "$STRATEGIES" && $FROM_STAGE -ge 4 ]]; then
    STRATEGIES=$(grep '^\[strategy\.blueprint-cluster' config/perps.toml \
        | sed 's/\[strategy\.\(.*\)\]/\1/' | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$STRATEGIES" ]]; then
        fail "No blueprint strategies in config. Run stage 3 first."
        exit 1
    fi
    info "Using existing strategies: ${STRATEGIES}"
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 4: BACKTEST
# ═══════════════════════════════════════════════════════════════════
if [[ $FROM_STAGE -le 4 && $TO_STAGE -ge 4 ]]; then
    stage_header 4 "Backtest (Validate on Historical HL Data)"

    # Default: last 14 days
    if [[ -z "$BACKTEST_START" ]]; then
        BACKTEST_START=$(date -d "14 days ago" +%Y-%m-%d 2>/dev/null \
            || date -v-14d +%Y-%m-%d 2>/dev/null || echo "2026-05-10")
    fi

    info "Strategies: ${STRATEGIES}"
    info "Markets: ${MARKETS}"
    info "Period: ${BACKTEST_START} → ${BACKTEST_END:-now}"
    info "Interval: ${BACKTEST_INTERVAL}, Balance: \$${BACKTEST_BALANCE}"

    BT_ARGS=(
        --backtest
        --strategies "$STRATEGIES"
        --markets "$MARKETS"
        --backtest-start "$BACKTEST_START"
        --backtest-interval "$BACKTEST_INTERVAL"
        --paper-balance "$BACKTEST_BALANCE"
    )
    if [[ -n "$BACKTEST_END" ]]; then
        BT_ARGS+=(--backtest-end "$BACKTEST_END")
    fi

    ./target/release/dalgona "${BT_ARGS[@]}"
    ok "Backtest complete"

    # Evaluate gate
    info "Evaluating backtest gate..."
    python3 scripts/evaluate_gate.py --stage backtest

    # Check if any strategies passed
    PASSED=$(python3 -c "
import json
s = json.load(open('data/pipeline-state.json'))
passed = [k for k,v in s.items() if v.get('backtest_passed')]
print(','.join(passed))
" 2>/dev/null || echo "")

    if [[ -n "$PASSED" ]]; then
        STRATEGIES="$PASSED"
        ok "Promoted to paper: ${STRATEGIES}"
    else
        warn "No strategies passed backtest gate"
        if [[ $TO_STAGE -gt 4 ]]; then
            fail "Pipeline halted — no strategies qualified for paper trading"
            exit 1
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 5: PAPER TRADE
# ═══════════════════════════════════════════════════════════════════
if [[ $FROM_STAGE -le 5 && $TO_STAGE -ge 5 ]]; then
    stage_header 5 "Paper Trade (Live Prices, Simulated PnL)"

    info "Strategies: ${STRATEGIES}"
    info "Markets: ${MARKETS}"
    info "Balance: \$${PAPER_BALANCE}"
    warn "PAPER MODE — no real transactions"
    info "Press Ctrl+C to stop paper trading and evaluate"

    ./target/release/dalgona \
        --paper \
        --strategies "$STRATEGIES" \
        --markets "$MARKETS" \
        --paper-balance "$PAPER_BALANCE" \
        --paper-output data/paper-results

    ok "Paper trading session complete"

    info "Evaluating paper gate..."
    python3 scripts/evaluate_gate.py --stage paper

    PASSED=$(python3 -c "
import json
s = json.load(open('data/pipeline-state.json'))
passed = [k for k,v in s.items() if v.get('paper_passed')]
print(','.join(passed))
" 2>/dev/null || echo "")

    if [[ -n "$PASSED" ]]; then
        STRATEGIES="$PASSED"
        ok "Paper gate passed: ${STRATEGIES}"
    else
        warn "No strategies passed paper gate"
    fi
fi

# ═══════════════════════════════════════════════════════════════════
# STAGE 6: LIVE
# ═══════════════════════════════════════════════════════════════════
if [[ $FROM_STAGE -le 6 && $TO_STAGE -ge 6 ]]; then
    stage_header 6 "Live Trading (REAL CAPITAL)"

    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  WARNING: LIVE TRADING WITH REAL CAPITAL     ║${NC}"
    echo -e "${RED}║  This will execute real on-chain transactions ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    if [[ -z "$KEYPAIR" ]]; then
        KEYPAIR="${SOLANA_KEYPAIR:-}"
    fi
    if [[ -z "$KEYPAIR" ]]; then
        fail "No keypair. Set --keypair or SOLANA_KEYPAIR env var"
        exit 1
    fi

    info "Strategies: ${STRATEGIES}"
    info "Market: ${LIVE_MARKET}"
    info "Keypair: ${KEYPAIR}"
    echo ""

    read -p "Type 'EXECUTE' to confirm live trading: " CONFIRM
    if [[ "$CONFIRM" != "EXECUTE" ]]; then
        warn "Live trading cancelled"
        exit 0
    fi

    info "Starting live trading..."
    ./target/release/dalgona \
        --keypair "$KEYPAIR" \
        --market "$LIVE_MARKET" \
        --strategy "$(echo "$STRATEGIES" | cut -d',' -f1)"

    ok "Live trading session ended"
fi

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━ Pipeline Complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
if [[ -f data/pipeline-state.json ]]; then
    info "Pipeline state: data/pipeline-state.json"
fi
info "Backtest results: data/backtest-results/summary.json"
info "Backtest trades: data/backtest-trades.json"
if [[ -d data/paper-results ]]; then
    info "Paper results: data/paper-results/summary.json"
fi
echo ""
