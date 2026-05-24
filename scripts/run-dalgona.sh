#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${PROJECT_DIR}/config/perps.toml"
KEYPAIR="${SOLANA_KEYPAIR:-}"
DRY_RUN=""
MARKET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --keypair) KEYPAIR="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        --market)  MARKET="--market $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$KEYPAIR" ]]; then
    echo "ERROR: Set SOLANA_KEYPAIR or pass --keypair <path>"
    echo "  export SOLANA_KEYPAIR=~/.config/solana/id.json"
    exit 1
fi

BIN="${PROJECT_DIR}/target/release/dalgona"
if [[ ! -f "$BIN" ]]; then
    echo "Building dalgona..."
    cd "$PROJECT_DIR" && cargo build --release
fi

exec "$BIN" \
    --config "$CONFIG" \
    --keypair "$KEYPAIR" \
    $DRY_RUN \
    $MARKET
