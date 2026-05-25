#!/usr/bin/env python3
"""evaluate_gate.py — Pipeline Gate Evaluator

Evaluates backtest or paper trading results to determine whether
strategies should be promoted to the next pipeline stage.

Gates:
  backtest -> paper:   Sharpe >= 0.5, net_pnl > 0, win_rate >= 35%
  paper    -> live:    Sharpe >= 1.0, net_pnl > 0, win_rate >= 40%, trades >= 10

Usage:
    python scripts/evaluate_gate.py --stage backtest
    python scripts/evaluate_gate.py --stage paper --results data/paper-results/summary.json
"""

import argparse
import json
import sys
from pathlib import Path

# Gate thresholds
BACKTEST_GATE = {
    "min_sharpe": 0.5,
    "min_win_rate": 35.0,
    "require_positive_pnl": True,
    "min_trades": 5,
}

PAPER_GATE = {
    "min_sharpe": 1.0,
    "min_win_rate": 40.0,
    "require_positive_pnl": True,
    "min_trades": 10,
}


def evaluate_backtest(results_path="data/backtest-results/summary.json"):
    """Evaluate backtest results against promotion gates."""
    p = Path(results_path)
    if not p.exists():
        print(f"Error: {results_path} not found. Run backtest first.")
        return [], []

    with open(p) as f:
        data = json.load(f)

    cells = data.get("cells", [])
    if not cells:
        print("No backtest cells found.")
        return [], []

    gate = BACKTEST_GATE
    passed = []
    failed = []

    print(f"\n{'Strategy':<30} {'Market':<8} {'Trades':>6} {'WinR%':>6} "
          f"{'Sharpe':>7} {'NetPnL':>10} {'Result':<8}")
    print("-" * 85)

    for cell in cells:
        name = cell["strategy"]
        market = cell["market"]
        key = f"{name}:{market}"
        trades = cell.get("trade_count", 0)
        wr = cell.get("win_rate", 0)
        sharpe = cell.get("sharpe_ratio", 0)
        net = cell.get("net_pnl", 0)

        reasons = []
        if trades < gate["min_trades"]:
            reasons.append(f"trades({trades})<{gate['min_trades']}")
        if wr < gate["min_win_rate"]:
            reasons.append(f"WR({wr:.1f}%)<{gate['min_win_rate']}%")
        if sharpe < gate["min_sharpe"]:
            reasons.append(f"Sharpe({sharpe:.2f})<{gate['min_sharpe']}")
        if gate["require_positive_pnl"] and net <= 0:
            reasons.append(f"PnL(${net:.2f})<=0")

        ok = len(reasons) == 0
        tag = "PASS" if ok else "FAIL"
        print(f"{name:<30} {market:<8} {trades:>6} {wr:>5.1f}% "
              f"{sharpe:>7.2f} ${net:>9.2f} {tag:<8}")

        if ok:
            passed.append(key)
        else:
            failed.append((key, reasons))

    print(f"\nPassed: {len(passed)}/{len(cells)}")
    if failed:
        print("Failed reasons:")
        for key, reasons in failed:
            print(f"  {key}: {', '.join(reasons)}")

    return passed, failed


def evaluate_paper(results_path="data/paper-results/summary.json"):
    """Evaluate paper trading results against promotion gates."""
    p = Path(results_path)
    if not p.exists():
        print(f"Error: {results_path} not found. Run paper trading first.")
        return [], []

    with open(p) as f:
        data = json.load(f)

    results = data.get("results", data.get("cells", []))
    if not results:
        print("No paper results found.")
        return [], []

    gate = PAPER_GATE
    passed = []
    failed = []

    print(f"\n{'Strategy':<30} {'Market':<8} {'Trades':>6} {'WinR%':>6} "
          f"{'Sharpe':>7} {'NetPnL':>10} {'Fees':>8} {'Result':<8}")
    print("-" * 95)

    for cell in results:
        name = cell.get("strategy", "")
        market = cell.get("market", "")
        key = f"{name}:{market}"
        trades = cell.get("trade_count", 0)
        wr = cell.get("win_rate", 0)
        sharpe = cell.get("sharpe_ratio", 0)
        net = cell.get("net_pnl", 0)
        fees = cell.get("total_fees", 0)

        reasons = []
        if trades < gate["min_trades"]:
            reasons.append(f"trades({trades})<{gate['min_trades']}")
        if wr < gate["min_win_rate"]:
            reasons.append(f"WR({wr:.1f}%)<{gate['min_win_rate']}%")
        if sharpe < gate["min_sharpe"]:
            reasons.append(f"Sharpe({sharpe:.2f})<{gate['min_sharpe']}")
        if gate["require_positive_pnl"] and net <= 0:
            reasons.append(f"PnL(${net:.2f})<=0")

        ok = len(reasons) == 0
        tag = "PASS" if ok else "FAIL"
        print(f"{name:<30} {market:<8} {trades:>6} {wr:>5.1f}% "
              f"{sharpe:>7.2f} ${net:>9.2f} ${fees:>7.2f} {tag:<8}")

        if ok:
            passed.append(key)
        else:
            failed.append((key, reasons))

    print(f"\nPassed: {len(passed)}/{len(results)}")
    return passed, failed


def update_pipeline_state(stage, passed, failed):
    """Update data/pipeline-state.json with gate results."""
    state_path = Path("data/pipeline-state.json")
    state = {}
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    for key in passed:
        strat = key.split(":")[0]
        if strat not in state:
            state[strat] = {"discovered_at": now}
        if stage == "backtest":
            state[strat]["backtest_passed"] = True
            state[strat]["backtest_evaluated_at"] = now
            state[strat]["stage"] = "paper"
        elif stage == "paper":
            state[strat]["paper_passed"] = True
            state[strat]["paper_evaluated_at"] = now
            state[strat]["stage"] = "awaiting_approval"

    for key, reasons in failed:
        strat = key.split(":")[0]
        if strat not in state:
            state[strat] = {"discovered_at": now}
        if stage == "backtest":
            state[strat]["backtest_passed"] = False
            state[strat]["backtest_fail_reasons"] = reasons
            state[strat]["stage"] = "backtest_failed"
        elif stage == "paper":
            state[strat]["paper_passed"] = False
            state[strat]["paper_fail_reasons"] = reasons
            state[strat]["stage"] = "paper_failed"

    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\nPipeline state updated: {state_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate pipeline gate")
    ap.add_argument("--stage", required=True, choices=["backtest", "paper"])
    ap.add_argument("--results", help="Path to results JSON")
    args = ap.parse_args()

    print(f"=== Pipeline Gate: {args.stage} ===")

    if args.stage == "backtest":
        rp = args.results or "data/backtest-results/summary.json"
        passed, failed = evaluate_backtest(rp)
    else:
        rp = args.results or "data/paper-results/summary.json"
        passed, failed = evaluate_paper(rp)

    update_pipeline_state(args.stage, passed, failed)

    if passed:
        # Extract unique strategy names
        strats = sorted(set(k.split(":")[0] for k in passed))
        print(f"\nPromoted strategies: {', '.join(strats)}")
        if args.stage == "backtest":
            print(f"Next: paper trade with:")
            s = ",".join(strats)
            print(f"  cargo run -- --paper --strategies {s} --markets BTC,SOL,ETH")
        elif args.stage == "paper":
            print(f"\n*** HUMAN APPROVAL REQUIRED ***")
            print(f"Strategies ready for live: {', '.join(strats)}")
            print(f"Review data/pipeline-state.json then run:")
            s = ",".join(strats)
            print(f"  cargo run -- --strategies {s} --markets SOL --keypair ~/.config/solana/id.json")
        return 0
    else:
        print("\nNo strategies passed the gate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
