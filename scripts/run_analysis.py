#!/usr/bin/env python3
"""run_analysis.py — Stage 2: Analyze discovered wallets.

Clusters fills into positions, computes wallet metrics, classifies
strategy types, and generates data-driven blueprints.

Usage:
    python scripts/run_analysis.py
"""

import json
import os
import sys

# Resolve project root (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from analysis.cluster_analysis import analyze_clusters, save_results
from analysis.blueprint_generator import generate_all_blueprints

def main():
    print("=== Python Poaching & Cluster Analysis Pipeline ===")
    
    # Target directories
    os.makedirs("data/reports", exist_ok=True)
    os.makedirs("data/blueprints", exist_ok=True)
    
    input_file = "data/wallets-hl.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run discovery first.")
        sys.exit(1)
        
    print(f"Loading scraped wallets from {input_file}...")
    with open(input_file, "r") as f:
        wallets_data = json.load(f)
        
    print(f"Loaded {len(wallets_data)} wallets.")
    
    print("Clustering wallets by strategy similarity and metrics...")
    results = analyze_clusters(wallets_data)
    
    print("Saving cluster report...")
    save_results(results, "data/reports/clusters-report.json")
    print("Report saved to data/reports/clusters-report.json")
    
    print("Generating statistical strategy blueprints...")
    blueprints = generate_all_blueprints(
        results["clusters"], 
        results["profiles"], 
        "data/blueprints"
    )
    
    print(f"Pipeline complete! Generated {len(blueprints)} blueprints in data/blueprints/")
    for bp in blueprints:
        print(f"  - {bp['source_cluster_id']}: {bp['strategy_name']} (wallets: {bp['sample_size']['wallets']}, confidence: {bp['confidence_score']:.2f})")

if __name__ == "__main__":
    main()
