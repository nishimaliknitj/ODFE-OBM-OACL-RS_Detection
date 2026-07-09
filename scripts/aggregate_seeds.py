#!/usr/bin/env python3
"""
Aggregates evaluation results across multiple seeds into mean +/- std,
reproducing the format of Table 13 in the paper.

Usage
-----
    python scripts/aggregate_seeds.py --dataset dior --seeds 42 123 2024
"""
from __future__ import annotations

import argparse
import json
import os
import statistics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["dior", "dota", "fair1m"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    map_values = []

    for seed in args.seeds:
        result_path = os.path.join(
            args.results_dir, f"odfe_obm_oacl_{args.dataset}_seed{seed}_eval_results.json"
        )
        if not os.path.exists(result_path):
            print(f"[WARN] Missing results file for seed {seed}: {result_path}")
            continue
        with open(result_path, "r") as f:
            data = json.load(f)
        map_values.append(data["mAP@0.5"] * 100)

    if not map_values:
        print("No results found. Run scripts/run_multiseed.sh first.")
        return

    mean = statistics.mean(map_values)
    std = statistics.stdev(map_values) if len(map_values) > 1 else 0.0

    print(f"Dataset: {args.dataset.upper()}")
    print(f"Seeds evaluated: {len(map_values)} / {len(args.seeds)}")
    print(f"mAP@0.5 per seed: {[f'{v:.2f}' for v in map_values]}")
    print(f"Mean +/- Std: {mean:.1f} +/- {std:.1f}")


if __name__ == "__main__":
    main()
