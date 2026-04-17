#!/usr/bin/env python
"""
Multi-seed experiment runner.

Runs DQN and DRQN training across multiple seeds and aggregates results.
Usage:
    python run_experiments.py --mode pong --seeds 42 123 456
    python run_experiments.py --mode minigrid-dqn --seeds 42 123 456
    python run_experiments.py --mode minigrid-drqn --seeds 42 123 456
"""

import os
# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import json


def run_experiment(script: str, seed: int, args: List[str]) -> bool:
    """Run a single experiment with given seed."""
    cmd = [sys.executable, script, "--seed", str(seed)] + args
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_multi_seed_experiments(
    script: str,
    experiment_name: str,
    seeds: List[int],
    additional_args: List[str] = None
):
    """Run experiments across multiple seeds."""
    from utils.logging import MultiSeedAggregator
    
    if additional_args is None:
        additional_args = []
    
    print(f"\nStarting multi-seed experiments for {experiment_name}")
    print(f"Seeds: {seeds}")
    print(f"Script: {script}")
    
    successful_seeds = []
    failed_seeds = []
    
    for seed in seeds:
        try:
            success = run_experiment(script, seed, additional_args)
            if success:
                successful_seeds.append(seed)
            else:
                failed_seeds.append(seed)
        except Exception as e:
            print(f"Error running seed {seed}: {e}")
            failed_seeds.append(seed)
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"Aggregating results for {experiment_name}")
    print(f"{'='*80}\n")
    
    aggregator = MultiSeedAggregator()
    
    try:
        results = aggregator.load_seed_results(experiment_name, successful_seeds)
        if results:
            summary = aggregator.save_aggregate_summary(experiment_name, results)
            print("\nAggregate Summary:")
            print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Error aggregating results: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Successful seeds: {successful_seeds}")
    print(f"Failed seeds: {failed_seeds}")
    
    return len(failed_seeds) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed reinforcement learning experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["pong", "minigrid-dqn", "minigrid-drqn"],
        required=True,
        help="Which experiment to run"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Seeds to use for experiments"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to train for"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps"
    )
    
    args = parser.parse_args()
    
    # Define experiments
    experiments = {
        "pong": {
            "script": "train_dqn_pong.py",
            "experiment_name": "dqn_pong",
        },
        "minigrid-dqn": {
            "script": "train_dqn_minigrid.py",
            "experiment_name": "dqn_minigrid",
        },
        "minigrid-drqn": {
            "script": "train_drqn_minigrid.py",
            "experiment_name": "drqn_minigrid",
        },
    }
    
    exp_config = experiments[args.mode]
    
    # Build additional arguments
    additional_args = []
    if args.episodes:
        additional_args.extend(["--episodes", str(args.episodes)])
    if args.max_steps:
        additional_args.extend(["--max-steps", str(args.max_steps)])
    
    # Run experiments
    success = run_multi_seed_experiments(
        script=exp_config["script"],
        experiment_name=exp_config["experiment_name"],
        seeds=args.seeds,
        additional_args=additional_args
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
