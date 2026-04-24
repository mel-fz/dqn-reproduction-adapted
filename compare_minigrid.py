"""
compare_minigrid.py

Generate a side-by-side comparison figure of DQN (feed-forward) vs DRQN
(recurrent) on MiniGrid-MemoryS7-v0, using logs produced by the training
scripts.

Usage:
    python compare_minigrid.py
    python compare_minigrid.py --seeds 42 123 456
    python compare_minigrid.py --seeds 42 123 456 --log-dir results/logs --out results/minigrid_comparison.png
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_experiment(log_dir, experiment_name, seeds):
    rewards, successes, found = [], [], []
    for seed in seeds:
        path = Path(log_dir) / f"{experiment_name}_seed{seed}.json"
        if not path.exists():
            print(f"Warning: {path} not found — skipping seed {seed}")
            continue
        with open(path) as f:
            data = json.load(f)
        episodes = data.get("episodes", [])
        rewards.append([ep["reward"] for ep in episodes])
        successes.append([
            float(ep["success"]) for ep in episodes
            if ep.get("success") is not None
        ])
        found.append(seed)
    return rewards, successes, found


def smooth(arr, window):
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_band(ax, all_series, window, color, label):
    smoothed = [smooth(np.array(s, dtype=float), window)
                for s in all_series if len(s) >= window]
    if not smoothed:
        return
    min_len = min(len(s) for s in smoothed)
    arr = np.array([s[:min_len] for s in smoothed])
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    x = np.arange(min_len)
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


def main():
    parser = argparse.ArgumentParser(description="Compare DQN vs DRQN on MiniGrid")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--out", default="results/minigrid_comparison.png")
    parser.add_argument("--window", type=int, default=20,
                        help="Smoothing window in episodes (default: 20)")
    parser.add_argument("--lstm-size", type=int, default=128,
                        help="LSTM size of the DRQN run to compare (default: 128)")
    args = parser.parse_args()

    dqn_r, dqn_s, dqn_seeds = load_experiment(args.log_dir, "dqn_minigrid", args.seeds)
    drqn_r, drqn_s, drqn_seeds = load_experiment(
        args.log_dir, f"drqn_minigrid_lstm{args.lstm_size}", args.seeds
    )

    if not dqn_r and not drqn_r:
        print("No log files found. Run experiments first with run_experiments.py.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if dqn_r:
        plot_band(ax, dqn_r, args.window, color='steelblue',
                  label=f'DQN feed-forward (n={len(dqn_seeds)})')
    if drqn_r:
        plot_band(ax, drqn_r, args.window, color='darkorange',
                  label=f'DRQN recurrent (n={len(drqn_seeds)})')
    ax.set_title(f'Episode Reward ({args.window}-ep avg)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if dqn_s:
        plot_band(ax, dqn_s, args.window, color='steelblue',
                  label=f'DQN feed-forward (n={len(dqn_seeds)})')
    if drqn_s:
        plot_band(ax, drqn_s, args.window, color='darkorange',
                  label=f'DRQN recurrent (n={len(drqn_seeds)})')
    ax.set_title(f'Success Rate ({args.window}-ep avg)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('DQN vs DRQN on MiniGrid-MemoryS7-v0', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
