"""
ablation_lstm.py

Visualize the effect of LSTM hidden size on DRQN performance on
MiniGrid-MemoryS7-v0. Loads logs produced by train_drqn_minigrid.py
and plots reward + success rate curves for each size.

Usage:
    python ablation_lstm.py
    python ablation_lstm.py --sizes 64 128 256 --seeds 42 123 456
    python ablation_lstm.py --sizes 64 128 256 --seeds 42 123 456 --out results/ablation_lstm.png
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

COLORS = ['steelblue', 'darkorange', 'forestgreen', 'crimson']


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
    if len(arr) < window:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_band(ax, all_series, window, color, label):
    smoothed = [smooth(np.array(s, dtype=float), window) for s in all_series]
    smoothed = [s for s in smoothed if len(s) > 0]
    if not smoothed:
        return
    min_len = min(len(s) for s in smoothed)
    arr = np.array([s[:min_len] for s in smoothed])
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    x = np.arange(min_len)
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


def main():
    parser = argparse.ArgumentParser(description="LSTM size ablation for DRQN on MiniGrid")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256],
                        help="LSTM hidden sizes to compare")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--out", default="results/ablation_lstm.png")
    parser.add_argument("--window", type=int, default=20,
                        help="Smoothing window in episodes")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    any_loaded = False
    for size, color in zip(args.sizes, COLORS):
        exp_name = f"drqn_minigrid_lstm{size}"
        rewards, successes, found = load_experiment(args.log_dir, exp_name, args.seeds)
        if not rewards:
            continue
        any_loaded = True
        label = f"LSTM size={size} (n={len(found)})"
        plot_band(axes[0], rewards,   args.window, color, label)
        plot_band(axes[1], successes, args.window, color, label)

    if not any_loaded:
        print("No log files found. Run experiments first:")
        for size in args.sizes:
            for seed in args.seeds:
                print(f"  python train_drqn_minigrid.py --seed {seed} --lstm-size {size}")
        return

    axes[0].set_title(f"Episode Reward ({args.window}-ep avg)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Success Rate ({args.window}-ep avg)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("DRQN Ablation: LSTM Hidden Size on MiniGrid-MemoryS7-v0",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
