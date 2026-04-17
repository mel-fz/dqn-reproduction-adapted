"""
Logging utilities for tracking training metrics across experiments.

Provides JSON-based logging with support for:
- Episode-level metrics (reward, length, success)
- Training-level metrics (loss, epsilon, buffer size)
- Experiment metadata (seed, config, environment)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class ExperimentLogger:
    """Logs training metrics to JSON files for later analysis."""
    
    def __init__(
        self,
        log_dir: str = "results/logs",
        experiment_name: str = "experiment",
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (e.g., 'dqn_pong')
            seed: Random seed for reproducibility
            config: Dictionary of hyperparameters/config
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.seed = seed
        self.config = config or {}
        
        # Create unique log file for this experiment + seed
        self.log_file = self.log_dir / f"{experiment_name}_seed{seed}.json"
        
        # Initialize log structure
        self.data = {
            "metadata": {
                "experiment_name": experiment_name,
                "seed": seed,
                "config": self.config,
            },
            "episodes": [],  # Episode-level data
            "training": [],  # Training step data
            "evaluation": [],  # Evaluation data
        }
        
        self.episode_buffer = {}  # Temporary buffer for episode data
        
    def log_episode_start(self, episode_num: int, step: int):
        """Log the start of an episode."""
        self.episode_buffer = {
            "episode": episode_num,
            "step_start": step,
            "step_end": None,
            "reward": 0.0,
            "length": 0,
            "success": None,  # For task-specific success
            "metrics": {},  # Task-specific metrics
        }
    
    def log_episode_step(self, reward: float, length: int):
        """Update episode statistics during episode."""
        self.episode_buffer["reward"] = reward
        self.episode_buffer["length"] = length
    
    def log_episode_end(self, step: int, success: Optional[bool] = None, metrics: Optional[Dict] = None):
        """Log the end of an episode and save to buffer."""
        self.episode_buffer["step_end"] = step
        if success is not None:
            self.episode_buffer["success"] = success
        if metrics is not None:
            self.episode_buffer["metrics"].update(metrics)
        
        self.data["episodes"].append(self.episode_buffer)
    
    def log_training_step(
        self,
        step: int,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
        buffer_size: Optional[int] = None,
        lr: Optional[float] = None
    ):
        """Log training step metrics."""
        train_data = {
            "step": step,
            "loss": loss,
            "epsilon": epsilon,
            "buffer_size": buffer_size,
            "learning_rate": lr,
        }
        self.data["training"].append(train_data)
    
    def log_evaluation(
        self,
        step: int,
        eval_reward: float,
        eval_episodes: int = 10,
        success_rate: Optional[float] = None,
        metrics: Optional[Dict] = None
    ):
        """Log evaluation results."""
        eval_data = {
            "step": step,
            "eval_reward": eval_reward,
            "eval_episodes": eval_episodes,
            "success_rate": success_rate,
        }
        if metrics:
            eval_data.update(metrics)
        
        self.data["evaluation"].append(eval_data)
    
    def save(self):
        """Save logs to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Logs saved to {self.log_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from current logs."""
        if not self.data["episodes"]:
            return {}
        
        episode_rewards = [ep["reward"] for ep in self.data["episodes"]]
        episode_lengths = [ep["length"] for ep in self.data["episodes"]]
        episode_successes = [ep["success"] for ep in self.data["episodes"] if ep["success"] is not None]
        
        summary = {
            "total_episodes": len(self.data["episodes"]),
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "total_steps": self.data["episodes"][-1]["step_end"] if self.data["episodes"] else 0,
        }
        
        if episode_successes:
            success_rate = np.mean(episode_successes)
            summary["success_rate"] = float(success_rate)
        
        if self.data["evaluation"]:
            eval_rewards = [e["eval_reward"] for e in self.data["evaluation"]]
            summary["final_eval_reward"] = float(eval_rewards[-1])
        
        return summary


class MultiSeedAggregator:
    """Aggregates results across multiple seeds for comparison."""
    
    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
    
    def load_seed_results(self, experiment_name: str, seeds: List[int]) -> Dict[int, Dict]:
        """Load results for multiple seeds."""
        results = {}
        
        for seed in seeds:
            log_file = self.log_dir / f"{experiment_name}_seed{seed}.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    results[seed] = json.load(f)
            else:
                print(f"Warning: Log file not found for {experiment_name} seed {seed}")
        
        return results
    
    def aggregate_episode_rewards(self, results: Dict[int, Dict]) -> Dict[str, Any]:
        """Aggregate episode rewards across seeds."""
        all_rewards = []
        
        for seed, data in results.items():
            rewards = [ep["reward"] for ep in data.get("episodes", [])]
            all_rewards.append(rewards)
        
        if not all_rewards:
            return {}
        
        # Ensure all have same length (pad with NaN if needed)
        max_len = max(len(r) for r in all_rewards)
        padded = []
        for rewards in all_rewards:
            padded.append(rewards + [np.nan] * (max_len - len(rewards)))
        
        padded_array = np.array(padded)
        
        return {
            "mean_per_episode": np.nanmean(padded_array, axis=0).tolist(),
            "std_per_episode": np.nanstd(padded_array, axis=0).tolist(),
            "final_mean": float(np.nanmean(padded_array[:, -1])),
            "final_std": float(np.nanstd(padded_array[:, -1])),
            "all_seeds": {str(seed): rewards for seed, rewards in zip(results.keys(), all_rewards)}
        }
    
    def save_aggregate_summary(
        self,
        experiment_name: str,
        results: Dict[int, Dict],
        output_file: Optional[str] = None
    ):
        """Save aggregated results across seeds."""
        if output_file is None:
            output_file = self.log_dir / f"{experiment_name}_aggregate.json"
        
        # Aggregate different metrics
        episode_rewards = self.aggregate_episode_rewards(results)
        
        # Final metrics per seed
        final_metrics = {}
        for seed, data in results.items():
            episodes = data.get("episodes", [])
            if episodes:
                final_reward = episodes[-1]["reward"]
                avg_reward = np.mean([ep["reward"] for ep in episodes])
                success_rate = None
                
                successes = [ep["success"] for ep in episodes if ep["success"] is not None]
                if successes:
                    success_rate = float(np.mean(successes))
                
                final_metrics[str(seed)] = {
                    "final_reward": float(final_reward),
                    "avg_reward": float(avg_reward),
                    "success_rate": success_rate,
                    "num_episodes": len(episodes),
                }
        
        summary = {
            "experiment": experiment_name,
            "seeds": list(results.keys()),
            "episode_rewards": episode_rewards,
            "final_metrics": final_metrics,
        }
        
        # Overall summary
        if final_metrics:
            avg_rewards = [m["avg_reward"] for m in final_metrics.values()]
            summary["overall"] = {
                "mean_avg_reward": float(np.mean(avg_rewards)),
                "std_avg_reward": float(np.std(avg_rewards)),
            }
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregate results saved to {output_file}")
        
        return summary
