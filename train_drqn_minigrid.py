"""
train_drqn_minigrid.py

Trains a Deep Recurrent Q-Network (DRQN) on MiniGrid-MemoryS7-v0.

The key difference from train_dqn_minigrid.py:
  - We store full EPISODES in the replay buffer (not individual transitions)
  - We sample random sequences of length `seq_len` from those episodes
  - The LSTM hidden state is reset at the start of each episode and carried
    forward step-by-step during training

This is necessary because the LSTM needs to see transitions IN ORDER to
learn temporal dependencies. Randomly shuffled single transitions (like
standard DQN replay) would break the sequence and the LSTM couldn't learn.
"""

import os
# Fix OpenMP conflict between numpy, scipy, and torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
from collections import deque
import argparse

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.drqn import DRQN
from utils.logging import ExperimentLogger, MultiSeedAggregator


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_obs(obs):
    """Convert MiniGrid dict observation to (C, H, W) float32 array."""
    image = obs["image"]                        # (H, W, C)
    image = np.transpose(image, (2, 0, 1))      # -> (C, H, W)
    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# Episode Replay Buffer
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
    """
    Stores complete episodes and samples fixed-length sequences from them.

    Why episodes instead of transitions?
    Because the LSTM needs to process frames IN ORDER to build up a hidden
    state. If we sampled random single transitions (standard DQN style),
    the sequence would be meaningless and the LSTM couldn't learn memory.
    """

    def __init__(self, capacity=500):
        """
        Args:
            capacity: Maximum number of EPISODES to store (not transitions).
                      Old episodes are dropped when full.
        """
        self.buffer = deque(maxlen=capacity)
        self.current_episode = []  # transitions being collected this episode

    def start_episode(self):
        """Call at the beginning of each episode to start collecting."""
        self.current_episode = []

    def push(self, state, action, reward, next_state, done):
        """Add a single transition to the current episode."""
        self.current_episode.append((state, action, reward, next_state, done))

    def end_episode(self):
        """Call at the end of each episode to save it to the buffer."""
        if len(self.current_episode) > 0:
            self.buffer.append(list(self.current_episode))
        self.current_episode = []

    def sample(self, batch_size, seq_len):
        """
        Sample a batch of random sequences.

        For each item in the batch, we pick a random episode and a random
        starting point within it, then grab seq_len consecutive transitions.
        If the episode is shorter than seq_len, we pad with zeros.

        Args:
            batch_size: Number of sequences to return
            seq_len:    Length of each sequence

        Returns:
            states:      (batch_size, seq_len, C, H, W)
            actions:     (batch_size, seq_len)
            rewards:     (batch_size, seq_len)
            next_states: (batch_size, seq_len, C, H, W)
            dones:       (batch_size, seq_len)
        """
        # Filter to episodes long enough to sample from
        valid_episodes = [ep for ep in self.buffer if len(ep) >= 1]
        if len(valid_episodes) < batch_size:
            return None

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        sampled_episodes = random.choices(valid_episodes, k=batch_size)

        for episode in sampled_episodes:
            ep_len = len(episode)

            # Pick a random start; if episode shorter than seq_len, start at 0
            if ep_len >= seq_len:
                start = random.randint(0, ep_len - seq_len)
                seq = episode[start: start + seq_len]
            else:
                seq = episode  # use whole episode, will pad below

            states, actions, rewards, next_states, dones = zip(*seq)

            states      = np.array(states,      dtype=np.float32)
            actions     = np.array(actions,     dtype=np.int64)
            rewards     = np.array(rewards,     dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones       = np.array(dones,       dtype=np.float32)

            # Pad with zeros if episode was shorter than seq_len
            actual_len = len(seq)
            if actual_len < seq_len:
                pad = seq_len - actual_len
                C, H, W = states.shape[1], states.shape[2], states.shape[3]
                states      = np.concatenate([states,      np.zeros((pad, C, H, W), dtype=np.float32)],  axis=0)
                next_states = np.concatenate([next_states, np.zeros((pad, C, H, W), dtype=np.float32)],  axis=0)
                actions     = np.concatenate([actions,     np.zeros(pad,            dtype=np.int64)],     axis=0)
                rewards     = np.concatenate([rewards,     np.zeros(pad,            dtype=np.float32)],   axis=0)
                dones       = np.concatenate([dones,       np.ones(pad,             dtype=np.float32)],   axis=0)  # padded steps = "done"

            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)

        return (
            np.array(batch_states),       # (B, T, C, H, W)
            np.array(batch_actions),      # (B, T)
            np.array(batch_rewards),      # (B, T)
            np.array(batch_next_states),  # (B, T, C, H, W)
            np.array(batch_dones),        # (B, T)
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Training update
# ---------------------------------------------------------------------------

def update_model(policy_net, target_net, replay_buffer, optimizer,
                 batch_size, seq_len, gamma, device):
    """
    Sample a batch of sequences and do one gradient update.

    The LSTM hidden state is reset to zero at the start of each sequence.
    This is the standard DRQN approach ("random episode sampling with
    zero-initialised hidden state").
    """
    result = replay_buffer.sample(batch_size, seq_len)
    if result is None:
        return None

    states, actions, rewards, next_states, dones = result

    # Move everything to tensors on the right device
    states      = torch.tensor(states,      dtype=torch.float32).to(device)  # (B, T, C, H, W)
    actions     = torch.tensor(actions,     dtype=torch.long).to(device)     # (B, T)
    rewards     = torch.tensor(rewards,     dtype=torch.float32).to(device)  # (B, T)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)  # (B, T, C, H, W)
    dones       = torch.tensor(dones,       dtype=torch.float32).to(device)  # (B, T)

    # Q-values from policy net for all timesteps
    # hidden=None means the LSTM starts fresh (zeroed) for each sequence
    q_values_all, _ = policy_net(states, hidden=None)         # (B, T, n_actions)
    q_values = q_values_all.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, T)

    # Target Q-values from target net
    with torch.no_grad():
        next_q_all, _ = target_net(next_states, hidden=None)  # (B, T, n_actions)
        next_q = next_q_all.max(dim=2)[0]                     # (B, T)
        target_q = rewards + gamma * next_q * (1 - dones)     # (B, T)

    loss = nn.SmoothL1Loss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(seed=42, num_episodes=500, lstm_hidden_size=128, seq_len=8,
          batch_size=32, gamma=0.99, lr=1e-3,
          epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
          target_update_freq=10, buffer_capacity=500):
    """
    Full DRQN training run on MiniGrid-MemoryS7-v0.

    Args:
        seed:              Random seed for reproducibility
        num_episodes:      How many episodes to train for
        lstm_hidden_size:  Size of the LSTM hidden state
        seq_len:           Length of sequences sampled for training
        batch_size:        Number of sequences per gradient update
        gamma:             Discount factor
        lr:                Learning rate
        epsilon_start:     Starting exploration rate
        epsilon_end:       Minimum exploration rate
        epsilon_decay:     Multiplicative decay per episode
        target_update_freq: How often (in episodes) to sync target net
        buffer_capacity:   Max episodes stored in replay buffer

    Returns:
        episode_rewards: List of total reward per episode
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed {seed} | Device: {device}")

    env = gym.make("MiniGrid-MemoryS7-v0")
    env.reset(seed=seed)

    n_actions   = env.action_space.n
    input_shape = (3, 7, 7)

    policy_net = DRQN(n_actions=n_actions, input_shape=input_shape,
                      lstm_hidden_size=lstm_hidden_size).to(device)
    target_net = DRQN(n_actions=n_actions, input_shape=input_shape,
                      lstm_hidden_size=lstm_hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer      = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer  = EpisodeReplayBuffer(capacity=buffer_capacity)

    epsilon        = epsilon_start
    episode_rewards = []
    
    # Setup logger
    config = {
        "learning_rate": lr,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "target_update_freq": target_update_freq,
        "lstm_hidden_size": lstm_hidden_size,
        "buffer_capacity": buffer_capacity,
        "environment": "MiniGrid-MemoryS7-v0",
        "model": "DRQN (recurrent)",
    }
    logger = ExperimentLogger(
        log_dir="results/logs",
        experiment_name="drqn_minigrid",
        seed=seed,
        config=config
    )
    
    total_steps = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state  = preprocess_obs(obs)
        
        # Log episode start
        logger.log_episode_start(episode + 1, total_steps)

        # Fresh hidden state at the start of each episode
        hidden = policy_net.init_hidden(batch_size=1, device=device)

        replay_buffer.start_episode()

        done         = False
        total_reward = 0.0
        loss_value   = None
        step_count   = 0

        while not done:
            # Select action — hidden state is updated inside act()
            action, hidden = policy_net.act(state, hidden, epsilon=epsilon, device=device)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_obs(next_obs)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state        = next_state
            total_reward += reward
            step_count   += 1
            total_steps  += 1

            # Train on a batch of sequences
            loss_value = update_model(
                policy_net    = policy_net,
                target_net    = target_net,
                replay_buffer = replay_buffer,
                optimizer     = optimizer,
                batch_size    = batch_size,
                seq_len       = seq_len,
                gamma         = gamma,
                device        = device,
            )

        replay_buffer.end_episode()
        episode_rewards.append(total_reward)
        
        # Log episode end
        logger.log_episode_step(total_reward, step_count)
        logger.log_episode_end(total_steps, success=(total_reward > 0), metrics={"steps": step_count, "loss": loss_value})

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Sync target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Logging
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {total_reward:.3f} | "
              f"Avg(10): {avg_reward:.3f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Steps: {step_count} | "
              f"Loss: {loss_value:.4f}" if loss_value else
              f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {total_reward:.3f} | "
              f"Avg(10): {avg_reward:.3f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Steps: {step_count} | Loss: N/A")

    # Save logger
    logger.save()
    env.close()
    return episode_rewards


# ---------------------------------------------------------------------------
# Multi-seed entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DRQN on MiniGrid MemoryEnv")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit to run all 3 seeds)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--lstm-size", type=int, default=128, help="LSTM hidden size")
    args = parser.parse_args()

    # Single-seed mode (when called with --seed argument)
    if args.seed is not None:
        print(f"\n{'='*50}")
        print(f"Training DRQN — Seed {args.seed}")
        print(f"{'='*50}")
        rewards = train(seed=args.seed, num_episodes=args.episodes, lstm_hidden_size=args.lstm_size)
        print(f"\nTraining complete for seed {args.seed}")
        return
    
    # Multi-seed mode (default: run 3 seeds and aggregate)
    seeds       = [42, 123, 456]   # 3 seeds as required by the assignment
    num_episodes = args.episodes
    all_rewards  = {}

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Training DRQN — Seed {seed}")
        print(f"{'='*50}")
        rewards = train(seed=seed, num_episodes=num_episodes, lstm_hidden_size=args.lstm_size)
        all_rewards[seed] = rewards

    # Aggregate and save results
    aggregator = MultiSeedAggregator()
    results = aggregator.load_seed_results("drqn_minigrid", seeds)
    if results:
        aggregator.save_aggregate_summary("drqn_minigrid", results)

    # ------------------------------------------------------------------
    # Plot: one curve per seed + a mean curve
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    smoothed_all = []
    window = 20

    for seed, rewards in all_rewards.items():
        # Smooth with a rolling average
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        smoothed_all.append(smoothed)
        plt.plot(smoothed, alpha=0.4, label=f"Seed {seed}")

    # Mean across seeds
    min_len  = min(len(s) for s in smoothed_all)
    mean_curve = np.mean([s[:min_len] for s in smoothed_all], axis=0)
    plt.plot(mean_curve, color='black', linewidth=2, label="Mean")

    plt.xlabel("Episode")
    plt.ylabel(f"Reward ({window}-ep avg)")
    plt.title("DRQN on MiniGrid-MemoryS7 (3 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/drqn_minigrid_results.png", dpi=150)
    print("\nPlot saved to results/drqn_minigrid_results.png")


if __name__ == "__main__":
    main()
