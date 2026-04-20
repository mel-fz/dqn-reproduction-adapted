"""
evaluate.py

Load a saved checkpoint and evaluate it.

Usage:
    # Evaluate DQN on Pong
    python evaluate.py --model checkpoints/dqn_pong_seed42_final.pth --env pong --episodes 30

    # Evaluate DRQN on MiniGrid
    python evaluate.py --model checkpoints/drqn_minigrid_seed42_final.pth --env minigrid --episodes 100
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch

from models.dqn import DQN
from models.drqn import DRQN
from utils.preprocessing import make_atari_env


def preprocess_minigrid_obs(obs):
    import numpy as np
    image = obs["image"]
    image = np.transpose(image, (2, 0, 1))
    return image.astype(np.float32)


def evaluate_pong(model_path, n_episodes=30, seed=42):
    """
    Evaluate a DQN checkpoint on Pong.
    Reports average reward and win rate over n_episodes.
    This matches the paper's evaluation protocol (30 episodes, epsilon=0.05).
    """
    import gymnasium as gym
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = make_atari_env('ALE/Pong-v5', seed=seed)
    n_actions = env.action_space.n

    model = DQN(n_actions).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from {model_path}")
    print(f"Checkpoint was at step {checkpoint['steps']}, episode {checkpoint['episodes']}")

    episode_rewards = []
    win_rates       = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done         = False
        ep_reward    = 0
        points_won   = 0
        points_lost  = 0

        while not done:
            with torch.no_grad():
                action = model.act(state, epsilon=0.05)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if reward > 0:
                points_won += 1
            elif reward < 0:
                points_lost += 1

        episode_rewards.append(ep_reward)
        total_points = points_won + points_lost
        win_rate = points_won / total_points if total_points > 0 else 0.0
        win_rates.append(win_rate)
        print(f"  Episode {ep+1:3d}: reward={ep_reward:6.1f}  win_rate={win_rate:.2f}")

    env.close()

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean win rate: {np.mean(win_rates):.2f} ± {np.std(win_rates):.2f}")
    return episode_rewards, win_rates


def evaluate_minigrid(model_path, n_episodes=100, seed=42):
    """
    Evaluate a DRQN checkpoint on MiniGrid-MemoryS7-v0.
    Reports average reward and success rate.
    Success = episode reward > 0 (agent picked the correct object).
    """
    import gymnasium as gym
    import minigrid  # noqa
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make("MiniGrid-MemoryS7-v0")
    env.reset(seed=seed)
    n_actions   = env.action_space.n
    input_shape = (3, 7, 7)

    model = DRQN(n_actions=n_actions, input_shape=input_shape).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from {model_path}")

    episode_rewards = []
    successes       = []

    for ep in range(n_episodes):
        obs, _  = env.reset()
        state   = preprocess_minigrid_obs(obs)
        hidden  = model.init_hidden(batch_size=1, device=str(device))
        done    = False
        ep_reward = 0

        while not done:
            with torch.no_grad():
                action, hidden = model.act(state, hidden, epsilon=0.05, device=str(device))
            obs, reward, terminated, truncated, _ = env.step(action)
            state     = preprocess_minigrid_obs(obs)
            done      = terminated or truncated
            ep_reward += reward

        episode_rewards.append(ep_reward)
        # Success = agent received positive reward (found the right object)
        successes.append(1 if ep_reward > 0 else 0)
        print(f"  Episode {ep+1:3d}: reward={ep_reward:.3f}  success={'YES' if ep_reward > 0 else 'no'}")

    env.close()

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward:    {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Success rate:   {np.mean(successes):.2f}")
    return episode_rewards, successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--env',      required=True, choices=['pong', 'minigrid'], help='Which environment')
    parser.add_argument('--episodes', type=int, default=30, help='Number of evaluation episodes')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    if args.env == 'pong':
        evaluate_pong(args.model, n_episodes=args.episodes, seed=args.seed)
    else:
        evaluate_minigrid(args.model, n_episodes=args.episodes, seed=args.seed)


if __name__ == '__main__':
    main()
