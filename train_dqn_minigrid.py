import random
from collections import deque

import gymnasium as gym
import minigrid 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.minigrid_dqn import MiniGridDQN
from utils.replay_buffer import ReplayBuffer

def preprocess_obs(obs):
    # Convert the observation to a tensor and permute dimensions to (C, H, W)
    image = obs["image"] # (H, W, C)
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return image.astype(np.float32)  # Normalize to [0, 1]

def update_model(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("MiniGrid-MemoryS7-v0")

    n_actions = env.action_space.n
    input_shape = (3, 7, 7)  # C, H, W

    policy_net = MiniGridDQN(n_actions=n_actions, input_shape=input_shape).to(device)
    target_net = MiniGridDQN(n_actions=n_actions, input_shape=input_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000)

    num_episodes = 100
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    target_update_freq = 10

    epsilon = epsilon_start
    episode_rewards = []

    print(f"Training on device: {device}")
    print(f"Number of actions: {n_actions}")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = preprocess_obs(obs)

        done = False
        total_reward = 0.0
        loss_value = None
        step_count = 0

        while not done:
            action = policy_net.act(state, epsilon=epsilon, device=device)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_obs(next_obs)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            loss_value = update_model(
                policy_net = policy_net, 
                target_net = target_net,
                replay_buffer = replay_buffer,
                optimizer = optimizer,
                batch_size = batch_size,
                gamma = gamma,
                device = device
            )

            episode_rewards.append(total_reward)

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {total_reward:.3f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Steps: {step_count} | "
              f"Loss: {loss_value if loss_value is not None else 'N/A'}"
            )
        
    env.close()

if __name__ == "__main__":
    main()