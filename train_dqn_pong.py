import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.dqn import DQN
from models.replay_buffer import ReplayBuffer
from utils.preprocessing import make_atari_env


class DQNAgent:
    def __init__(
        self,
        env,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1_000_000,
        replay_buffer_size=100_000,
        batch_size=32,
        target_update_freq=10_000,
        learning_starts=10_000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.n_actions = env.action_space.n
        self.device = device

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

        self.policy_net = DQN(self.n_actions).to(device)
        self.target_net = DQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate,
            alpha=0.95,
            eps=0.01
        )

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.steps = 0
        self.episodes = 0

    def select_action(self, state):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.steps / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end)
        )
        return self.policy_net.act(state, self.epsilon)

    def update(self):
        if len(self.replay_buffer) < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values   = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames=500_000, eval_freq=50_000, save_freq=250_000, seed=42):
        episode_rewards = []
        win_rates       = []
        losses          = []
        eval_rewards    = []

        state, _ = self.env.reset()
        episode_reward = 0
        points_won     = 0
        points_lost    = 0

        pbar = tqdm(total=num_frames, desc=f"Seed={seed}")

        while self.steps < num_frames:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            clipped_reward = np.clip(reward, -1, 1)

            if reward > 0:
                points_won += 1
            elif reward < 0:
                points_lost += 1

            self.replay_buffer.push(state, action, clipped_reward, next_state, done)

            state           = next_state
            episode_reward += reward
            self.steps     += 1

            loss = self.update()
            if loss is not None:
                losses.append(loss)

            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if done:
                episode_rewards.append(episode_reward)
                total_points = points_won + points_lost
                win_rate = points_won / total_points if total_points > 0 else 0.0
                win_rates.append(win_rate)
                self.episodes += 1

                pbar.set_postfix({
                    'Ep':  self.episodes,
                    'R':   f'{episode_reward:.0f}',
                    'WR':  f'{win_rate:.2f}',
                    'eps': f'{self.epsilon:.3f}',
                })

                state          = self.env.reset()[0]
                episode_reward = 0
                points_won     = 0
                points_lost    = 0

            if self.steps % eval_freq == 0:
                eval_reward = self.evaluate()
                eval_rewards.append((self.steps, eval_reward))
                print(f"\nEval @ {self.steps}: {eval_reward:.2f}")

            if self.steps % save_freq == 0:
                os.makedirs('checkpoints', exist_ok=True)
                self.save_checkpoint(f'checkpoints/dqn_pong_seed{seed}_{self.steps}.pth')

            pbar.update(1)

        pbar.close()
        os.makedirs('checkpoints', exist_ok=True)
        self.save_checkpoint(f'checkpoints/dqn_pong_seed{seed}_final.pth')

        return episode_rewards, win_rates, losses, eval_rewards

    def evaluate(self, n_episodes=5):
        self.policy_net.eval()
        total_reward = 0
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                with torch.no_grad():
                    action = self.policy_net.act(state, epsilon=0.05)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            total_reward += ep_reward
        self.policy_net.train()
        return total_reward / n_episodes

    def save_checkpoint(self, path):
        torch.save({
            'steps':                 self.steps,
            'episodes':              self.episodes,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict':  self.optimizer.state_dict(),
            'epsilon':               self.epsilon,
        }, path)
        print(f"Saved: {path}")


def run_seed(seed, num_frames=500_000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env   = make_atari_env('ALE/Pong-v5', seed=seed)
    agent = DQNAgent(env=env)

    print(f"\n{'='*50}")
    print(f"Training DQN on Pong — Seed {seed}")
    print(f"Device: {agent.device} | Actions: {agent.n_actions}")
    print(f"{'='*50}")

    results = agent.train(num_frames=num_frames, seed=seed)
    env.close()
    return results


def plot_multiseed_results(all_results, seeds):
    os.makedirs('results', exist_ok=True)
    window = 20

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    reward_smoothed_all  = []
    winrate_smoothed_all = []

    for seed, (ep_rewards, win_rates, _, _) in zip(seeds, all_results):
        if len(ep_rewards) >= window:
            sr = np.convolve(ep_rewards, np.ones(window) / window, mode='valid')
            sw = np.convolve(win_rates,  np.ones(window) / window, mode='valid')
        else:
            sr = np.array(ep_rewards)
            sw = np.array(win_rates)

        reward_smoothed_all.append(sr)
        winrate_smoothed_all.append(sw)
        axes[0].plot(sr, alpha=0.4, label=f'Seed {seed}')
        axes[1].plot(sw, alpha=0.4, label=f'Seed {seed}')

    min_r  = min(len(s) for s in reward_smoothed_all)
    min_w  = min(len(s) for s in winrate_smoothed_all)
    mean_r = np.mean([s[:min_r] for s in reward_smoothed_all], axis=0)
    mean_w = np.mean([s[:min_w] for s in winrate_smoothed_all], axis=0)

    axes[0].plot(mean_r, color='black', linewidth=2, label='Mean')
    axes[1].plot(mean_w, color='black', linewidth=2, label='Mean')

    axes[0].set_title(f'DQN Pong — Episode Reward ({window}-ep avg)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()

    axes[1].set_title(f'DQN Pong — Win Rate ({window}-ep avg)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Rate')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/pong_multiseed_results.png', dpi=150)
    print("Plot saved to results/pong_multiseed_results.png")


def main():
    seeds       = [42, 123, 7]
    num_frames  = 500_000
    all_results = []

    for seed in seeds:
        results = run_seed(seed, num_frames=num_frames)
        all_results.append(results)

        os.makedirs('results', exist_ok=True)
        ep_rewards, win_rates, losses, eval_rewards = results
        np.save(f'results/pong_seed{seed}_rewards.npy',  np.array(ep_rewards))
        np.save(f'results/pong_seed{seed}_winrates.npy', np.array(win_rates))
        np.save(f'results/pong_seed{seed}_eval.npy',     np.array(eval_rewards))
        print(f"Seed {seed} done.")

    plot_multiseed_results(all_results, seeds)


if __name__ == '__main__':
    main()
