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
    """
    DQN Agent that handles training and action selection.
    """
    def __init__(
        self,
        env,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1_000_000,
        replay_buffer_size=1_000_000,
        batch_size=32,
        target_update_freq=10_000,
        learning_starts=50_000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.n_actions = env.action_space.n
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Networks
        self.policy_net = DQN(self.n_actions).to(device)
        self.target_net = DQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=learning_rate,
            alpha=0.95,
            eps=0.01
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Tracking
        self.steps = 0
        self.episodes = 0
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.steps / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end)
        )
        
        return self.policy_net.act(state, self.epsilon)
    
    def update(self):
        """Perform one step of training."""
        if len(self.replay_buffer) < self.learning_starts:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (Huber loss is more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_frames=10_000_000, eval_freq=100_000, save_freq=500_000):
        """
        Main training loop.
        
        Args:
            num_frames: Total number of frames to train for
            eval_freq: How often to evaluate (in frames)
            save_freq: How often to save checkpoint (in frames)
        """
        episode_rewards = []
        episode_lengths = []
        losses = []
        eval_rewards = []
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        pbar = tqdm(total=num_frames, desc="Training")
        
        while self.steps < num_frames:
            # Select and perform action
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.steps += 1
            
            # Train
            loss = self.update()
            if loss is not None:
                losses.append(loss)
            
            # Update target network
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Episode ended
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                self.episodes += 1
                
                pbar.set_postfix({
                    'Episode': self.episodes,
                    'Reward': f'{episode_reward:.1f}',
                    'Epsilon': f'{self.epsilon:.3f}',
                    'Buffer': len(self.replay_buffer)
                })
                
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Evaluate
            if self.steps % eval_freq == 0:
                eval_reward = self.evaluate(n_episodes=10)
                eval_rewards.append((self.steps, eval_reward))
                print(f"\nEvaluation at {self.steps} steps: {eval_reward:.2f}")
            
            # Save checkpoint
            if self.steps % save_freq == 0:
                self.save_checkpoint(f'checkpoints/dqn_pong_{self.steps}.pth')
            
            pbar.update(1)
        
        pbar.close()
        
        # Save final model
        self.save_checkpoint('checkpoints/dqn_pong_final.pth')
        
        # Plot results
        self.plot_results(episode_rewards, losses, eval_rewards)
        
        return episode_rewards, losses, eval_rewards
    
    def evaluate(self, n_episodes=10):
        """Evaluate the current policy."""
        self.policy_net.eval()
        total_reward = 0
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.policy_net.act(state, epsilon=0.05)  # Small epsilon for evaluation
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        self.policy_net.train()
        return total_reward / n_episodes
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'steps': self.steps,
            'episodes': self.episodes,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        print(f"Checkpoint loaded from {path}")
    
    def plot_results(self, episode_rewards, losses, eval_rewards):
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Smoothed rewards (moving average)
        if len(episode_rewards) > 100:
            smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(smoothed)
            axes[0, 1].set_title('Smoothed Episode Rewards (100-episode avg)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
        
        # Loss
        if losses:
            axes[1, 0].plot(losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Evaluation rewards
        if eval_rewards:
            steps, rewards = zip(*eval_rewards)
            axes[1, 1].plot(steps, rewards, marker='o')
            axes[1, 1].set_title('Evaluation Rewards')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig('results/training_results.png', dpi=150)
        print("Training plots saved to results/training_results.png")


def main():
    # Create environment
    env = make_atari_env('ALE/Pong-v5', seed=42)
    
    # Create agent
    agent = DQNAgent(
        env=env,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1_000_000,
        replay_buffer_size=1_000_000,
        batch_size=32,
        target_update_freq=10_000,
        learning_starts=50_000
    )
    
    print(f"Training on device: {agent.device}")
    print(f"Number of actions: {agent.n_actions}")
    
    # Train
    agent.train(
        num_frames=10_000_000,  # 10M frames (paper uses 50M)
        eval_freq=100_000,
        save_freq=500_000
    )


if __name__ == '__main__':
    main()