import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_dqn_pong import DQNAgent
from utils.preprocessing import make_atari_env

print("Testing training loop (short run)...")

# Create environment
env = make_atari_env('ALE/Pong-v5', seed=42)

# Create agent with smaller buffer for testing
agent = DQNAgent(
    env=env,
    replay_buffer_size=10_000,
    learning_starts=1_000,
    batch_size=32
)

print(f"Device: {agent.device}")

# Run for just 5000 steps to test
print("\nRunning short training test (5000 steps)...")
agent.train(num_frames=5000, eval_freq=2500, save_freq=5000)

print("\nTraining loop test successful!")