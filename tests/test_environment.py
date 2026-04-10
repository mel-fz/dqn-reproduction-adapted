import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

# Register ALE environments
gym.register_envs(ale_py)

print("Testing Atari Pong environment...")
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')

# Reset and get initial observation
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")
print(f"Observation dtype: {obs.dtype}")
print(f"Observation range: [{obs.min()}, {obs.max()}]")
print(f"Action space: {env.action_space}")
print(f"Number of actions: {env.action_space.n}")

# Visualize gameplay
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: Action={action}, Reward={reward}")

# Visualize a frame
plt.figure(figsize=(8, 10))
plt.imshow(obs)
plt.title(f"Raw Atari Pong Frame\nShape: {obs.shape}")
plt.axis('off')
plt.tight_layout()
plt.savefig('pong_frame.png', dpi=150, bbox_inches='tight')
print("\nSaved frame to pong_frame.png")

env.close()