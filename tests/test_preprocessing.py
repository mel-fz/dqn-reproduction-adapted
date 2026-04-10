import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import make_atari_env

# Create preprocessed environment
env = make_atari_env('ALE/Pong-v5', seed=42)

# Reset
obs, info = env.reset()
print(f"Preprocessed observation shape: {obs.shape}")  # Should be (4, 84, 84)
print(f"Observation dtype: {obs.dtype}")  # Should be float32
print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")  # Should be [0, 1]

# Visualize the 4 stacked frames
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(obs[i], cmap='gray')
    axes[i].set_title(f'Frame {i+1}')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('stacked_frames.png', dpi=150, bbox_inches='tight')
print("Saved stacked frames to stacked_frames.png")

# Take a few steps
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: Shape={obs.shape}, Reward={reward}")
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Episode ended, reset environment")

env.close()
print("\nPreprocessing test successful!")