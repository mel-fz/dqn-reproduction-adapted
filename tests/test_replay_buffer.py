import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.replay_buffer import ReplayBuffer

print("Testing Replay Buffer...")

# Create buffer
buffer = ReplayBuffer(capacity=100)
print(f"Initial buffer size: {len(buffer)}")

# Add some fake transitions
for i in range(50):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = np.random.randint(0, 6)
    reward = np.random.randn()
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done = np.random.rand() > 0.9
    
    buffer.push(state, action, reward, next_state, done)

print(f"Buffer size after 50 pushes: {len(buffer)}")

# Sample a mini-batch
batch_size = 32
states, actions, rewards, next_states, dones = buffer.sample(batch_size)

print(f"\nSampled batch:")
print(f"  States shape: {states.shape}")  # Should be (32, 4, 84, 84)
print(f"  Actions shape: {actions.shape}")  # Should be (32,)
print(f"  Rewards shape: {rewards.shape}")  # Should be (32,)
print(f"  Next states shape: {next_states.shape}")  # Should be (32, 4, 84, 84)
print(f"  Dones shape: {dones.shape}")  # Should be (32,)

print(f"\nData types:")
print(f"  States dtype: {states.dtype}")
print(f"  Actions dtype: {actions.dtype}")
print(f"  Rewards dtype: {rewards.dtype}")
print(f"  Next states dtype: {next_states.dtype}")
print(f"  Dones dtype: {dones.dtype}")

# Test buffer overflow (should cap at 100)
for i in range(60):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    buffer.push(state, 0, 0.0, state, False)

print(f"\nBuffer size after 110 total pushes (capacity=100): {len(buffer)}")
print("Replay buffer test successful!")