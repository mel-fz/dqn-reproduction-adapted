import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.dqn import DQN

print("Testing DQN Network...")

# Create network
n_actions = 6  # Pong has 6 actions
model = DQN(n_actions=n_actions)

print(f"\nModel architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass with single state
print("\n--- Testing single state ---")
state = np.random.rand(4, 84, 84).astype(np.float32)
state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

q_values = model(state_tensor)
print(f"Input shape: {state_tensor.shape}")
print(f"Output Q-values shape: {q_values.shape}")
print(f"Q-values: {q_values.detach().numpy()}")

# Test forward pass with batch
print("\n--- Testing batch ---")
batch_size = 32
batch_states = np.random.rand(batch_size, 4, 84, 84).astype(np.float32)
batch_tensor = torch.FloatTensor(batch_states)

q_values_batch = model(batch_tensor)
print(f"Batch input shape: {batch_tensor.shape}")
print(f"Batch output shape: {q_values_batch.shape}")

# Test action selection
print("\n--- Testing action selection ---")
state = np.random.rand(4, 84, 84).astype(np.float32)

# With epsilon=0 (greedy)
action_greedy = model.act(state, epsilon=0.0)
print(f"Greedy action: {action_greedy}")

# With epsilon=1 (random)
actions_random = [model.act(state, epsilon=1.0) for _ in range(10)]
print(f"Random actions (epsilon=1): {actions_random}")

# With epsilon=0.1 (mostly greedy)
actions_mixed = [model.act(state, epsilon=0.1) for _ in range(20)]
print(f"Mixed actions (epsilon=0.1): {actions_mixed}")
print(f"  Unique actions: {set(actions_mixed)}")

print("\nDQN network test successful!")