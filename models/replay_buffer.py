import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Stores frames as uint8 (0-255) instead of float32 to save 8x memory.
    Converts back to float32 on sample.
    """

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Convert float32 [0,1] -> uint8 [0,255] to save RAM
        self.buffer.append((
            (state * 255).astype(np.uint8),
            action,
            reward,
            (next_state * 255).astype(np.uint8),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32) / 255.0,
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32) / 255.0,
            np.array(dones,       dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)
