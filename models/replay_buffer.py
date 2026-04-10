import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores transitions and samples random mini-batches for training.
    """
    def __init__(self, capacity=1_000_000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (4, 84, 84)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (4, 84, 84)
            done: Whether episode ended (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of batched (states, actions, rewards, next_states, dones)
        """
        # Sample random indices
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)