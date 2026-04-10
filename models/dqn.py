import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network from "Human-level control through deep reinforcement learning"
    (Mnih et al., 2015)
    
    Architecture:
    - Conv1: 32 filters, 8x8 kernel, stride 4
    - Conv2: 64 filters, 4x4 kernel, stride 2
    - Conv3: 64 filters, 3x3 kernel, stride 1
    - FC1: 512 units
    - FC2: n_actions units (output layer)
    """
    
    def __init__(self, n_actions, input_shape=(4, 84, 84)):
        """
        Args:
            n_actions: Number of possible actions
            input_shape: Shape of input (channels, height, width)
        """
        super(DQN, self).__init__()
        
        self.n_actions = n_actions
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after conv layers
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def _get_conv_out(self, shape):
        """
        Calculate the output size of convolutional layers.
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)
            
        Returns:
            Q-values for each action, shape (batch_size, n_actions)
        """
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def act(self, state, epsilon=0.0):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Single state (4, 84, 84) or batch
            epsilon: Exploration rate
            
        Returns:
            action: Selected action (int)
        """
        if np.random.rand() < epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.n_actions)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                # Add batch dimension if needed
                if state.ndim == 3:
                    state = torch.FloatTensor(state).unsqueeze(0)
                else:
                    state = torch.FloatTensor(state)
                
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
                return action


import numpy as np