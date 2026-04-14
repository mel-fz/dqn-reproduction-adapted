import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network (DRQN) for MiniGrid MemoryEnv.

    Same conv layers as MiniGridDQN, but instead of going straight
    to a fully connected layer, the output is fed into an LSTM.
    This gives the agent memory across timesteps — exactly what
    MemoryEnv requires.

    Architecture:
        Conv -> Conv -> LSTM -> FC -> Q-values
    """

    def __init__(self, n_actions, input_shape=(3, 7, 7), lstm_hidden_size=128):
        super().__init__()

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lstm_hidden_size = lstm_hidden_size

        # Convolutional layers (same as MiniGridDQN)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Figure out the conv output size
        conv_out_size = self._get_conv_out(input_shape)

        # LSTM — this is the memory. It takes the flattened conv output
        # and remembers information across timesteps within an episode.
        self.lstm = nn.LSTM(
            input_size=conv_out_size,
            hidden_size=lstm_hidden_size,
            batch_first=True  # expects (batch, seq_len, features)
        )

        # Output layer: maps LSTM hidden state -> Q-value per action
        self.fc = nn.Linear(lstm_hidden_size, n_actions)

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()))

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor. Can be:
               - (batch, seq_len, C, H, W) for sequence training
               - (batch, C, H, W) for single-step inference
            hidden: Tuple of (h, c) LSTM hidden states. Pass None to
                    start a fresh episode.

        Returns:
            q_values: Shape (batch, seq_len, n_actions) or (batch, n_actions)
            hidden:   Updated LSTM hidden state (h, c) — save this between
                      steps during an episode
        """
        # Handle both single-frame and sequence inputs
        if x.dim() == 4:
            # Single frame: (batch, C, H, W) -> add seq_len=1
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, C, H, W = x.shape

        # Run conv layers over every frame in the sequence
        x = x.view(batch_size * seq_len, C, H, W)
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, conv_out)

        # Run LSTM over the sequence
        x, hidden = self.lstm(x, hidden)  # x: (batch, seq_len, lstm_hidden)

        # Map to Q-values
        q_values = self.fc(x)  # (batch, seq_len, n_actions)

        if squeeze_output:
            q_values = q_values.squeeze(1)  # (batch, n_actions)

        return q_values, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        """
        Create fresh (zeroed) hidden state for the start of a new episode.
        Call this at the beginning of every episode.
        """
        h = torch.zeros(1, batch_size, self.lstm_hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size).to(device)
        return (h, c)

    def act(self, state, hidden, epsilon=0.0, device='cpu'):
        """
        Select an action using epsilon-greedy, while tracking hidden state.

        Args:
            state:   Single frame as numpy array (C, H, W)
            hidden:  Current LSTM hidden state (h, c)
            epsilon: Exploration rate
            device:  torch device string

        Returns:
            action: int
            hidden: Updated hidden state — MUST be passed back in next call
        """
        if np.random.rand() < epsilon:
            # Random action — still need to update hidden state
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                _, hidden = self.forward(state_t, hidden)
            return np.random.randint(0, self.n_actions), hidden
        else:
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values, hidden = self.forward(state_t, hidden)
                action = q_values.argmax(dim=1).item()
            return action, hidden
