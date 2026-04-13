import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGridDQN(nn.Module):
    def __init__(self, n_actions, input_shape=(3, 7, 7)):
        super().__init__()

        self.n_actions = n_actions
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0, device="cpu"):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.n_actions)

        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()

            if state.ndim == 3:
                state = state.unsqueeze(0)

            state = state.to(device)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()