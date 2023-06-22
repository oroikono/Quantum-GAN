import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_modes):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modes)
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, hidden_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_modes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)
