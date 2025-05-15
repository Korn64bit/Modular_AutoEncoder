import torch
from torch import nn

class AutoEncoder_V_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28,
                      out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128,
                      out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64,
                      out_features=36),
            nn.ReLU(),

            nn.Linear(in_features=36,
                      out_features=18),
            nn.ReLU(),

            nn.Linear(in_features=18,
                      out_features=9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=9,
                      out_features=18),
            nn.ReLU(),

            nn.Linear(in_features=18,
                      out_features=36),
            nn.ReLU(),

            nn.Linear(in_features=36,
                      out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64,
                      out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128,
                      out_features=28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x