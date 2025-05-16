import torch
from torch import nn

# Model V_1
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
    
# Model V_2
class AutoEncoder_V_2(nn.Module):
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
                      out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,
                      out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,
                      out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8,
                      out_features=4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=4,
                      out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8,
                      out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,
                      out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,
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
    
# Model V_3
class AutoEncoder_V_3(nn.Module):
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
                      out_features=32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32,
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


class VariationalAutoEncoder_V_1(nn.Module):
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
                      out_features=32),
        )
        self.fc_mu = nn.Linear(in_features=32,
                               out_features=16)
        self.fc_log_var = nn.Linear(in_features=32,
                                    out_features=16)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=16,
                      out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,
                      out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=28*28),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        z = self.reparametrize(mu, log_var)

        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var