import torch
import torch.nn as nn
from utils import get_device
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, Homomorphism
from ff_transformer import S1FFTransformer
from config import Config

LINE_LEN = 100
LINE_KEY = 10

device = get_device()

def step(x, steps, dt):
    # x: [bs, line_len, 2] 
    # du/dt = -v * e^-(du/dx^2 + dv/dx^2)
    # dv/dt = +u * e^-(dv/dx^2 + dv/dx^2)
    for _ in range(steps):
        u = x[..., 0]
        v = x[..., 1]
        dudx = torch.roll(u, 1, dims=-1) - u
        dvdx = torch.roll(v, 1, dims=-1) - v

        mag = torch.exp(-(dudx * dudx + dvdx * dvdx))
        dudt = mag * -v
        dvdt = mag * u
    
        x = torch.stack((u + dudt * dt / steps, v + dvdt * dt / steps), dim=-1)
    return x

class HeatPredictor(nn.Module, Predictor):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * LINE_LEN, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * LINE_LEN),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def run(self, x):
        return self.model(torch.flatten(x, start_dim=-2)).reshape(x.shape)
    def forward(self, x):
        return self.run(x)

    def needs_training(self):
        return True

class HeatHomomorphism(nn.Module, Homomorphism):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6 * LINE_LEN, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * LINE_LEN),
        )

    def forward(self, x):
        return self.model(torch.flatten(x, start_dim=-3)).reshape(x.shape)

    def apply_y(self, x, y):
        return self.model(torch.cat((torch.flatten(x, start_dim=-3), torch.flatten(y, start_dim=-2)), dim=-1)).reshape(y.shape)


class HeatDataset(torch.utils.data.Dataset):
    def __init__(self, N): 
        self.N = N
        lerp = S1FFTransformer(LINE_LEN, 5)

        key = torch.normal(0, 2, (N, 5, 2), device=device)
        self.tensor = lerp.smooth_function(key)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.tensor[index], step(self.tensor[index], 1, 0.1)


if __name__ == '__main__':
    config = Config()

    predictor = HeatPredictor()
    transformer = S1FFTransformer(LINE_LEN, LINE_KEY)
    homomorphism = HeatHomomorphism()
    basis = GroupBasis(2, transformer, homomorphism, 1, config.standard_basis, coeff_epsilon=1, lr=1e-3)
    dataset = HeatDataset(config.N)

    gdn = LocalTrainer(predictor, basis, dataset, config)   
    gdn.train()

