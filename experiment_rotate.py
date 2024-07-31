import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
import math
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

from itertools import permutations
from utils import get_device, rmse 
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import TorusFFTransformer
from config import Config

device = get_device()

_lerp = None
def lerp():
    global _lerp
    if _lerp is None:
        _lerp = TorusFFTransformer(20, 20, 4, 4)  
    return _lerp

def perform_flow(features, flow, steps=1, delta=1):
    # features: [bs, channels, r, c]
    inverse = -flow * delta / steps
    r = features.shape[2]
    c = features.shape[3]

    index_map = torch.zeros_like(flow)
    index_map[torch.arange(r), :, 0] = torch.arange(r, dtype=flow.dtype).view(r, 1).to(device)
    index_map[:, torch.arange(c), 1] = torch.arange(c, dtype=flow.dtype).view(1, c).to(device)
    index_map += inverse

    u_map = torch.zeros_like(index_map).long()
    u_map[..., 0] = 1
    v_map = torch.zeros_like(index_map).long()
    v_map[..., 1] = 1

    base_00 = torch.floor(index_map).long()
    base_01 = base_00 + u_map
    base_10 = base_00 + v_map
    base_11 = base_00 + u_map + v_map

    st = index_map - torch.floor(index_map)
    s = st[..., 0].detach()
    t = st[..., 1].detach()

    def step(curr):
        x_padded = F.pad(curr, (1, 1, 1, 1), mode='constant', value=0) 
        def factor(tensor, mul):
            clipped_x = tensor[..., 0].clip(0, r)
            clipped_y = tensor[..., 1].clip(0, c)
            return x_padded[..., clipped_x + 1, clipped_y + 1] * mul
       
        return factor(base_00, (1 - s) * (1 - t)) + factor(base_01, s * (1 - t)) + factor(base_10, s * (1 - t)) + factor(base_11, s * t)

    for _ in range(steps):
        features = step(features)
    return features

class Flow(nn.Module):
    def __init__(self, input_features, num_flows, output_features, bad=False):
        super().__init__()

        self.bad = bad

        self.input_features = input_features
        self.num_flows = num_flows
        self.output_features = output_features

        # self.main_flow = nn.Parameter(torch.empty(num_flows, 100, 2))
        # torch.nn.init.normal_(self.main_flow, 0, 0.02)
        self.main_flow = torch.empty((num_flows, 16, 2), dtype=torch.double, device=device)
        for i in range(4):
            for j in range(4):
                dx = i - 1.5
                dy = j - 1.5
                self.main_flow[:, 4 * i + j] = torch.tensor(
                    [
                        [-dx, -dy],
                        [-dx, 0],
                        [-dx, dy],

                        [0, -dy],
                        [0, 0],
                        [0, dy],
            
                        [dx, -dy],
                        [dx, 0],
                        [dx, dy],
                    ],
                dtype=torch.double, device=device)
        """
        self.main_flow = torch.tensor([
              [[-1, -1]],
              [[-1, 0]],
              [[-1, 1]],

              [[0, -1]],
              [[0, 0]],
              [[0, 1]],
 
              [[1, -1]],
              [[1, 0]],
              [[1, 1]],
        ], device=device).expand((num_flows, 16, 2)).double()
        """

        self.combinator = nn.Parameter(torch.empty(output_features, input_features * num_flows, device=device).double())
        self.bias = nn.Parameter(torch.empty(output_features, device=device).double())

        # same as conv initialization
        n = input_features * num_flows
        stdv = 1. / math.sqrt(n)
        torch.nn.init.kaiming_uniform_(self.combinator, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.combinator)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.c = nn.Conv2d(self.input_features, self.output_features, 3, padding=1)
        self.c.weight = nn.Parameter(self.combinator.view(self.output_features, self.input_features, 3, 3))
        self.c.bias = self.bias
         
    def forward(self, x):
        flow = lerp().smooth_function(self.main_flow)

        ret = torch.empty((self.num_flows, x.shape[0], self.input_features, 20, 20)).double().to(x.device)
        x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0) 
        for i in range(self.num_flows):
            # dy, dx = self.main_flow[i][0]
            # ret[i] = x_padded[:, :, 1+dy:21+dy, 1+dx:21+dx]
            ret[i] = perform_flow(x, -flow[i])

        ret = ret.permute(1, 3, 4, 2, 0).reshape(x.shape[0], 20, 20, self.input_features * self.num_flows, 1)
        ret = (self.combinator @ ret).squeeze(-1) + self.bias
        ret = ret.permute(0, 3, 1, 2)

        comp = self.c(x)

        return ret
    
class CloudPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        i = 6
        f = 9
        self.model = nn.Sequential(
            Flow(3, f, i),
            nn.ReLU(),
            Flow(i, f, i),
            nn.ReLU(),
            Flow(i, f, 3),
            nn.ReLU(),
            Flow(3, f, 3),
        )

        """
        k = 3
        self.model = nn.Sequential(
            nn.Conv2d(3, i, k, padding='same'),
            nn.ReLU(),
            nn.Conv2d(i, i, k, padding='same'),
            nn.ReLU(),
            nn.Conv2d(i, 3, k, padding='same'),
            nn.ReLU(),
            nn.Conv2d(3, 3, k, padding='same'),
        )
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def forward(self, x):
        return self.model(x)

    def loss(self, xx, yy):
        return rmse(xx, yy)

    def needs_training(self):
        return True

class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        super().__init__()

        self.x = torch.randn((N, 3, 20, 20), device=device).double()
        self.y = 1 - self.x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    dataset = CloudDataset(1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    model = CloudPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 100
    for e in range(epochs):
        total_loss = 0.0
        for xx, yy in tqdm.tqdm(dataloader):
            y_pred = model(xx)
            loss = model.loss(y_pred, yy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / len(dataloader)
        print(f"Epoch {e+1}/{epochs}, Total Loss: {total_loss:.4f}")
