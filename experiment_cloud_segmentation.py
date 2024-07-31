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

device = get_device(no_mps=False)

_lerp = None
def lerp():
    global _lerp
    if _lerp is None:
        _lerp = TorusFFTransformer(450, 480, 2, 2)  
    return _lerp

def perform_flow(features, flow, steps=1, delta=1):
    # features: [bs, channels, r, c]
    inverse = flow * delta / steps
    r = features.shape[2]
    c = features.shape[3]

    index_map = torch.zeros_like(flow)
    index_map[torch.arange(r), :, 0] = torch.arange(r, dtype=torch.float32).view(r, 1).to(device)
    index_map[:, torch.arange(c), 1] = torch.arange(c, dtype=torch.float32).view(1, c).to(device)
    index_map += inverse

    u_map = torch.zeros_like(index_map)
    u_map[..., 0] = 1
    v_map = torch.zeros_like(index_map)
    v_map[..., 1] = 1


    base_00 = torch.round(index_map).long()
    base_01 = base_00 + u_map
    base_10 = base_00 + v_map
    base_11 = base_00 + u_map + v_map

    st = index_map - torch.floor(index_map) 
    s = st[..., 0]
    t = st[..., 1]

    def step(curr):
        def factor(tensor, mul):
            clipped_x = tensor[..., 0].clip(0, r - 1).long()
            clipped_y = tensor[..., 1].clip(0, c - 1).long()
            return curr[..., clipped_x, clipped_y] * mul
       
        return factor(base_00, 1)
        # return factor(base_00, s * t) + factor(base_01, s * (1 - t)) + factor(base_10, (1 - s) * t) + factor(base_11, (1 - s) * (1 - t))

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
        ], device=device).expand((num_flows, 4, 2))

        self.combinator = nn.Parameter(torch.empty(output_features, input_features * num_flows, device=device))
        self.bias = nn.Parameter(torch.empty(output_features, device=device))

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
        # flow = lerp().smooth_function(self.main_flow)
        # ret = perform_flow(x, flow)

        # Instead of using torch.roll, let's create a padded version of x
        x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0) 

        ret = torch.empty((self.num_flows, x.shape[0], self.input_features, 450, 480)).to(x.device)
        for i in range(self.num_flows):
            dy, dx = self.main_flow[i][0]
            ret[i] = x_padded[:, :, 1+dy:451+dy, 1+dx:481+dx]
            # ret[i] = perform_flow(x, flow[i])

        ret = ret.permute(1, 3, 4, 0, 2).reshape(x.shape[0], 450, 480, self.input_features * self.num_flows, 1)
        ret = (self.combinator @ ret).squeeze(-1) + self.bias
        ret = ret.permute(0, 3, 1, 2)
        comp = self.c(x)

        return ret

        if self.training and not self.bad:
            comp = self.c(x)
            return comp
        else:
            flow = lerp().smooth_function(self.main_flow)

            x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0) 

            ret = perform_flow(x, flow)
            """
            ret = torch.empty((self.num_flows, x.shape[0], self.input_features, 450, 480)).to(x.device)
            for i in range(self.num_flows):
                dy, dx = self.main_flow[i][0]
                ret[i] = 
                ret[i] = x_padded[:, :, 1+dy:451+dy, 1+dx:481+dx]
            """

            ret = ret.permute(1, 3, 4, 0, 2).reshape(x.shape[0], 450, 480, self.input_features * self.num_flows, 1)
            ret = (self.combinator @ ret).squeeze(-1) + self.bias
            ret = ret.permute(0, 3, 1, 2)

            return ret
    
class CloudPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        i = 6
        f = 9
        self.model = nn.Sequential(
            Flow(3, f, i),
            nn.BatchNorm2d(i),
            nn.ReLU(),
            Flow(i, f, i),
            nn.BatchNorm2d(i),
            nn.ReLU(),
            Flow(i, f, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            Flow(3, f, 3),
            nn.Sigmoid(),
        )

        """
        k = 3
        self.model = nn.Sequential(
            nn.Conv2d(3, i, k, padding='same'),
            nn.BatchNorm2d(i),
            nn.ReLU(),
            nn.Conv2d(i, i, k, padding='same'),
            nn.BatchNorm2d(i),
            nn.ReLU(),
            nn.Conv2d(i, 3, k, padding='same'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, k, padding='same'),
            nn.Sigmoid(),
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
    def __init__(self, image_dir='data/WSISEG-Database/whole sky images/',
                 annotation_dir='data/WSISEG-Database/annotation/',
                 l=1,h=340):
        super().__init__()

        self.image_paths = []
        self.annotation_paths = []
        self.l = l
        self.h = h
        for i in range(l, h + 1):
            self.image_paths.append(image_dir + f'ASC100-1006_{i:03}')
            self.annotation_paths.append(annotation_dir + f'ASC100-1006_{i:03}')

        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        global flow
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(img_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('L')  

        to_tensor = torchvision.transforms.functional.pil_to_tensor

        image = to_tensor(image).to(device)/ 255
        annotation = to_tensor(annotation).to(device).reshape(450, 480)
        annotation_int = torch.zeros((3, 450, 480), dtype=torch.float32).to(device)
        annotation_int[0] = (annotation==0).float()
        annotation_int[1] = (annotation==100).float()
        annotation_int[2] = (annotation==255).float()

        return image, annotation_int


if __name__ == '__main__':
    dataset = CloudDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    model = CloudPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    flow = nn.Parameter(torch.empty((100, 2))).to(device)
    torch.nn.init.normal_(flow, 0, 2)
    optimizer2 = torch.optim.Adam(model.parameters())

    epochs = 10
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

    """
        total_loss = 0.0
        for xx, yy in dataloader:
            flowed_x = perform_flow(xx, flow)
            flowed_y = perform_flow(yy, flow)
            y_pred = model(flowed_x)
            loss = rmse(y_pred, flowed_y)

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

            total_loss += loss.item() / len(dataloader)
        print(f"Epoch {e+1}/{epochs}, Total Flow Loss: {total_loss:.4f}")

    model = CloudPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    dataset = CloudDataset(l=-339,h=340)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for e in range(epochs // 2):
        total_loss = 0.0
        for xx, yy in dataloader:
            y_pred = model(xx)
            loss = model.loss(y_pred, yy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / len(dataloader)
        print(f"Epoch {e+1}/{epochs}, Total Loss: {total_loss:.4f}")

    numel = 60
    dataset = CloudDataset(l=341,h=400)
    acc = 0
    model.eval()
    for i in range(numel):
        x,y = dataset[i]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y_pred = model(x)
        _, mx = torch.max(y_pred, dim=1)
        _, rmx = torch.max(y, dim=1)

        non_background = rmx != 0
        acc += torch.sum(((mx == rmx) & non_background).float()) / torch.sum(non_background) / numel
    print("Accuracy with Diffeomorphism", acc)

    # create new dataset
    model = CloudPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    dataset = CloudDataset(l=1,h=340)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for e in range(epochs):
        total_loss = 0.0
        for xx, yy in dataloader:
            y_pred = model(xx)
            loss = model.loss(y_pred, yy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / len(dataloader)
        print(f"Epoch {e+1}/{epochs}, Total Loss: {total_loss:.4f}")
    """
    numel = 60
    dataset = CloudDataset(l=341,h=400)
    acc = 0
    model.eval()
    for i in range(numel):
        x,y = dataset[i]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y_pred = model(x)
        _, mx = torch.max(y_pred, dim=1)
        _, rmx = torch.max(y, dim=1)

        non_background = rmx != 0
        acc += torch.sum(((mx == rmx) & non_background).float()) / torch.sum(non_background) / numel
    print("Accuracy without Diffeomorphism", acc)
