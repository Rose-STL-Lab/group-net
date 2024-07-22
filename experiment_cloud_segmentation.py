import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

from utils import get_device, rmse 
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import TorusFFTransformer
from config import Config

device = get_device(no_mps=False)

lerp = TorusFFTransformer(450, 480, 10, 10) 
def perform_flow(features, flow, steps=1, delta=1):
    # features: [bs, channels, r, c]
    inverse = -flow * delta / steps
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
       
        return factor(base_00, s * t) + factor(base_01, s * (1 - t)) + factor(base_10, (1 - s) * t) + factor(base_11, (1 - s) * (1 - t))

    for _ in range(steps):
        features = step(features)
    return features

class Flow(nn.Module):
    def __init__(self, input_features, num_flows, output_features):
        super().__init__()
        self.input_features = input_features
        self.num_flows = num_flows
        self.output_features = output_features

        self.main_flow = nn.Parameter(torch.empty(num_flows, 100, 2))
        torch.nn.init.normal_(self.main_flow, 0, 0.02)
        self.combinator = nn.Parameter(torch.empty(output_features, input_features * num_flows))
        torch.nn.init.normal_(self.combinator, 0, 0.02)
         
    def forward(self, x):
        flow = lerp.smooth_function(self.main_flow)
        ret = torch.empty((self.num_flows, x.shape[0], self.input_features, 450, 480)).to(device)
        for i in range(self.num_flows):
            ret[i] = perform_flow(x, flow[i])

        ret = ret.swapaxes(0, 1).reshape(x.shape[0], self.input_features * self.num_flows, 450, 480)
        ret = ret.permute(0, 2, 3, 1).reshape(x.shape[0], 450, 480, self.input_features * self.num_flows, 1)
        ret = self.combinator @ ret
        return ret.squeeze(-1).permute(0, 3, 1, 2)
    
class CloudPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            Flow(3, 4, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            Flow(3, 4, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            Flow(3, 4, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            Flow(3, 4, 3),
            nn.Sigmoid(),
        )
        self.model = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding='same'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding='same'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding='same'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding='same'),
            nn.Sigmoid(),
        )
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
        org_idx = idx
        if self.l + idx <= 0:
            idx = 1 - (self.l + idx) - self.l

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

        if False and self.l + org_idx <= 0:
            org_image = image

            image = image.unsqueeze(0)
            annotation_int = flow.unsqueeze(0)

            image = perform_flow(image, flow)
            annotation_int = perform_flow(image, flow)

            image = image.squeeze(0)
            annotation_int = annotation_int.squeeze(0)

            """
            def tensor_to_image(tensor):
                # Convert tensor to numpy array
                np_array = tensor.cpu().numpy()
                
                # Normalize the numpy array
                np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min())
                
                # Convert to uint8 format
                np_array = (np_array * 255).astype(np.uint8)
                
                # Convert to PIL image
                if np_array.shape[0] == 1:
                    np_array = np_array.squeeze(0)  # If single channel, remove channel dimension
                    image = Image.fromarray(np_array, mode='L')  # 'L' for (8-bit pixels, black and white)
                else:
                    np_array = np_array.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                    image = Image.fromarray(np_array)
                
                return image

            image = tensor_to_image(org_image)
            annotation_image = tensor_to_image(image2)
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.title("Image")
            plt.imshow(image)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Annotation Image")
            plt.imshow(annotation_image)
            plt.axis('off')

            plt.show()
            """

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
        for xx, yy in dataloader:
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
