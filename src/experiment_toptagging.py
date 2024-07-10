# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import get_device, affine_coord
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import SingletonFFTransformer

device = get_device()
    
class ClassPredictor(Predictor):
    def __init__(self, n_dim, n_components, n_classes):
        super().__init__()
        self.n_dim = n_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Linear(n_dim * n_components, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
    
    def __call__(self, x):
        return self.model(x.reshape(-1, self.n_dim * self.n_components).to(device))

    def run(self, x):
        return self.model(x.reshape(-1, self.n_dim * self.n_components).to(device))

    def needs_training(self):
        return True

class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/train.h5', flatten=False, n_component=3, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = self.X * np.random.uniform(1-noise, 1+noise, size=self.X.shape)
        self.y = df[:, -1]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.y = torch.LongTensor(self.y).to(device)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    epochs = 25
    N = 1000
    bs = 64

    n_dim = 4
    n_component = 2
    n_class = 2
    # n_channel = 7
    # d_input_size = n_dim * n_component
    # emb_size = 32

    # predictor from LieGAN codebase
    predictor = ClassPredictor(n_dim, n_component, n_class)

    transformer = SingletonFFTransformer()
    basis = GroupBasis(4, transformer, 6, 3)
    basis.predictor = predictor

    # dataset from LieGAN codebase
    dataset = TopTagging(n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)
    gdn.train(loader, epochs)