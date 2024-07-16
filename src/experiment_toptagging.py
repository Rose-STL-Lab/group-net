# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import get_device 
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis
from ff_transformer import SingletonFFTransformer
from config import Config

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def __call__(self, x):
        return self.run(x)

    def run(self, x):
        ret = self.model(x.reshape(-1, self.n_dim * self.n_components).to(device))
        return ret

    def loss(self, xx, yy):
        return nn.functional.cross_entropy(xx, yy)

    def needs_training(self):
        return True

class TopTagging(torch.utils.data.Dataset):
    def __init__(self, path='./data/top-tagging/train.h5', flatten=False, n_component=2, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4*n_component]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.len = self.X.shape[0]
        
        self.y = torch.LongTensor(df[:, -1]).to(device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    epochs = 125
    bs = 64

    n_dim = 4
    n_component = 30
    n_class = 2

    config = Config()

    # predictor = torch.load('models/toptagclass.pt')
    predictor = ClassPredictor(n_dim, n_component, n_class)

    transformer = SingletonFFTransformer((n_component, ))
    basis = GroupBasis(4, transformer, 7, config.standard_basis, loss_type='cross_entropy')

    dataset = TopTagging(n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    gdn = LocalTrainer(predictor, basis)
    gdn.train(loader, epochs)

