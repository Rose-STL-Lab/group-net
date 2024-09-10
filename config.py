import argparse
from utils import get_device


class Config:
    def __init__(self, n=20000, epochs=100, bs=16):
        parser = argparse.ArgumentParser()
        parser.add_argument('--standard_basis', default=False, action='store_true')
        parser.add_argument('--batch_size', type=int, default=bs, help='batch size')
        parser.add_argument('--epochs', type=int, default=epochs)
        parser.add_argument('--N', type=int, default=n,
                            help='for randomly generated datasets, the number of elements to generate')

        parser.add_argument('--reuse_predictor', default=False, action='store_true')
        parser.add_argument('--task', type=str)
        args = parser.parse_args()

        self.standard_basis = args.standard_basis
        self.batch_size = args.batch_size
        self.N = args.N
        self.epochs = args.epochs
        self.device = get_device()

        self.reuse_predictor = args.reuse_predictor
        self.task = args.task

