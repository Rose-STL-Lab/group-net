# Modified from LieGAN Codebase (https://github.com/Rose-STL-Lab/LieGAN/blob/master)
import sys
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import time
import pandas as pd
import numpy as np
<<<<<<< HEAD
from utils import get_device, sum_reduce
from local_symmetry import Predictor, LocalTrainer
from group_basis import GroupBasis, TrivialHomomorphism
from ff_transformer import SingletonFFTransformer
=======
import tqdm
from genetic import Genetic
from utils import get_device
from local_symmetry import Predictor
>>>>>>> gauge_discovery
from config import Config
from top import dataset, gnn


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

<<<<<<< HEAD
device = get_device()
dtype = torch.float32
    
=======
device = get_device(no_mps=True)


>>>>>>> gauge_discovery
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

    def name(self):
        return "toptag"

    def run(self, x):
<<<<<<< HEAD
        # unsqueeze is bc homomorphism assumes at least one manifold dimension
        return self.model(x.reshape(-1, self.n_dim * self.n_components)).unsqueeze(-1)
=======
        ret = self.model(x.flatten(-2))
        return ret
>>>>>>> gauge_discovery

    def loss(self, y_pred, y_true):
        return nn.functional.cross_entropy(y_pred.squeeze(-1), y_true.squeeze(-1))

    def needs_training(self):
        return True


class TopTagging(torch.utils.data.Dataset):
    def __init__(self, N, path='./data/top-tagging/train.h5', flatten=False, n_component=2, noise=0.0):
        super().__init__()
        df = pd.read_hdf(path, key='table')
        df = df.to_numpy()
        self.X = df[:, :4 * n_component]
        self.X = torch.FloatTensor(self.X).to(device)
        if not flatten:
            self.X = self.X.reshape(-1, n_component, 4)
        self.len = self.X.shape[0]
<<<<<<< HEAD
        
        self.y = torch.LongTensor(df[:, -1]).unsqueeze(-1).to(device)
=======

        self.y = torch.LongTensor(df[:, -1]).to(device)
        self.N = N
>>>>>>> gauge_discovery

    def __len__(self):
        return min(self.N, self.len)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run(e, loader, partition):
    tik = time.time()
    loader_length = len(loader)

    res = {'time':0, 'correct':0, 'loss': 0, 'counter': 0, 'acc': 0,
           'loss_arr':[], 'correct_arr':[],'label':[],'score':[]}

    for i, data in enumerate(dataloaders['train']):
        batch_size, n_nodes, _ = data['Pmu'].size()

        if partition == "predictor":
            atom_positions = data['Pmu'].reshape(batch_size * n_nodes, -1).to(device, dtype)
        else:
            xx = data['Pmu'].to(device, dtype)
            yy = data['is_signal'].reshape(-1, 1).to(device, dtype).long()

            xp, yp = basis.apply(xx, yy)
            yp = yp.view(-1).to(device)

            atom_positions = xp.reshape(batch_size * n_nodes, -1).to(device, dtype)

        atom_mask = data['atom_mask'].reshape(batch_size * n_nodes, -1).to(device)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(device)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(device,dtype)
        nodes = gnn.psi(nodes)
        edges = [a.to(device) for a in data['edges']]
        label = data['is_signal'].to(device, dtype).long()
        
        pred = ddp_model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask,
                        edge_mask=edge_mask, n_nodes=n_nodes)
    
        predict = pred.max(1).indices
        
        if partition == "predictor":
            correct = torch.sum(predict == label).item()
            loss = nn.functional.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            correct = torch.sum(predict == yp).item()
            loss = basis.loss(pred, yp) 

            reg = basis.regularization(e)
            loss += reg

            basis.optimizer.zero_grad()
            loss.backward()
            basis.optimizer.step()

        res['time'] = time.time() - tik
        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['correct_arr'].append(correct)

        running_loss = sum(res['loss_arr'][-100:])/len(res['loss_arr'][-100:])
        running_acc = sum(res['correct_arr'][-100:])/(len(res['correct_arr'][-100:])*batch_size)
        avg_time = res['time']/res['counter'] * batch_size
        tmp_counter = sum_reduce(res['counter'], device = device)
        tmp_loss = sum_reduce(res['loss'], device = device) / tmp_counter
        tmp_acc = sum_reduce(res['correct'], device = device) / tmp_counter
        
        if i % config.log_interval == 0:
            if partition == "predictor":
                print(">> Predictor: \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                    (e + 1, config.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))
            else:
                print(">> Basis: \t Epoch %d/%d \t Batch %d/%d \t Loss %.4f \t Running Acc %.3f \t Total Acc %.3f \t Avg Batch Time %.4f" %
                    (e + 1, config.epochs, i, loader_length, running_loss, running_acc, tmp_acc, avg_time))

    res['counter'] = sum_reduce(res['counter'], device = device).item()
    res['loss'] = sum_reduce(res['loss'], device = device).item() / res['counter']
    res['acc'] = sum_reduce(res['correct'], device = device).item() / res['counter']
    print("Time: train: %.2f \t Train loss %.4f \t Train acc: %.4f" % (res['time'],res['loss'],res['acc']))

    if partition == "predictor":
        torch.save(ddp_model.state_dict(), f"./models/predictor/checkpoint-epoch-{e}.pt")
    
    return 

def train():
    for e in range(0, config.epochs):
        #train predictor
        run(e, dataloaders['train'], "predictor")

        #train basis
        run(e, dataloaders['train'], "basis")

        print("Discovered Basis \n", basis.summary())

if __name__ == '__main__':
    torch.set_printoptions(precision=9, sci_mode=False)

    n_dim = 4
    n_class = 2
    ortho_factor = 0.1
    growth_factor = 1

    config = Config()

<<<<<<< HEAD
    dist.init_process_group(backend='nccl') 
    train_sampler, dataloaders = dataset.retrieve_dataloaders(
                                    config.batch_size,
                                    1,
                                    num_train=-1,
                                    datadir="./data/top-tagging-converted",
                                    nobj = config.n_component)

    predictor = gnn.LorentzNet(n_scalar = 1, n_hidden = config.n_hidden, n_class = n_class,
                       dropout = config.dropout, n_layers = config.n_layers,
                       c_weight = config.c_weight)
    predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
    predictor = predictor.to(device)
    ddp_model = DistributedDataParallel(predictor, device_ids=[0])

    ## load best model if needed
    best_model = torch.load(f"./models/predictor/checkpoint-epoch-0.pt")
    ddp_model.load_state_dict(best_model)

    ## predictor optimizer
    optimizer = optim.AdamW(ddp_model.parameters(), lr=0.0003, weight_decay=config.weight_decay)
    
    transformer = SingletonFFTransformer((config.n_component, ))
    homomorphism = TrivialHomomorphism([1], 1)
    basis = GroupBasis(4, transformer, homomorphism, 7, config.standard_basis, loss_type='cross_entropy')
=======
    dataset = TopTagging(config.N, n_component=n_component)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if config.reuse_predictor:
        predictor = torch.load('predictors/toptag.pt')
    else:
        print("Disabled right now")
        exit(1)

        print("Training Predictor (once finished, run with --reuse_predictor to train with predictor)")
        predictor = ClassPredictor(n_dim, n_component, n_class)

        for e in range(config.epochs):
            p_losses = []
            for xx, yy in tqdm.tqdm(loader):
                y_pred = predictor.run(xx)
                y_true = yy

                p_loss = predictor.loss(y_pred, y_true)
                p_losses.append(float(p_loss.detach().cpu()))

                predictor.optimizer.zero_grad()
                p_loss.backward()
                predictor.optimizer.step()

            p_losses = np.mean(p_losses) if len(p_losses) else 0
            torch.save(predictor, "predictors/" + predictor.name() + '.pt')
            print("Epoch", e, "Predictor loss", p_losses)

        print("Finished training predictor. Rerun using --reuse_predictor to discover symmetry")
        sys.exit(0)

    # discover infinitesimal generators via gradient descent
    if not config.skip_continuous:
        lie = torch.nn.Parameter(torch.empty(7, 4, 4))
        torch.nn.init.normal_(lie, 0, 0.02)
        optimizer = torch.optim.Adam([lie])

        for e in range(config.epochs):
            average_loss = []
            for xx, yy in tqdm.tqdm(loader):
                coeff = torch.normal(0, 1, (xx.shape[0], lie.shape[0])).to(device) 
                sampled_lie = torch.sum(lie * coeff.unsqueeze(-1).unsqueeze(-1), dim=-3)

                g = torch.matrix_exp(sampled_lie)
                g_x = torch.einsum('bij, bcj -> bci', g, xx)

                # b, 2
                y_pred = predictor.run(g_x)
                # with respect to yy or predictor.run(xx) aren't that different
                y_tind = yy

                loss = torch.nn.functional.cross_entropy(y_pred, y_tind)

                # reg
                trace = torch.einsum('kdf,kdf->k', lie, lie)
                mag = torch.sqrt(trace / lie.shape[1])
                norm = lie / mag.unsqueeze(-1).unsqueeze(-1)

                if config.standard_basis:
                    norm = torch.abs(norm)
                reg = growth_factor * -torch.mean(mag) + ortho_factor * torch.sum(torch.abs(torch.triu(torch.einsum('bij,cij->bc', norm, norm), diagonal=1)))

                loss = loss + reg

                average_loss.append(loss.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            average_loss = np.mean(average_loss)
            print("Loss", average_loss, "Basis", lie.detach())


    # discover discrete generators 

    # basis of orthogonal lie algebra to so+(1, 3)
    basis_vectors = torch.tensor([
        [[0,0,0,1.],
         [0,0,0,0],
         [0,0,0,0],
         [-1,0,0,0]],

        [[0,0,1,0],
         [0,0,0,0],
         [-1,0,0,0],
         [0,0,0,0]],

        [[0,1,0,0],
         [-1,0,0,0],
         [0,0,0,0],
         [0,0,0,0]],

        [[0,0,0,0],
         [0,0,1,0],
         [0,1,0,0],
         [0,0,0,0]],

        [[0,0,0,0],
         [0,0,0,1],
         [0,0,0,0],
         [0,1,0,0]],

        [[0,0,0,0],
         [0,0,0,0],
         [0,0,0,1],
         [0,0,1,0]],

        [[1,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]],

        [[0,0,0,0],
         [0,1,0,0],
         [0,0,0,0],
         [0,0,0,0]],

        [[0,0,0,0],
         [0,0,0,0],
         [0,0,1,0],
         [0,0,0,0]],

        [[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,1]],
    ])

    matrices = torch.nn.Parameter(torch.zeros(400, 4, 4))
    torch.nn.init.normal_(matrices, 0, 1)
    optimizer = torch.optim.Adam([matrices])

    for e in range(40):
        average_losses = []
        for xx, yy in tqdm.tqdm(loader):
            det = torch.abs(torch.det(matrices).unsqueeze(-1).unsqueeze(-1))
            normalized = matrices / (det ** 0.25)
            g_x = torch.einsum('pij, bcj -> pbci', normalized, xx)
            x = xx.unsqueeze(0).expand(g_x.shape)

            # p b 2
            y_pred = predictor.run(g_x)
            # p b
            y_true = predictor.run(x)
            y_tind = torch.nn.functional.softmax(y_true, dim=-1)

            y_pred = torch.permute(y_pred, (1, 2, 0))
            y_tind = torch.permute(y_tind, (1, 2, 0))

            losses = torch.mean(torch.nn.functional.cross_entropy(y_pred, y_tind, reduction='none'), dim=0)
            loss = torch.mean(losses)
            average_losses.append(losses.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        min_loss = np.min(np.mean(average_losses, axis=0))
        min_index = np.argmin(np.mean(average_losses, axis=0))
        det = torch.abs(torch.det(matrices).unsqueeze(-1).unsqueeze(-1))
        normalized = matrices / (det ** 0.25)
        print("Loss", min_loss, normalized[min_index].detach())
>>>>>>> gauge_discovery

    train()