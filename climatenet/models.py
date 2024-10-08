###########################################################################
#CGNet: A Light-weight Context Guided Network for Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1811.08201.pdf
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from climatenet.modules import *
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import xarray as xr
from climatenet.utils.utils import Config
from os import path
import pathlib

class CGNet():
    '''
    The high-level CGNet class. 
    This allows training and running CGNet without interacting with PyTorch code.
    If you are looking for a higher degree of control over the training and inference,
    we suggest you directly use the CGNetModule class, which is a PyTorch nn.Module.

    Parameters
    ----------
    config : Config
        The model configuration.
    model_path : str
        Path to load the model and config from.

    Attributes
    ----------
    config : dict
        Stores the model config
    network : CGNetModule
        Stores the actual model (nn.Module)
    optimizer : torch.optim.Optimizer
        Stores the optimizer we use for training the model
    '''

    def __init__(self, equivariant, device, config: Config = None, model_path: str = None):
        self.device = device
        self.equivariant = equivariant
    
        if config is not None and model_path is not None:
            raise ValueError('''Config and weight path set at the same time. 
            Pass a config if you want to create a new model, 
            and a weight_path if you want to load an existing model.''')

        if config is not None:
            # Create new model
            self.config = config
            self.network = CGNetModule(equivariant, classes=len(self.config.labels), channels=len(list(self.config.fields))).to(device)
        elif model_path is not None:
            # Load model
            self.config = Config(path.join(model_path, 'config.json'))
            self.network = CGNetModule(equivariant, classes=len(self.config.labels), channels=len(list(self.config.fields))).to(device)
            self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))
        else:
            raise ValueError('''You need to specify either a config or a model path.''')

        self.optimizer = Adam(self.network.parameters(), lr=self.config.lr)        
        
    def train(self, dataset: ClimateDatasetLabeled):
        '''Train the network on the given dataset for the given amount of epochs'''
        self.network.train()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.train_batch_size, collate_fn=collate, num_workers=4, shuffle=True)
        for epoch in range(1, self.config.epochs+1):

            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(loader)
            aggregate_cm = np.zeros((3,3))

            for features, labels in epoch_loader:
                timestamps = features.coords['time'].values.tolist()

                # Push data on GPU and pass forward
                features = torch.tensor(features.values).to(self.device)
                labels = torch.tensor(labels.values).to(self.device)
                
                outputs = torch.softmax(self.network(features), 1)

                # Update training CM
                predictions = torch.max(outputs, 1)[1]

                aggregate_cm += get_cm(predictions, labels, 3)

                # Pass backward
                loss = jaccard_loss(outputs, labels)
                epoch_loader.set_description(f'Loss: {loss.item()}')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

            print('Epoch stats:')
            print(aggregate_cm)
            ious = get_iou_perClass(aggregate_cm)
            print('IOUs: ', ious, ', mean: ', ious.mean())

    def predict(self, dataset: ClimateDataset, save_dir: str = None):
        '''Make predictions for the given dataset and return them as xr.DataArray'''
        self.network.eval()
        collate = ClimateDataset.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate)
        epoch_loader = tqdm(loader)

        predictions = []
        for batch in epoch_loader:
            features = torch.tensor(batch.values).to(self.device)
        
            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            preds = torch.max(outputs, 1)[1].cpu().numpy()

            coords = batch.coords
            del coords['variable']
            
            dims = [dim for dim in batch.dims if dim != "variable"]
            
            predictions.append(xr.DataArray(preds, coords=coords, dims=dims, attrs=batch.attrs))

        return xr.concat(predictions, dim='time')

    def evaluate(self, dataset: ClimateDatasetLabeled):
        '''Evaluate on a dataset and return statistics'''
        self.network.eval()
        collate = ClimateDatasetLabeled.collate
        loader = DataLoader(dataset, batch_size=self.config.pred_batch_size, collate_fn=collate, num_workers=4)

        epoch_loader = tqdm(loader)
        aggregate_cm = np.zeros((3,3))

        for features, labels in epoch_loader:
            timestamps = features.coords['time'].values.tolist()
            features = torch.tensor(features.values).to(self.device)
            labels = torch.tensor(labels.values).to(self.device)
                
            with torch.no_grad():
                outputs = torch.softmax(self.network(features), 1)
            predictions = torch.max(outputs, 1)[1]
            
            aggregate_cm += get_cm(predictions, labels, 3)

        print('Evaluation stats:')
        print(aggregate_cm)
        ious = get_iou_perClass(aggregate_cm)
        print('IOUs: ', ious, ', mean: ', ious.mean())

    def save_model(self, save_path: str):
        '''
        Save model weights and config to a directory.
        '''
        # create save_path if it doesn't exist
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 

        # save weights and config
        self.config.save(path.join(save_path, 'config.json'))
        torch.save(self.network.state_dict(), path.join(save_path, 'weights.pth'))

    def load_model(self, model_path: str):
        '''
        Load a model. While this can easily be done using the normal constructor, this might make the code more readable - 
        we instantly see that we're loading a model, and don't have to look at the arguments of the constructor first.
        '''
        self.config = Config(path.join(model_path, 'config.json'))
        self.network = CGNetModule(self.equivariant, classes=len(self.config.labels), channels=len(list(self.config.fields))).to(self.device)
        self.network.load_state_dict(torch.load(path.join(model_path, 'weights.pth')))


class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """
    def __init__(self, equivariant, classes=19, channels=4, M=3, N= 21, dropout_flag = False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(equivariant, channels, 32, 3, 2)      # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(equivariant, 32, 32, 3, 1)                          
        self.level1_2 = ConvBNPReLU(equivariant, 32, 32, 3, 1)      

        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + channels)
        
        #stage 2
        self.level2_0 = ContextGuidedBlock_Down(equivariant, 32 + channels, 64, dilation_rate=2,reduction=8)  
        self.level2 = nn.ModuleList()
        for i in range(0, M-1):
            self.level2.append(ContextGuidedBlock(equivariant, 64 , 64, dilation_rate=2, reduction=8))  #CG block
        self.bn_prelu_2 = BNPReLU(128 + channels)
        
        #stage 3
        self.level3_0 = ContextGuidedBlock_Down(equivariant, 128 + channels, 128, dilation_rate=4, reduction=16) 
        self.level3 = nn.ModuleList()
        for i in range(0, N-1):
            self.level3.append(ContextGuidedBlock(equivariant, 128 , 128, dilation_rate=4, reduction=16)) # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False),Conv(equivariant, 256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(equivariant, 256, classes, 1, 1))

        #init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        
        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1,  output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return out
      
   
