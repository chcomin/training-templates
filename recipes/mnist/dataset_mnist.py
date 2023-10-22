import random
from torchvision.datasets import MNIST
import numpy as np
import torch
from torch.utils.data import Dataset

class MNISTV2(Dataset):
    '''Create Dataset with similar attributes as MNIST. This class is necessary for splitting
    the dataset into train and validation.'''
   
    def __init__(self, data, targets, return_as_color=False):
        super().__init__()

        self.data = data
        self.targets = targets
        self.return_as_color = return_as_color

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])
        img = img.float()/255
        img = img[None]
        if self.return_as_color:
            img = img.repeat(3, 1, 1)

        return img, target
    
def download(directory):

    MNIST(directory, train=True, download=True)

def create_datasets(root, train_val_split, download=False, seed=None):

    ds = MNIST(root, train=True, download=download)
    data, targets = ds.data, ds.targets

    # Split into train and validation
    n_valid = round(train_val_split*data.shape[0])
    random.seed(seed)
    indices = list(range(data.shape[0]))
    random.shuffle(indices)
    data_train, targets_train = data[indices[n_valid:]], targets[indices[n_valid:]]
    data_valid, targets_valid = data[indices[:n_valid]], targets[indices[:n_valid]]

    ds_train = MNISTV2(data_train, targets_train)
    ds_valid = MNISTV2(data_valid, targets_valid)

    return ds_train, ds_valid