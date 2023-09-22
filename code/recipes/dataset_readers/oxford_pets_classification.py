import random
from functools import partial
from PIL import Image
import numpy as np
from torchvision.datasets import OxfordIIITPet
import torch
from torchvision.transforms import v2 as tv_transf
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class OxfordIIITPetV2(Dataset):
    '''Create Dataset with similar attributes as OxfordIIITPet. This class is necessary for splitting
    the dataset into train and validation. It also defines a .getitem function that returns the images
    without any transformation.'''

    _cats = [0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33]
    _dogs = [1, 2, 3, 4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 28, 29, 30, 31, 34, 35, 36]
   
    def __init__(self, images, labels, segs, classes, class_to_idx, transforms=None, target_type='category'):
        super().__init__()

        self._images = images
        self._labels = labels
        self._segs = segs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transforms = transforms
        self.target_type = target_type
        self.species = ['cat', 'dog']

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx, use_transforms=True):
        image = Image.open(self._images[idx]).convert("RGB")
        label = self._labels[idx]
        if label in self._cats:
            label = 0
        elif label in self._dogs:
            label = 1

        if self.transforms and use_transforms:
            image = self.transforms(image)

        return image, label

    def getitem(self, idx):
        return self.__getitem__(idx, use_transforms=False)
    
class ClassificationPresetTrain:
    def __init__(self, crop_size, mean, std,
    ):
        
        self.transforms = tv_transf.Compose([
            tv_transf.PILToTensor(),
            tv_transf.RandomResizedCrop(crop_size, antialias=True),
            tv_transf.RandomHorizontalFlip(),
            tv_transf.ConvertDtype(torch.float),
            tv_transf.Normalize(mean=mean, std=std)])

    def __call__(self, img):
        return self.transforms(img)

class ClassificationPresetEval:
    def __init__(self, crop_size, resize_size, mean, std,
    ):
        
        self.transforms = tv_transf.Compose([
            tv_transf.PILToTensor(),
            tv_transf.Resize(resize_size, antialias=True),
            tv_transf.CenterCrop(crop_size),
            tv_transf.ConvertDtype(torch.float),
            tv_transf.Normalize(mean=mean, std=std)])

    def __call__(self, img):
        return self.transforms(img)

def view_images(ds, batch, n):

    for i in range(n):
        img_or, label_or = ds.getitem(i)
        img = batch[0][i]
        label = batch[1][i]

        plt.figure(figsize=[15, 3])
        plt.subplot(1, 4, 1)
        plt.imshow(img_or)
        plt.subplot(1, 4, 2)
        plt.imshow(label_or)
        plt.subplot(1, 4, 3)
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(1, 4, 4)
        plt.imshow(label, vmax=2)

def create_datasets(root, train_val_split, download=False, seed=None):
        '''Create train and validation datasets. Parameters `root` and `download` are the same as in OxfordIIITPet.'''
        
        # Get whole dataset
        ds = OxfordIIITPet(root, split='trainval', target_types=('category', 'segmentation'), download=download)
        # Get properties
        images, labels, segs, classes, class_to_idx = ds._images, ds._labels, ds._segs, ds.classes, ds.class_to_idx

        # Split into train and validation
        n_valid = round(train_val_split*len(images))
        random.seed(seed)
        data = list(zip(images, labels, segs))
        random.shuffle(data)
        data_train = list(zip(*data[n_valid:]))
        data_valid = list(zip(*data[:n_valid]))

        # Set transformations
        train_transforms = ClassificationPresetTrain(crop_size=224, mean=(0.4394, 0.4394, 0.4394), std=(0.2347, 0.2347, 0.2347))
        valid_transforms = ClassificationPresetEval(crop_size=224, resize_size=256, mean=(0.4394, 0.4394, 0.4394), std=(0.2347, 0.2347, 0.2347))
        
        ds_train = OxfordIIITPetV2(*data_train, classes, class_to_idx, train_transforms)
        ds_valid = OxfordIIITPetV2(*data_valid, classes, class_to_idx, valid_transforms)

        return ds_train, ds_valid