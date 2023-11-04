import random
from functools import partial
from PIL import Image
import numpy as np
from torchvision.datasets import OxfordIIITPet
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from gtimer import Timer

class OxfordIIITPetV2(Dataset):
    '''Create Dataset with similar attributes as OxfordIIITPet. This class is necessary for splitting
    the dataset into train and validation. It also defines a .getitem function that returns the images
    without any transformation.'''
   
    def __init__(self, images, labels, segs, classes, class_to_idx, transforms=None, target_type='segmentation'):
        super().__init__()

        self._images = images
        self._labels = labels
        self._segs = segs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transforms = transforms
        self.target_type = target_type

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx, use_transforms=True):
        image = Image.open(self._images[idx]).convert("RGB")

        if self.target_type == "category":
            label = self._labels[idx]
        else: 
            label = Image.open(self._segs[idx])

        if self.transforms and use_transforms:
            image, label = self.transforms(image, label)

        return image, label

    def getitem(self, idx):
        return self.__getitem__(idx, use_transforms=False)
    
class OxfordIIITPetDummy(Dataset):
   
    def __init__(self, images):
        super().__init__()

        self._images = images
        self.img = torch.rand(3,700,700)
        self.label = torch.randint(0,2,(700,700), dtype=torch.long)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return self.img, self.label
    
def download(directory):
    OxfordIIITPet(directory, split='trainval', target_types='segmentation', download=True)

def transform(image, label, size_smaller, size_larger, mean, std):

    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)

    image, label = conversions(image, label)
    image = v2.functional.resize(image, size=size_smaller, max_size=size_larger, 
                                interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
    label = v2.functional.resize(label[None], size=size_smaller, max_size=size_larger, 
                                interpolation=v2.InterpolationMode.NEAREST_EXACT)[0]
    image = image.float()/255.
    image = (image-mean)/std

    return image, label

def conversions(img, label):

    # Transform to tensor. Range is [0,255]
    img_t = torch.from_numpy(np.array(img).transpose((2, 0, 1))).contiguous()
    
    # Transform to tensor. Range is [0,3]
    label_t = torch.from_numpy(np.array(label)).contiguous()
    label_t = label_t.to(dtype=torch.long)
    # Original label is 1: Foreground, 2: Background, 3: Border, change to the more common
    # 0: Background, 1: Foreground, 255: Border
    label_t = 2 - label_t
    label_t[label_t==-1] = 255

    return img_t, label_t

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    '''Function for collating batch.'''
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

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
        train_transforms = partial(transform, size_smaller=480, size_larger=2*480, mean=(0.4394, 0.4394, 0.4394), std=(0.2347, 0.2347, 0.2347))
        valid_transforms = partial(transform, size_smaller=480, size_larger=2*480, mean=(0.4394, 0.4394, 0.4394), std=(0.2347, 0.2347, 0.2347))
        
        ds_train = OxfordIIITPetV2(*data_train, classes, class_to_idx, train_transforms)
        ds_valid = OxfordIIITPetV2(*data_valid, classes, class_to_idx, valid_transforms)

        #ds_train = OxfordIIITPetDummy(list(range(len(ds)-n_valid)))
        #ds_valid = OxfordIIITPetDummy(list(range(n_valid)))

        return ds_train, ds_valid