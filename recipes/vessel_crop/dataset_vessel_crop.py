'''Creates the vessel_mini dataset.'''

from functools import partial
import numpy as np
from PIL import Image
import torch
import albumentations as aug
from albumentations.pytorch import ToTensorV2
from torchtrainer.imagedataset import ImageSegmentationDataset

def name_to_label_map(img_path):
    return img_path.replace('.tiff', '.png')

def img_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil)

def label_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil)//255

def transform(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`"""
    res = transform_comp(image=image, mask=mask)
    image, mask = res['image'], res['mask']

    return image, mask.long()

def zscore(img, **kwargs):
    return ((img-img.mean())/img.std()).astype(np.float32)

def create_transform(mean, std, type='train'):
    """Create a transform function with signature transform(image, label)."""

    if type=='train':
        transform_comp = aug.Compose([
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            #aug.Normalize(mean=mean, std=std),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])
    elif type=='validate':
        transform_comp = aug.Compose([
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            #aug.Normalize(mean=mean, std=std),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])        

    transform_func = partial(transform, transform_comp=transform_comp)

    return transform_func

def create_datasets(img_dir, label_dir, train_val_split=0.1, seed=None):
    """Create dataset from directory. 
    
    Args
    crop_size: (height,width) to crop the images
    train_val_split: percentage of images used for validation
    use_simple: if True, use only crop, normalization and ToTensor. If False, use many data augmentation
    transformations.
    seed: seed for splitting the data
    """

    mean_data = 0.
    std_data = 1.
    class_weights = torch.tensor([0.3414, 0.6586])  

    train_transform = create_transform(mean=mean_data, std=std_data, type='train')
    valid_transform = create_transform(mean=mean_data, std=std_data, type='validate')

    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener, label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)

    meta = {
        'mean_data':mean_data,
        'std_data':std_data,
        'class_weights':class_weights
    }

    return ds_train, ds_valid, meta
