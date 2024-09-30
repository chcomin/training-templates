'''Script used for creating a VessMAP dataset object.'''

from pathlib import Path
from functools import partial
import os
import numpy as np
from PIL import Image
import albumentations as aug
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from .. import util


class VessMAP(Dataset):

    def __init__(self, root, transforms=None):
        """
        Args:
            root: root directory.
            transforms: function that receives as input an image and a label and
            returns a transformed image and label.
        """

        root = Path(root)
        images_folder = root / "images"
        labels_folder = root / "annotator1/labels"

        files = os.listdir(images_folder)
        

        files_labels = [file.replace('.tiff', '.png') for file in files]
        files_labels = [labels_folder/f""]

        self.images = files
        self.labels = files_labels
        self.transforms = transforms

    def __getitem__(self, idx, apply_transform=True):

        image = np.array(Image.open(self.images[idx]))
        target = np.array(Image.open(self.labels[idx]))//255

        if self.transforms is not None and apply_transform:
            image, target = self.transforms(image, target)

        return image, target
    
    def __len__(self):
        return len(self.images)

def multiplicative(img, mult_val_limit, **kwargs):
    """Generate custom function for Albumentations. The augmentation multiply
    the image by a constant value drawn uniformly in the range defined by `mult_val_limit`."""
    mult_val = np.random.rand()*(mult_val_limit[1]-mult_val_limit[0]) + mult_val_limit[0]
    return (img*mult_val).astype(np.uint8)

def zscore(img, **kwargs):
    return ((img-img.mean())/img.std()).astype(np.float32)

def create_transform(type='train-simple'):
    """Create a transform function with signature transform(image, label)."""

    if type=='train-simple':
        transform_comp = aug.Compose([
            #aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])
    elif type=='train-geometric':
        transform_comp = aug.Compose([
            aug.Flip(),
            aug.RandomRotate90(),
            aug.Transpose(),
            aug.RandomSizedCrop(min_max_height=(230,256), height=256, width=256),
            #aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2(),
        ])
    elif type=='train-full':
        transform_comp = aug.Compose([
            aug.OneOf([
                aug.GaussianBlur(blur_limit=(3, 7)),
                aug.Sharpen(alpha=(0.2, 0.5), lightness=(0., 1.)),
                aug.UnsharpMask(blur_limit=(3, 15), alpha=(0.4, 1.), threshold=0, p=0.1)  # slow, 0.038s/img
            ], p=1.),
            aug.OneOf([
                aug.RandomGamma(gamma_limit=(90, 150)),
                aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(-0.2, 0.5)),
                aug.Lambda(name='multiply', image=partial(multiplicative, mult_val_limit=(0.7, 1.)), p=0.1)
            ], p=1.),
            aug.GaussNoise(var_limit=(50, 250), p=1.),  # slow, 0.051s/img
            aug.Flip() ,
            aug.RandomRotate90(),
            aug.Transpose(),
            aug.ShiftScaleRotate(shift_limit_x=0.1, shift_limit_y=0.1, scale_limit=0.25, rotate_limit=45),
            #aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2(),
        ])
    elif type=='validation':
        transform_comp = aug.Compose([
            #aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])        

    transform_func = partial(util.transform_wrapper, transform_comp=transform_comp)

    return transform_func

def create_datasets(img_dir, label_dir, train_val_split=0.1, transform_type='train-simple', seed=None):
    """Create dataset from directory. 
    
    Args
    train_val_split: percentage of images used for validation
    use_simple: data augmentation to use. If 'train-simple', the images are only normalized and transformed 
    to tensors.
    seed: seed for splitting the data
    """

    train_transform = create_transform(type=transform_type)
    valid_transform = create_transform(type='validation')

    ds = VessMAP(root)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)


    return ds_train, ds_valid
