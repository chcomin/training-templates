'''Creates the vessel_mini dataset.'''

from functools import partial
import numpy as np
from PIL import Image
import torch
import albumentations as aug
from albumentations.pytorch import ToTensorV2
from torchtrainer.imagedataset import ImageSegmentationDataset
from torchvision.datasets.utils import download_and_extract_archive

def download(directory):

    url = 'https://www.dropbox.com/scl/fi/2h9yoz64gr7svjppj746t/vessel.tar.gz?rlkey=zx53tq0guohrk5ulx0uf7sefl&dl=1'
    download_root = directory
    extract_root = directory
    filename = 'vessel.tar.gz'
    download_and_extract_archive(url, download_root, extract_root, filename, remove_finished=True)

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
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
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
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2(),
        ])
    elif type=='validation':
        transform_comp = aug.Compose([
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])        

    transform_func = partial(transform, transform_comp=transform_comp)

    return transform_func

def create_datasets(img_dir, label_dir, train_val_split=0.1, use_simple=True, seed=None):
    """Create dataset from directory. 
    
    Args
    train_val_split: percentage of images used for validation
    use_simple: data augmentations to use. If True, the images are only normalization and transformed 
    to tensors. If False, use many data augmentations.
    seed: seed for splitting the data
    """

    if use_simple:
        train_type = 'train-simple'
    else:
        train_type = 'train-full'

    train_transform = create_transform(type=train_type)
    valid_transform = create_transform(type='validation')

    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener, label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)


    return ds_train, ds_valid

# Statistics for vessel_mini
# Original data:
# mean: 34.5217
# std: 12.3777
# With CLAHE clip_limit=(3., 3.), tile_grid_size=(16, 16)
# mean: 52.046542909953075 
# std: 28.702299672350073
# Class weights (background, vessel)
# 0.3414, 0.6586

# Statistics for the whole vessel data
# Original data:
# mean: 35.0115
# std: 12.291
# With CLAHE clip_limit=(3., 3.), tile_grid_size=(16, 16)
# mean: 51.969
# std: 28.4325
# Class weights (background, vessel)
# 0.367, 0.633