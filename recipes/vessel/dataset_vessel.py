'''Script used for creating a VessMAP dataset object.'''

from functools import partial
import numpy as np
from PIL import Image
import albumentations as aug
from albumentations.pytorch import ToTensorV2
from torchtrainer.imagedataset import ImageSegmentationDataset
from torchvision.datasets.utils import download_and_extract_archive

def download(directory):

    url = 'https://zenodo.org/records/10045265/files/VessMAP.zip?download=1'
    download_root = directory
    extract_root = f'{directory}/vessmap'
    filename = 'VessMAP.zip'
    download_and_extract_archive(url, download_root, extract_root, filename, remove_finished=True)

def name_to_label_map(img_path):
    """Maps an image filename to the respective label."""
    return img_path.replace('.tiff', '.png')

def img_opener(img_path):
    """Opens an image given its path. The function must return a numpy array because
    it is the type that algumentations takes as input."""
    img_pil = Image.open(img_path)
    return np.array(img_pil)

def label_opener(img_path):
    """Opens a label given its path."""
    img_pil = Image.open(img_path)
    return np.array(img_pil)//255

def transform(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`. This function is useful because
    albumentations transforms need the image= and mask= keywords as input, and it also returns a
    dictionary. Using this function, we can call the transforms as 
    
    image_t, label_t = transform(image, label)
    
    instead of having to deal with dictionaries."""

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

    transform_func = partial(transform, transform_comp=transform_comp)

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

    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener, label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)


    return ds_train, ds_valid
