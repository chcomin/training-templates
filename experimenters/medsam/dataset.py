from pathlib import Path
from random import random

from torchtrainer.datasets.vessel import TrainTransforms, ValidTransforms
from torchtrainer.datasets.vessel_base import VessMAP
from torchtrainer.util.train_util import Subset


class AdjustTransform:
    """Adjust the image to three channels and the target to float with the channel dimension.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):

        img, target = self.transforms(img, target)

        if img.shape[0] != 3:
            img = img.expand(3, -1, -1)
        target = target.float().unsqueeze(0)

        return img, target

def get_dataset(
        dataset_path, 
        split_strategy="file", 
        resize_size=(256, 256), 
        split_folder=None
        ):
    """Get the VessMAP dataset for training.

    Parameters
    ----------
    dataset_path
        Path to the dataset root folder
    split_strategy
        Strategy to split the dataset. Possible values are:
        "rand_<split>": Use <split> fraction of the images to validate
        "file": Use the train.csv and val.csv files to split the dataset
    resize_size
        Size to resize the images
    split_folder
        Folder containing the train.csv and val.csv files (only used if split_strategy is "file")
    """

    class_weights = (0.26, 0.74)
    ignore_index = None
    collate_fn = None

    dataset_path = Path(dataset_path)

    if "rand" in split_strategy:
        ds = VessMAP(dataset_path, keepdim=True)
        split = float(split_strategy.split("_")[1])
        n = len(ds)
        n_valid = int(n*split)

        indices = list(range(n))
        random.shuffle(indices)
        
        indices_train = indices[n_valid:]
        class_atts = {
            "images":[ds.images[idx] for idx in indices_train], 
            "labels":[ds.labels[idx] for idx in indices_train], 
            "classes":ds.classes
        }
        ds_train = Subset(ds, indices_train, **class_atts)

        indices_valid = indices[:n_valid]
        class_atts = {
            "images":[ds.images[idx] for idx in indices_valid], 
            "labels":[ds.labels[idx] for idx in indices_valid], 
            "classes":ds.classes
        }
        ds_valid = Subset(ds, indices_valid, **class_atts)

    elif "file" in split_strategy:
        split_folder = Path(split_folder)
        if split_strategy=="file_train_val":
            other_split = "val"
        elif split_strategy=="file_train_test":
            other_split = "test"
        
        with open(split_folder/"train.csv") as file:
            files_train = file.read().splitlines()
        with open(split_folder/f"{other_split}.csv") as file:
            files_valid = file.read().splitlines()
        ds_train = VessMAP(dataset_path, keepdim=True, files=files_train)
        ds_valid = VessMAP(dataset_path, keepdim=True, files=files_valid)

    ds_train.transforms = AdjustTransform(TrainTransforms(resize_size=resize_size))
    ds_valid.transforms = AdjustTransform(ValidTransforms(resize_size=resize_size))

    return ds_train, ds_valid, class_weights, ignore_index, collate_fn
