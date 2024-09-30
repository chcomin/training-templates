import random
from pathlib import Path
import numpy.random as np_random
import torch
from torch.utils.data import Dataset

class Subset(Dataset):
    """Create a new Dataset containing a subset of images from the input Dataset.
    """

    def __init__(self, ds, indices, transform=None):
        """
        Args:
            ds : input Dataset
            indices: indices to use for the new dataset
            transform: transformations to apply to the data. Defaults to None.
        """

        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):

        items = self.ds[self.indices[idx]]
        if self.transform is not None:
            items = self.transform(*items)

        return items

    def __len__(self):
        return len(self.indices)
    
class Logger:
    """Simple class for logging data. The class is initialized with a list
    containing metrics names. The method `add_data` can be used for adding new 
    metric values for a given epoch."""

    def __init__(self, columns):
        self.data = {}
        self.columns = columns

    def add_data(self, epoch, new_data):
        """Add new data. `new_data` must be a list containing values on the same
        order as self.columns."""
        self.data[epoch] = new_data

    def get_data(self):
        """Get each logged metric as a separate list."""
        epochs, metrics = zip(*self.data.items())
        metrics = zip(*metrics)

        return (epochs, *metrics)

    def state_dict(self):
        return {'columns':self.columns, 'data':self.data}
    
    def load_state_dict(self, state_dict):
        self.columns = state_dict['columns']
        self.data = state_dict['data']

def transform_wrapper(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`. This function is useful because
    albumentations transforms need the image= and mask= keywords as input, and it also returns a
    dictionary. Using this function, we can call the transforms as 
    
    image_t, label_t = transform(image, label)
    
    instead of having to deal with dictionaries."""

    res = transform_comp(image=image, mask=mask)
    image, mask = res['image'], res['mask']

    return image, mask.long()

def seed_worker(worker_id):
    """Set Python and numpy seeds for dataloader workers. Each worker receives a different seed in initial_seed()."""
    worker_seed = torch.initial_seed() % 2**32
    np_random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    "Seed Pytorch, numpy and Python."
    torch.manual_seed(seed)
    random.seed(seed) 
    np_random.seed(seed)

def initial_setup(user_params, default_params=None):
    """Initial setup before training."""

    # The Pytorch dev team recommends to set the precision to high. The default
    # value "highest" is unnecessary for the vast majority of situations.
    torch.set_float32_matmul_precision('high')

    # Use default value of a parameter in case it was not provide.
    if default_params is None:
        params = user_params
    else:
        params = default_params.copy()
        for k, v in user_params.items():
            params[k] = v

    # Add parameters as a meta attribute, which is useful for tracking experiments.
    if params['meta'] is None:
        params['meta'] = params.copy()
    else:
        params['meta'] = (params['meta'], params.copy())
    
    # Set deterministic training if a seed is provided
    seed = params['seed']
    if seed is not None:
        seed_everything(seed)

    # Create experiment folder
    experiment_folder = Path(params['log_dir'])/str(params["experiment"])
    experiment_folder.mkdir(parents=True, exist_ok=True)

    return params, experiment_folder