# Pytorch-template

Recipes for training neural networks using [PyTorch](https://pytorch.org/). The format of the recipes is the following:

* A dataset reader script is responsible for doing all dataset operations (reading data, pre-processing, data augmentation, etc). The script must contain a `create_datasets` function that returns the train and validation datasets as well as any optional meta information about the datasets. The parameters of the function are not fixed, that is, they can be changed between different reader scripts. The name of the script should be the name of the dataset (e.g., mnist.py).
* A `train_*.py` script responsible for creating and training the network. The script must contain a `run` function that receives a dictionary of parameters, creates the dataset (by calling `create_datasets` from the dataset reader) and the network model and train the network. It is recommended that the `run` function returns the train and validation datasets, the trained model, as well as the metrics logged during training (losses, accuracies, etc). If the training involves a methodology that cannot be easily set by a parameter, a new script should be created (e.g., train_classification.py, train_segmentation.py, etc).
* A `experimenter_*.ipynb` notebook that calls the `run` function of a train script and do some experiments.

This repository is organized as follows:

```
code: The codes for all recipes
├── recipes: Folder containing training recipes
    ├── dataset_readers: Folder containing reader scripts for datasets
    ├── train_*.py: A training recipe
    ├── train_*.py: Another training recipe
    ...
├── experimenter_*.ipynb: Experiments using a training recipe
├── experimenter_*.ipynb: Experiments using another training recipe
...
└── download_dataset.py: A script for downloading some specific datasets.
```

This structure allows easily importing the required scripts inside the notebooks and training scripts.

The code depends on [Torchtrainer](https://github.com/chcomin/torchtrainer)
