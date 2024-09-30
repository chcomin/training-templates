# Training templates

Recipes for training neural networks using [PyTorch](https://pytorch.org/). The format of the recipes is the following:

* A dataset reader (`dataset_*.py`) script is responsible for doing all dataset operations (reading data, pre-processing, data augmentation, etc). The script must contain a `create_datasets` function that returns the train and validation datasets as well as any optional meta information about the datasets. The parameters of the function are not fixed, that is, they can be changed between different reader scripts. The name of the script should contain the name of the dataset (e.g., dataset_mnist.py).
* A `train_*.py` script responsible for creating and training the network. The script must contain a `run` function that receives a dictionary of parameters, creates the dataset (by calling `create_datasets` from the dataset reader) and the network model and train the network. It is recommended that the `run` function returns the train and validation datasets, the trained model, as well as the metrics logged during training (losses, accuracies, etc). If the training involves a methodology that cannot be easily set by a parameter, a new script should be created (e.g., train_classification.py, train_segmentation.py, etc).
* An `experimenter_*.ipynb` notebook that calls the `run` function of a train script and do some experiments.

This repository is organized as follows:

* recipes: Directory containing training recipes
  * <recipe_1>
    * dataset_*.py: Script that reads the dataset from the disk and returns one or more torch.utils.data.Dataset objects.
    * train_*.py: Script containing a `run` function that receives a dictionary of parameters and train the network.
    * experimenter_*.ipynb: Experiments using the training script.
  * <recipe_2>
    * ...
  * ...
* util: Some utility functions. Forget about it.

The recipes use torchtrainer, a package that contains some utility functions for creating datasets, network architectures and calculating performance metrics. You can download the package at:

[https://github.com/chcomin/torchtrainer](https://github.com/chcomin/torchtrainer)

You can put the folder containing the package (torchtrainer) in the same directory as this notebook, or install the package by running the following command:

`pip install -e torchtrainer`

or if using conda:

`conda develop torchtrainer`

where `torchtrainer` is the directory containing the `pyproject.toml` file.
