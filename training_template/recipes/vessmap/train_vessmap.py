from pathlib import Path
import random
import numpy.random as np_random
import torch
import torch.nn as nn
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# The dictionary below contains some common parameters used for training.
# Specific recipes might have different parameters
default_params = {
    ## Dataset
    'img_dir': None,            # Images path
    'label_dir': None ,         # Labels path   
    'train_val_split': 0.1,     # Train/validation split
    'transform_type': None,     # Which data augmentation to use
    ## Model
    # 
    ## Training
    'epochs': 5,                # Number of epochs to train
    'lr': 0.01,                 # Learning rate
    'batch_size_train': 64,     # Batch size used for training
    'batch_size_valid': 64,     # Batch size used for validation
    'optimizer_class': 'sgd',
    'momentum': 0.9,            # Momentum for optimizer
    'weight_decay': 0.1,        # Weight decay used for regularization
    'seed': 12,                 # Seed for random number generators
    'loss': 'cross_entropy',    # Loss function
    'scheduler_power': 0.9,     # Power for the polynomial learning rate scheduler
    'class_weights': (1., 1.),  # Class weights to use for cross entropy calculation
    # Efficiency
    'device': 'cuda',           # Device to use for training
    'num_workers': 0,           # Number of workers for the dataloader
    'use_amp': False,           # Mixed precision training
    'pin_memory': False,        # Pinned memory sometimes lead to faster cpu->GPU copy
    # Logging
    'log_dir': 'logs/',         # Directory for logging metrics and model checkpoints
    'experiment':'default',     # Experiment tag
    'save_every':1,             # Number of epochs between model checkpoints     
    'save_best':True,           # Save model with best validation loss
    'meta': None,               # Additional metadata to save
    # Other 
    'resume': False             # Resume from previous training
}

##### Utility functions used during training
def plot_figure(logger, fig=None):
    """Plot loss and accuracy metrics in a notebook. It is assumed that the
    first two metrics in the logger correspond to the train a validation
    losses."""

    epochs, train_loss, valid_loss, *metrics = logger.get_data()
    #iou, prec, rec = zip(*metrics)

    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
        ax1.plot(epochs, train_loss, label='Train loss')
        ax1.plot(epochs, valid_loss, label='Valid loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_ylim((0,0.5))
        ax1.legend()
        for name, metric in zip(logger.columns[2:], metrics):
            ax2.plot(epochs, metric, label=name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylim((0,1.))
        ax2.legend()
        fig.tight_layout()
    else:
        ax1, ax2 = fig.axes
        ax1.lines[0].set_data(epochs, train_loss)
        ax1.lines[1].set_data(epochs, valid_loss)
        ax1.set_xlim((0,epochs[-1]))
        for idx, (name, metric) in enumerate(zip(logger.columns[2:], metrics)):
            ax2.lines[idx].set_data(epochs, metric)
        ax2.set_xlim((0,epochs[-1]))
        #fig.canvas.draw()

    display(fig)
    # Clear the previous plot. Wait for the new plot to complete before clearing.
    clear_output(wait=True) 

    return fig

def save_experiment(checkpoint_file, model, optimizer, epoch, meta, 
                    lr_scheduler=None, logger=None, scaler=None):
    """Save experiment data."""

    # Mandatory experiment data
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "meta": meta,
    }
    # Optional experiment data
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if logger is not None:
        checkpoint["logger"] = logger.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    # Save data
    torch.save(checkpoint, checkpoint_file)

def load_experiment(checkpoint_file, model, optimizer,   
                    lr_scheduler=None, logger=None, scaler=None):
    """Load experiment data."""

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if logger is not None:
        logger.load_state_dict(checkpoint["logger"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return start_epoch

##### End of utiliy functions

def train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, 
                    device, use_amp, scaler, ds_size):

    model.train()
    train_loss = 0.
    for batch_idx, (images, targets) in enumerate(data_loader_train):
        images = images.to(device)
        targets = targets.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):    # Forward pass in mixed precision
            output = model(images)
            loss = loss_func(output, targets)

        optimizer.zero_grad()
        if scaler is not None:                 # Reescale loss to avoid mixed precision errors
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss = loss.item()
        # Last batch might have different dimension, we multiply by batch size to calculate the average correctly
        train_loss += loss*images.shape[0]

    return train_loss/ds_size

def validate(model, data_loader_valid, loss_func, device, ds_size):

    model.eval()
    with torch.no_grad():
        valid_stats = torch.zeros(2)
        for batch_idx, (images, targets) in enumerate(data_loader_valid):
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            loss_valid = loss_func(output, targets).item()  
            # Example of accuracy metrics that can be calculated
            #metrics = torchtrainer.perf_funcs.segmentation_accuracy(output, targets)
            # Placeholder metric just as an example for this template. The actual
            # metrics will depend on the task
            acc = (output.argmax(dim=1)==targets).mean(dtype=float)
            # Validation metrics must be inside a tensor
            stats = torch.tensor([loss_valid, acc])
            
            # Last batch might have different dimension, we multiply by batch size to calculate the average correctly
            stats = stats*images.shape[0]
            valid_stats += stats
        valid_stats /= ds_size

    return valid_stats
        
def train(model, ds_train, ds_valid, experiment_folder, loss, class_weights, epochs, lr, 
          batch_size_train, batch_size_valid, optimizer_class, momentum=0.9, weight_decay=0., 
          scheduler_power=0.9, device='cuda', num_workers=0, use_amp=False, pin_memory=False, 
          resume=False, save_every=1, save_best=True, seed=None, meta=None, **kwargs):

    start_epoch = 1
    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,  # Avoid recreating workers at each epoch
        worker_init_fn=seed_worker,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
    )
 
    # Define loss function
    if loss=='cross_entropy':
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device)) 

    # Create optimizer and logger
    model.to(device)
    if optimizer_class=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_class=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=len(data_loader_train)*epochs, power=scheduler_power)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger = Logger(['Train loss', 'Valid loss', 'Acc'])
    checkpoint_file = str(experiment_folder/f'checkpoint.pth')
    if resume:
        # Resume training
        start_epoch = load_experiment(checkpoint_file, model, optimizer,   
                                    lr_scheduler, logger, scaler)
        if seed is not None:
            # Seed using the current epoch to avoid using the same seed as in epoch 0 when resuming
            seed_everything(seed+start_epoch)

    # Number of batches for each epoch
    #batches_per_epoch = len(ds_train)//batch_size_train + 1*(len(ds_train)%batch_size_train>0)
    best_valid_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+epochs):
        train_loss = train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, 
                               device, use_amp, scaler, len(ds_train))
        valid_stats = validate(model, data_loader_valid, loss_func, device, len(ds_valid))

        logger.add_data(epoch, [train_loss]+valid_stats.tolist())

        if (epoch+1)%save_every==0 or epoch==(start_epoch+epochs-1):
            # Save current experiment data
            save_experiment(checkpoint_file, model, optimizer, epoch, meta, 
                            lr_scheduler, logger, scaler)
        # Save model with lowest validation loss
        if save_best and valid_stats[0]<best_valid_loss:
            name_best = checkpoint_file.replace('.pth', '_best.pth')
            save_experiment(name_best, model, optimizer, epoch, meta, 
                            lr_scheduler, logger, scaler)
            best_valid_loss = valid_stats[0]

        # If epoch==1, create new figure. Else, use previous figure
        if epoch==1:
            fig = plot_figure(logger)
        else:
            plot_figure(logger, fig)

    return logger

def create_components(model, ds_train, ds_valid, loss, class_weights, epochs, lr, 
          batch_size_train, batch_size_valid, optimizer_class, momentum=0.9, weight_decay=0., 
          scheduler_power=0.9, device='cuda', num_workers=0, pin_memory=False, 
          **kwargs):

    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,  # Avoid recreating workers at each epoch
        worker_init_fn=seed_worker,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
    )
 
    # Define loss function
    if loss=='cross_entropy':
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device)) 

    # Create optimizer and logger
    if optimizer_class=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_class=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=len(data_loader_train)*epochs, power=scheduler_power)
    logger = Logger(['Train loss', 'Valid loss', 'Acc'])

    return data_loader_train, data_loader_valid, loss_func, optimizer, lr_scheduler, logger

def train(model, data_loader_train, data_loader_valid, loss_func, optimizer, lr_scheduler, logger, 
          experiment_folder, epochs, device='cuda', use_amp=False, 
          resume=False, save_every=1, save_best=True, seed=None, meta=None, **kwargs):

    start_epoch = 1
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    checkpoint_file = str(experiment_folder/f'checkpoint.pth')
    if resume:
        # Resume training
        start_epoch = load_experiment(checkpoint_file, model, optimizer,   
                                    lr_scheduler, logger, scaler)
        if seed is not None:
            # Seed using the current epoch to avoid using the same seed as in epoch 0 when resuming
            seed_everything(seed+start_epoch)

    # Number of batches for each epoch
    #batches_per_epoch = len(ds_train)//batch_size_train + 1*(len(ds_train)%batch_size_train>0)
    best_valid_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+epochs):
        train_loss = train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, 
                               device, use_amp, scaler, len(data_loader_train))
        valid_stats = validate(model, data_loader_valid, loss_func, device, len(data_loader_valid))

        logger.add_data(epoch, [train_loss]+valid_stats.tolist())

        if (epoch+1)%save_every==0 or epoch==(start_epoch+epochs-1):
            # Save current experiment data
            save_experiment(checkpoint_file, model, optimizer, epoch, meta, 
                            lr_scheduler, logger, scaler)
        # Save model with lowest validation loss
        if save_best and valid_stats[0]<best_valid_loss:
            name_best = checkpoint_file.replace('.pth', '_best.pth')
            save_experiment(name_best, model, optimizer, epoch, meta, 
                            lr_scheduler, logger, scaler)
            best_valid_loss = valid_stats[0]

        # If epoch==1, create new figure. Else, use previous figure
        if epoch==1:
            fig = plot_figure(logger)
        else:
            plot_figure(logger, fig)

    return logger

def run(user_params):

    params, experiment_folder = initial_setup(user_params, default_params)

    # Dataset loading. Typically, it will involve importing the dataset script 
    # as dataset_* and then doing 
    #ds_train, ds_valid = dataset_*.create_datasets(params['img_dir'], params['label_dir'], params['train_val_split'], params['transform_type'])

    # Model creation. Typically, it will involve importing the model script as
    # ModelName and then doing
    # model = ModelName(...)

    # Training
    logger = train(model, ds_train, ds_valid, experiment_folder, **params)
    
    return logger, ds_train, ds_valid, model


def _test_template():
    """Function for testing the template for development purposes."""
    import numpy as np
    from torchvision.datasets import MNIST
    from torchvision.models.resnet import resnet18

    def transform(img):
        img = np.array(img, dtype=np.float32)
        img = (img-33.3)/78.6
        img = np.tile(img, (3, 1, 1))
        return torch.from_numpy(img)

    user_params = default_params
    user_params['img_dir'] = 'K:/datasets/classification'
    user_params['class_weights'] = (1.,)*10

    ##
    params, experiment_folder = initial_setup(user_params)

    ds_train = MNIST(params['img_dir'], train=True, transform=transform, download=False)
    ds_train.data = ds_train.data[:10000]
    ds_train.targets = ds_train.targets[:10000]
    ds_valid = MNIST(params['img_dir'], train=False, transform=transform, download=False)

    model = resnet18()
    model.fc = nn.Linear(512, 10)

    # Training
    logger = TrainTemplate().train(model, ds_train, ds_valid, experiment_folder, **params)
    
    return logger, ds_train, ds_valid, model
