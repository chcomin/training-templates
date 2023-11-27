import time
from pathlib import Path
import random
import numpy.random as np_random
import torch
import torch.nn as nn
import torchtrainer   #https://github.com/chcomin/torchtrainer
import dataset_vessel
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# In case a parameter is not provided, the following default values are used:
default_params = {
    # Dataset
    'img_dir': Path('data/vessmap/images'),     # Images path
    'label_dir': Path('data/vessmap/annotator1/labels'),   # Labels path   
    'train_val_split': 0.1,                    # Train/validation split
    'transform_type': 'train-geometric',       # Which data augmentation to use
    # Model
    'model_layers': (8,),                      # Number of residual blocks at each layer of the model
    'model_channels': (64,),                   # Number of channels at each layer
    'decoder_layers': (1,),                    # Number of decoder blocks
    'model_type': 'unetv2',                    # Model to use
    # Training
    'epochs': 500,
    'lr': 0.01,
    'batch_size_train': 10,
    'batch_size_valid': 10, 
    'momentum': 0.9,                           # Momentum for optimizer
    'weight_decay': 0.1,
    'seed': 12,                                # Seed for random number generators
    'loss': 'cross_entropy',
    'scheduler_power': 0.9,                    # Power por the polynomial scheduler
    'class_weights': (0.2590, 0.7410),         # Class weights to use for cross entropy calculation
    # Efficiency
    'device': 'cuda',
    'num_workers': 0,                          # Number of workers for the dataloader
    'use_amp': False,                          # Mixed precision
    'pin_memory': False,
    'non_blocking': False,
    # Logging
    'log_dir': 'logs/',                        # Directory for logging metrics and model checkpoints
    'experiment':'default',                    # Experiment tag
    'save_every':1,                            # Number of epochs between checkpoints     
    'save_best':True,                          # Save model with best validation loss
    'meta': None,                              # Additional metadata to save
    # Other 
    'resume': False                            # Resume from previous training
}

def seed_worker(worker_id):
    """Set Python and numpy seeds for dataloader workers. Each worker receives a different seed in initial_seed()."""
    worker_seed = torch.initial_seed() % 2**32
    np_random.seed(worker_seed)
    random.seed(worker_seed)

def plot_figure(logger, fig=None):
    """Plot metrics in a notebook."""

    epochs, metrics = zip(*logger.data.items())
    train_loss, valid_loss, iou, prec, rec = zip(*metrics)

    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
        ax1.plot(epochs, train_loss, label='Train loss')
        ax1.plot(epochs, valid_loss, label='Valid loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_ylim((0,0.5))
        ax1.legend()
        ax2.plot(epochs, iou, label='IoU')
        ax2.plot(epochs, prec, label='Precision')
        ax2.plot(epochs, rec, label='Recall')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim((0,1.))
        ax2.legend()
        fig.tight_layout()
    else:
        ax1, ax2 = fig.axes
        ax1.lines[0].set_data(epochs, train_loss)
        ax1.lines[1].set_data(epochs, valid_loss)
        ax1.set_xlim((0,epochs[-1]))
        ax2.lines[0].set_data(epochs, iou)
        ax2.lines[1].set_data(epochs, prec)
        ax2.lines[2].set_data(epochs, rec)
        ax2.set_xlim((0,epochs[-1]))
        #fig.canvas.draw()

    display(fig)
    clear_output(wait=True) 

    return fig

def seed_everything(seed):
    "Seed Pytorch, numpy and Python."
    torch.manual_seed(seed)
    random.seed(seed) 
    np_random.seed(seed)

def process_params(user_params):
    '''Use default value of a parameter in case it was not provide. Also add the 
    parameters as a meta attribute.'''

    params = default_params.copy()
    for k, v in user_params.items():
        params[k] = v

    if params['meta'] is None:
        params['meta'] = params.copy()
    else:
        params['meta'] = (params['meta'], params.copy())

    return params

def initial_setup(params):

    torch.set_float32_matmul_precision('high')
    
    # Set deterministic training if a seed is provided
    seed = params['seed']
    if seed is not None:
        seed_everything(seed)

    experiment_folder = Path(params['log_dir'])/str(params["experiment"])
    experiment_folder.mkdir(parents=True, exist_ok=True)

    return experiment_folder

def print_line(text):
    print('\r\033[K'+text, end='')

def train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, device, non_blocking, use_amp, scaler, batches_per_epoch, ds_size):

    model.train()
    train_loss = 0.
    for batch_idx, (image, target) in enumerate(data_loader_train):
        image = image.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        with torch.cuda.amp.autocast(enabled=use_amp):    # Forward pass in mixed precision
            output = model(image)
            loss = loss_func(output, target)

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
        train_loss += loss*image.shape[0]

    return train_loss/ds_size

def validate(model, data_loader_valid, loss_func, device, ds_size):

    model.eval()
    with torch.no_grad():
        valid_stats = torch.zeros(4)
        for batch_idx, (image, target) in enumerate(data_loader_valid):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss_valid = loss_func(output, target).item()  
            metrics = torchtrainer.perf_funcs.segmentation_accuracy(output, target)

            # Last batch might have different dimension, we multiply by batch size to calculate the average correctly
            stats = torch.tensor([loss_valid, metrics['iou'], metrics['prec'], metrics['rec']])*image.shape[0]
            valid_stats += stats
        valid_stats /= ds_size

    return valid_stats
        
def train(model, ds_train, ds_valid, experiment_folder, loss, class_weights, epochs, lr, batch_size_train, batch_size_valid, momentum=0.9, weight_decay=0., scheduler_power=0.9, 
          device='cuda', num_workers=0, use_amp=False, pin_memory=False, non_blocking=False, resume=False, save_every=1, save_best=True, seed=None, meta=None, **kwargs):

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
    elif loss=='label_weighted_cross_entropy':
        loss_func = torchtrainer.perf_funcs.LabelWeightedCrossEntropyLoss()

    # Create optimizer and logger
    model.to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=len(data_loader_train)*epochs, power=scheduler_power)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger = torchtrainer.util.Logger(['Train loss', 'Valid loss', 'IoU', 'Prec', 'Rec'])
    checkpoint_file = str(experiment_folder/f'checkpoint.pth')
    if resume:
        # Resume training
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        logger.load_state_dict(checkpoint["logger"])
        if use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        if seed is not None:
            # Seed using the current epoch to avoid using the same seed as in epoch 0 when resuming
            seed_everything(seed+start_epoch)

    # Number of batches for each epoch
    batches_per_epoch = len(ds_train)//batch_size_train + 1*(len(ds_train)%batch_size_train>0)
    best_valid_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+epochs):
        time_start = time.time()

        loss = train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, device, non_blocking, use_amp, scaler, batches_per_epoch, len(ds_train))
        valid_stats = validate(model, data_loader_valid, loss_func, device, len(ds_valid))

        time_epoch = time.time() - time_start
        logger.add_data(epoch, [loss]+valid_stats.tolist())

        if (epoch+1)%save_every==0 or epoch==(start_epoch+epochs-1):
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                'logger':logger.state_dict(),
                "meta": meta,
            }
            if use_amp:
                checkpoint["scaler"] = scaler.state_dict()
            # Save data
            torch.save(checkpoint, checkpoint_file)
        # Save model with lowest validation loss
        if save_best and valid_stats[0]<best_valid_loss:
            torch.save(checkpoint, checkpoint_file.replace('.pth', '_best.pth'))
            best_valid_loss = valid_stats[0]

        if epoch==1:
            fig = plot_figure(logger)
        else:
            plot_figure(logger, fig)

    return logger

def run(user_params):

    params = process_params(user_params)
    experiment_folder = initial_setup(params)

    # Dataset
    ds_train, ds_valid = dataset_vessel.create_datasets(params['img_dir'], params['label_dir'], params['train_val_split'], params['transform_type'])

    # Model
    if params['model_type']=='unet':
        model = torchtrainer.models.resunet.ResUNet(params['model_layers'], params['model_channels'])
    if params['model_type']=='unetv2':
        model = torchtrainer.models.resunet.ResUNetV2(params['model_layers'], params['decoder_layers'], params['model_channels'])
    elif params['model_type']=='resnetseg':
        model = torchtrainer.models.resnet_seg.ResNetSeg(params['model_layers'], params['model_channels'])

    # Training
    logger = train(model, ds_train, ds_valid, experiment_folder, **params)
    
    return logger, ds_train, ds_valid, model
