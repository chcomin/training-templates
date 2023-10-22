import time
from pathlib import Path
import random
import numpy.random as np_random
import torch
import torch.nn as nn
import torchtrainer   #https://github.com/chcomin/torchtrainer
from .dataset_readers import mnist
import torchvision

default_params = {
    # Dataset
    'img_dir': None,                    # Images path
    'label_dir': None,                  # Labels path
    'crop_size': (256, 256),            # Crop size for training
    'train_val_split': 0.1,             # Train/validation split
    'use_transforms': False,            # Use data augmentation
    # Model
    'model_layers': (3, 3, 3),          # Number of residual blocks at each layer of the model
    'model_channels': (16,32,64),       # Number of channels at each layer
    'model_type': 'unet',               # Model to use
    # Training
    'epochs': 1,
    'lr': 0.01,
    'batch_size_train': 8,
    'batch_size_valid': 8, 
    'momentum': 0.9,                    # Momentum for optimizer
    'weight_decay': 0.,
    'seed': 12,                         # Seed for random number generators
    'loss': 'cross_entropy',
    'scheduler_power': 0.9,             # Power por the polynomial scheduler
    'class_weights': (0.367, 0.633),    # Weights to use for cross entropy
    # Efficiency
    'device': 'cuda',
    'num_workers': 3,                   # Number of workers for the dataloader
    'use_amp': True,                    # Mixed precision
    'pin_memory': False,            
    'non_blocking': False,
    # Logging
    'log_dir': 'logs_unet',             # Directory for logging metrics and model checkpoints
    'experiment':'unet_l_3_c_16_32_64', # Experiment tag
    'save_every':1,                     # Number of epochs between checkpoints
    'save_best':True,                   # Save model with best validation loss
    'meta': None,                       # Additional metadata to save
    # Other
    'resume': False,                    # Resume from previous training
}

class CustomCNN(nn.Module):
    def __init__(self, channels, fc_channels=128, kernel_size=3):
        super(CustomCNN, self).__init__()
       
        # First block
        self.conv1 = nn.Conv2d(1, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
       
        # Second block
        self.conv2 = nn.Conv2d(channels, 2 * channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(2 * channels)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
       
        # Third block
        self.conv3 = nn.Conv2d(2*channels, 4 * channels, kernel_size, padding=kernel_size//2, dilation=2)
        self.bn3 = nn.BatchNorm2d(4 * channels)
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(2)
       
        # Fourth block (no pooling after this)
        self.conv4 = nn.Conv2d(4*channels, 8 * channels, kernel_size, padding=kernel_size//2)
        self.bn4 = nn.BatchNorm2d(8 * channels)
        self.relu4 = nn.ReLU()
       
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(8*channels, fc_channels)
        self.relu_fc = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_channels, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
       
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.drop1(self.relu_fc(self.fc1(x)))
        x = self.fc2(x)
       
        return x

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
        print_line(f'Batch {batch_idx+1}/{batches_per_epoch}, Train loss: {loss}')

    return train_loss/ds_size

def validate(model, data_loader_valid, loss_func, device, ds_size):

    model.eval()
    with torch.no_grad():
        valid_stats = torch.zeros(2)
        for batch_idx, (image, target) in enumerate(data_loader_valid):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss_valid = loss_func(output, target).item()  
            res_labels = torch.argmax(output, dim=1)
            acc = (res_labels==target).sum()/len(target)

            # Last batch might have different dimension, we multiply by batch size to calculate the average correctly
            stats = torch.tensor([loss_valid, acc])*image.shape[0]
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
    
    loss_func = nn.CrossEntropyLoss() 

    # Create optimizer and logger
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=len(data_loader_train)*epochs, power=scheduler_power)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger = torchtrainer.util.Logger(['Train loss', 'Valid loss', 'Acc'])
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
        print(f'Epoch {epoch}/{start_epoch+epochs-1}')
        time_start = time.time()

        loss = train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, device, non_blocking, use_amp, scaler, batches_per_epoch, len(ds_train))
        valid_stats = validate(model, data_loader_valid, loss_func, device, len(ds_valid))

        time_epoch = time.time() - time_start
        print(f'\nEpoch finished in {time_epoch:.1f} seconds')
        print(f'Train loss: {loss}, Valid loss: {valid_stats[0]}, Acc: {valid_stats[1]}')
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

    return logger

def run(user_params):

    params = process_params(user_params)
    experiment_folder = initial_setup(params)

    # Dataset
    ds_train, ds_valid = mnist.create_datasets(params['root_dir'], params['train_val_split'])

    # Model
    model = CustomCNN(channels=params['channels'], fc_channels=params['fc_channels'], kernel_size=3)
    #model = torchvision.models.resnet50(num_classes=10, replace_stride_with_dilation=[False, False, True])

    # Training
    logger = train(model, ds_train, ds_valid, experiment_folder, **params)
    
    return logger, ds_train, ds_valid, model
