import time
from pathlib import Path
import random
import numpy.random as np_random
import torch
import torch.nn as nn
import torchtrainer   #https://github.com/chcomin/torchtrainer
from dataset import create_datasets

defaul_params = {
    # Dataset
    'img_dir': None,
    'label_dir': None,
    'crop_size': (256, 256),          
    'train_val_split': 0.1,
    'use_transforms': False,
    # Model
    'model_layers': (1, 1, 1, 1), 
    'model_channels': (4, 4, 4, 4), 
    # Training
    'epochs': 1,
    'lr': 0.001,
    'batch_size': 8,
    'momentum': 0.9,
    'weight_decay': 0.,
    'seed': 12,
    'loss': 'cross_entropy',
    'class_weights': (0.367, 0.633),
    # Efficiency
    'device': 'cuda',
    'num_workers': 3,
    'use_amp': False,
    'pin_memory': False,
    'non_blocking': False,
    # Other
    'resume': False,
    'log_dir': 'logs',
    'version':1,
    'save_best':True,
}

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

def train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, device, non_blocking, use_amp, scaler, batches_per_epoch):

    model.train()
    train_loss = 0.
    for batch_idx, (image, target) in enumerate(data_loader_train):
        image = image.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(image)
            loss = loss_func(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss = loss.item()
        train_loss += loss*image.shape[0]
        print(f'Batch {batch_idx+1}/{batches_per_epoch}, Train loss: {loss}')

    return train_loss

def validate(model, data_loader_valid, loss_func, device, ds_size):

    model.eval()
    with torch.no_grad():
        valid_stats = torch.zeros(4)
        for batch_idx, (image, target) in enumerate(data_loader_valid):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss_valid = loss_func(output, target).item()  
            acc = torchtrainer.perf_funcs.segmentation_accuracy(output, target)

            # Last batch might have different dimension, we multiply by batch size to calculate the average correctly
            stats = torch.tensor([loss_valid, acc['iou'], acc['prec'], acc['rec']])*image.shape[0]
            valid_stats += stats
        valid_stats /= ds_size

    return valid_stats
        
def train(model, ds_train, ds_valid, loss, class_weights, epochs, lr, batch_size, momentum=0.9, weight_decay=0., device='cuda', num_workers=0, 
          use_amp=False, pin_memory=False, non_blocking=False, resume=False, experiment_folder='logs', save_best=True, seed=None, meta=None):

    start_epoch = 1
    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,  # Avoid recreating workers at each epoch
        worker_init_fn=seed_worker
    )
    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0
    )
 
    # Define loss function
    if loss=='cross_entropy':
        loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device)) 
    elif loss=='label_weighted_cross_entropy':
        loss_func = torchtrainer.perf_funcs.LabelWeightedCrossEntropyLoss()

    # Create optimizer and logger
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=len(data_loader_train)*epochs, power=0.9)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger = torchtrainer.util.Logger(['Train loss', 'Valid loss', 'IoU', 'Prec', 'Rec', 'Time'])
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

    batches_per_epoch = len(ds_train)//batch_size + 1*(len(ds_train)%batch_size>0)
    best_valid_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'Epoch {epoch}/{start_epoch+epochs-1}')
        time_start = time.time()

        loss = train_one_epoch(model, data_loader_train, loss_func, optimizer, lr_scheduler, device, non_blocking, use_amp, scaler, batches_per_epoch)
        valid_stats = validate(model, data_loader_valid, loss_func, device, len(ds_valid))

        time_epoch = time.time() - time_start
        print(f'Epoch finished in {time_epoch:.1f} seconds')
        print(f'Valid loss: {valid_stats[0]}, IoU: {valid_stats[1]}, Prec: {valid_stats[2]}, Rec {valid_stats[3]}')
        logger.add_data(epoch, [loss/len(ds_train)]+valid_stats.tolist())

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
        torch.save(checkpoint, checkpoint_file)

        if save_best and valid_stats[0]<best_valid_loss:
            torch.save(checkpoint, checkpoint_file.replace('.pth', '_best.pth'))

    return logger

def run(user_params):

    params = defaul_params.copy()
    for k, v in user_params.items():
        params[k] = v
    
    # Set deterministic training if a seed is provided
    seed = params['seed']
    if seed is not None:
        seed_everything(seed)

    ds_train, ds_valid, _ = create_datasets(params['img_dir'], params['label_dir'], params['crop_size'], params['train_val_split'], use_simple=not params['use_transforms'])
    model = torchtrainer.models.resnet_seg.ResNetSeg(params['model_layers'], params['model_channels'])

    experiment_folder = Path(params['log_dir'])/f'version_{params["version"]}'
    experiment_folder.mkdir(parents=True, exist_ok=True)

    logger = train(model, ds_train, ds_valid, params['loss'], params['class_weights'], params['epochs'], params['lr'], params['batch_size'], momentum=params['momentum'], 
                   weight_decay=params['weight_decay'], device=params['device'], num_workers=params['num_workers'], use_amp=params['use_amp'], 
                   pin_memory=params['pin_memory'], non_blocking=params['non_blocking'], resume=params['resume'], experiment_folder=experiment_folder, 
                   save_best=params['save_best'], seed=seed, meta=params)
    
    return logger, ds_train, ds_valid, model
