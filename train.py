import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torchtrainer
from dataset import create_datasets

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_one_epoch(model, data_loader_train, criterion, optimizer, device, non_blocking, use_amp, scaler, batches_per_epoch):

    model.train()
    train_loss = 0.
    for batch_idx, (image, target) in enumerate(data_loader_train):
        image = image.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss = loss.item()
        train_loss += loss*image.shape[0]
        print(f'Batch {batch_idx+1}/{batches_per_epoch}, Train loss: {loss}')

    return train_loss

def validate(model, data_loader_valid, criterion, device, ds_size):

    model.eval()
    with torch.no_grad():
        valid_stats = torch.zeros(4)
        for batch_idx, (image, target) in enumerate(data_loader_valid):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss_valid = criterion(output, target).item()  
            acc = torchtrainer.perf_funcs.segmentation_accuracy(output, target)

            # Last batch might have different dimension, we multiply by batch size to calculate the average correctly
            stats = torch.tensor([loss_valid, acc['iou'], acc['prec'], acc['rec']])*image.shape[0]
            valid_stats += stats
        valid_stats /= ds_size

    return valid_stats
        
def train(model, ds_train, ds_valid, loss, class_weights, epochs, lr, batch_size, momentum=0.9, weight_decay=0., device='cuda', num_workers=0, 
          use_amp=False, pin_memory=False, non_blocking=False, resume=False, checkpoint_file='checkpoint.pth', save_best=True, meta=None):

    start_epoch = 1
    # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
        worker_init_fn=seed_worker
    )
    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False
    )
 
    # Define loss function
    if loss=='cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device)) 
    elif loss=='label_weighted_cross_entropy':
        criterion = torchtrainer.perf_funcs.LabelWeightedCrossEntropyLoss()

    # Create optimizer and logger
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger = torchtrainer.util.Logger(['Train loss', 'Valid loss', 'IoU', 'Prec', 'Rec', 'Time'])
    if resume:
        # Resume training
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        logger.load_state_dict(checkpoint["logger"])
        if use_amp:
            scaler.load_state_dict(checkpoint["scaler"])

    batches_per_epoch = len(ds_train)//batch_size + 1*(len(ds_train)%batch_size>0)
    best_valid_loss = torch.inf
    for epoch in range(start_epoch, start_epoch+epochs):
        print(f'Epoch {epoch}/{start_epoch+epochs-1}')
        time_start = time.time()

        loss = train_one_epoch(model, data_loader_train, criterion, optimizer, device, non_blocking, use_amp, scaler, batches_per_epoch)
        valid_stats = validate(model, data_loader_valid, criterion, device, len(ds_valid))

        time_epoch = time.time() - time_start
        print(f'Epoch finished in {time_epoch:.1f} seconds')
        print(f'Valid loss: {valid_stats[0]}, IoU: {valid_stats[1]}, Prec: {valid_stats[2]}, Rec {valid_stats[3]}')
        logger.add_data(epoch, [loss/len(ds_train)]+valid_stats.tolist())

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            #"lr_scheduler": lr_scheduler.state_dict(),
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

def run(params):
    
    # Set deterministic training if a seed is provided
    seed = params['seed']
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed) 
        np.random.seed(seed)

    ds_train, ds_valid, _ = create_datasets(params['img_dir'], params['label_dir'], params['crop_size'], params['train_val_split'], use_simple=not params['use_transforms'])
    model = torchtrainer.models.resnet_seg.ResNetSeg(params['model_layers'], params['model_channels'])

    logger = train(model, ds_train, ds_valid, params['loss'], params['class_weights'], params['epochs'], params['lr'], params['batch_size'], momentum=params['momentum'], 
                   weight_decay=params['weight_decay'], device=params['device'], num_workers=params['num_workers'], use_amp=params['use_amp'], 
                   pin_memory=params['pin_memory'], non_blocking=params['non_blocking'], resume=params['resume'], checkpoint_file=params['checkpoint_file'], 
                   save_best=params['save_best'], meta=params)
    
    return logger, ds_train, ds_valid, model
