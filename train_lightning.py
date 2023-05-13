import random
import numpy as np
import torch
from torch import optim, nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torchtrainer
from dataset import create_datasets

class LitSeg(pl.LightningModule):
    def __init__(self, model, loss, class_weights, lr, momentum, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        # Define loss function
        if loss=='cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=self.device)) 
        elif loss=='label_weighted_cross_entropy':
            criterion = torchtrainer.perf_funcs.LabelWeightedCrossEntropyLoss()

        self.criterion = criterion
        self.learnin_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)

        acc = torchtrainer.perf_funcs.segmentation_accuracy(output, y)

        self.log("val_loss", loss, prog_bar=True)       
        self.log_dict(acc)   
        self.log("hp_metric", loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learnin_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        return optimizer
    
    #def on_train_start(self):
    #    self.logger.log_hyperparams(self.hparams, {"val_loss": 0, "iou": 0})



def run(params):

    # Set deterministic training if a seed is provided
    seed = params['seed']
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    if params['use_amp']:
        precision = '16-mixed'
    else:
        precision = '32-true'
    if params['resume']:
        ckpt_path = 'last'
    else:
        ckpt_path = None
    if params['save_best']:
        checkpoint_train = ModelCheckpoint()
        checkpoint_loss = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", 
                                            filename="best_val_loss-{epoch:02d}-{val_loss:.2f}")
        callbacks = [checkpoint_train, checkpoint_loss]
    else:
        callbacks = None

    ds_train, ds_valid, _ = create_datasets(params['img_dir'], params['label_dir'], params['crop_size'], params['train_val_split'], use_simple=not params['use_transforms'])
    model = torchtrainer.models.resnet_seg.ResNetSeg(params['model_layers'], params['model_channels'])
    lit_model = LitSeg(model, params['loss'], params['class_weights'], params['lr'], params['momentum'], params['weight_decay'])
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory'],
        persistent_workers=params['num_workers']>0
    )

    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=params['batch_size'],
        shuffle=False
    )

    trainer = pl.Trainer(max_epochs=params['epochs'], callbacks=callbacks, precision=precision)
    trainer.fit(lit_model, data_loader_train, data_loader_valid, ckpt_path=ckpt_path)

    return ds_train, ds_valid, model, trainer

