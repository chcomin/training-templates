from pathlib import Path
import torch
from torch import optim, nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from lightning.pytorch.utilities import rank_zero_only
import torchtrainer   #https://github.com/chcomin/torchtrainer
from dataset_readers import vessel_cortex

defaul_params = {
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
    'save_best':True,                   # Save model with best validation loss
    'meta': None,                       # Additional metadata to save
    # Other
    'resume': False,                    # Resume from previous training
}

def process_params(user_params):
    '''Use default value of a parameter in case it was not provided'''

    params = defaul_params.copy()
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
        # workers=True sets different seeds for each worker.
        pl.seed_everything(seed, workers=True)

    experiment_folder = Path(params['log_dir'])/str(params["experiment"])
    experiment_folder.mkdir(parents=True, exist_ok=True)

    return experiment_folder

class LitSeg(pl.LightningModule):
    def __init__(self, model_layers, model_channels, loss, class_weights, lr, momentum, weight_decay, iters, scheduler_power, 
                 model_type, meta):
        super().__init__()
        self.save_hyperparameters()  # Add __init__ parameters to checkpoint file

        # Define loss function
        if loss=='cross_entropy':
            loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=self.device)) 
        elif loss=='label_weighted_cross_entropy':
            loss_func = torchtrainer.perf_funcs.LabelWeightedCrossEntropyLoss()

        # Model
        if model_type=='unet':
            model = torchtrainer.models.resunet.ResUNet(model_layers, model_channels)
        elif model_type=='resnetseg':
            model = torchtrainer.models.resnet_seg.ResNetSeg(model_layers, model_channels)

        self.loss_func = loss_func
        self.learnin_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.iters = iters
        self.scheduler_power = scheduler_power
        self.model = model

    def forward(self, x):
        '''Defining this method allows using an instance of this class as litseg(x).'''
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)
        acc = torchtrainer.perf_funcs.segmentation_accuracy(output, y, ('iou', 'prec', 'rec'))

        # In case something goes wrong
        if torch.isnan(loss):
            checkpoint = {
                "model": self.model,
                'batch': batch,
                "batch_idx": batch_idx,
            }
            torch.save(checkpoint, 'error_ckp.pth')

        self.log("val_loss", loss, prog_bar=True)       
        self.log_dict(acc, prog_bar=True)   
        self.log("hp_metric", loss)   # Metric to show with hyperparameters in Tensorboard

    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr=self.learnin_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr=self.learnin_rate, betas=(self.momentum, 0.999), weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.iters, power=self.scheduler_power)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
        }

        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler_config}

class MyLogger(Logger):
    '''Simple class for logging performance metrics.'''
    
    def __init__(self):
        self.metrics = {}

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        return 1

    @rank_zero_only
    def log_hyperparams(self, params):
        self.params = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        saved_metrics = self.metrics
        for k, v, in metrics.items():
            if k in saved_metrics:
                saved_metrics[k].append((step,v))
            else:
                saved_metrics[k] = [(step,v)]

def run(user_params):

    params = process_params(user_params)
    experiment_folder = initial_setup()

    # Mixed precision
    if params['use_amp']:
        precision = '16-mixed'
    else:
        precision = '32-true'

    # Create dataset and dataloaders
    ds_train, ds_valid, _ = vessel_cortex.create_datasets(params['img_dir'], params['label_dir'], params['crop_size'], params['train_val_split'], use_simple=not params['use_transforms'])
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=params['batch_size_train'],
        shuffle=True,
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory'],
        persistent_workers=params['num_workers']>0   # Avoid recreating workers at each epoch
    )

    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=params['batch_size_valid'],
        shuffle=False,
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory'],
        persistent_workers=params['num_workers']>0   # Avoid recreating workers at each epoch
    )
    total_iters = len(data_loader_train)*params['epochs']   # For scheduler

    if params['resume']:
        # Resume previous experiment
        checkpoint_file = experiment_folder/'checkpoints/last.ckpt'
        seed = params['seed']
        lit_model = LitSeg.load_from_checkpoint(checkpoint_file) 
        start_epoch = lit_model.current_epoch + 1
        if seed is not None:
            # Seed using the current epoch to avoid using the same seed as in epoch 0 when resuming
            pl.seed_everything(seed+start_epoch, workers=True)
    else:
        checkpoint_file = None
        lit_model = LitSeg(params['model_layers'], params['model_channels'], params['loss'], params['class_weights'], params['lr'], params['momentum'], params['weight_decay'], 
                           total_iters, params['scheduler_power'], params['model_type'], params)
        start_epoch = 0

    callbacks = [LearningRateMonitor()]
    if params['save_best']:
        # Create callback for saving model with best validation loss
        checkpoint_loss = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", 
                                            filename="best_val_loss-{epoch:02d}-{val_loss:.2f}")
        callbacks.append(checkpoint_loss)
    # Callback for saving the model at the end of each epoch
    callbacks.append(ModelCheckpoint(save_last=True))

    logger_tb = TensorBoardLogger('.', name=params['log_dir'], version=params['experiment'])
    logger = MyLogger()

    trainer = pl.Trainer(max_epochs=start_epoch+params['epochs'], callbacks=callbacks, precision=precision, logger=[logger_tb, logger])
    trainer.fit(lit_model, data_loader_train, data_loader_valid, ckpt_path=checkpoint_file)

    return ds_train, ds_valid, lit_model, trainer
