import argparse
import configparser
import json
import os
import sys
from ctypes import cast
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchgeo

import yaml
from decouple import config
from PIL import Image
from pytorch_lightning import Trainer,LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask
import pytorch_lightning as pl
import wandb
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)

import xarray as xr



DATA_DIR = config("DATA_DIR_A4G")
LOG_DIR = config("LOG_DIR_A4G")
REPO_DIR = config("REPO_DIR_A4G")

bg_im = np.array(Image.open(f'{REPO_DIR}climatenet/bluemarble/BM.jpeg').resize((768,1152)))

class_labels = {0: "BG", 1: "TC",  2: "AR"} 

phase_length = 5
nr_phases = 3
parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf",
    type=str,
    nargs="?",
    const=True,
    default="conf.yaml",
    help="Choose config file for setup",
)
args = parser.parse_args()

conf = configparser.ConfigParser()
conf.read(args.conf)

var_list = conf["experiment"]["var_list"].split(',')
# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, path, transform=None, target_transform=None):

        # Define the  mask file and the json file for retrieving images
        self.data_dir = path
        self.var_list = var_list
        self.setname = setname
        assert self.setname in ["train", "test", "val"]

        self.file_names = os.listdir(f'{self.data_dir}{self.setname}/')

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]

            
        data = xr.load_dataset(f'{self.data_dir}{self.setname}/{img_name}')
        image = np.concatenate([np.array(data[var]) for var in self.var_list])
        mask = np.array(data['LABELS'])

        

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return {"image": image, "mask": mask}


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Scheduler(pl.Callback):
    def _prepare_epoch(self, trainer, model, epoch):
        phase = {'phase': DATA_DIR} #TODO --> change dir based on phase by including current epoch
        trainer.datamodule.set_phase(phase)

    def on_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)

class Data(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.path = DATA_DIR
      
    def set_phase(self, phase: dict):
        self.path = phase.get("phase", self.path)
       
    def train_dataloader(self):

        setname = "train"
        train_data = ImageDataset(setname, self.path)
        
        train_dataloader = DataLoader(
            train_data,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=True,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
        )
        return train_dataloader
    
    def val_dataloader(self):    

        setname = "val"
        val_data = ImageDataset(setname,self.path)
        val_dataloader = DataLoader(
            val_data,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=False,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):

        setname = "test"
        test_data = ImageDataset(setname,self.path)
        
        test_dataloader = DataLoader(
            test_data,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=False,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
        )
        return test_dataloader


class Model_Task(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = MetricCollection(
            [
                Accuracy(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                JaccardIndex(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def log_image(image, key, caption=""):
        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)
        y_hat_int = y_hat.argmax(dim=1)
        

        loss = self.loss(y_hat, y) 
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        y_numpy = y.cpu().numpy()
        y_hat_int_numpy = y_hat_int.cpu().numpy()

        datamodule = self.trainer.datamodule

        for y_hat_i, y_i in zip(y_hat_int_numpy,y_numpy):

            image = wandb.Image(bg_im.astype(np.uint8), masks={
            "predictions" : {
                "mask_data" : y_hat_i.astype(np.uint8),
                "class_labels" : class_labels
            },
            "ground_truth" : {
                "mask_data" :y_i.astype(np.uint8),
                "class_labels" : class_labels
            }
            })
            wandb.log({"predictions" : image})
            #trainer.logger.experiment.log({'examples': image})
            #log_image(image, 'validation results', 'plot mask from validation')

    def training_epoch_end(self, outputs):
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        new_metrics = {
            k: computed[k] for k in set(list(computed))
        }
        self.log_dict(new_metrics)
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs):
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        new_metrics = {
            k: computed[k] for k in set(list(computed))}
        
        self.log_dict(new_metrics)
        self.val_metrics.reset()

if __name__ == "__main__":
    



    wandb.init(entity="ai4good", project="segment_from_scratch")
    log_dir = LOG_DIR + time.strftime("%Y%m%d-%H%M%S")

    # checkpoints and loggers
    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=log_dir + "/checkpoints",
            save_top_k=1,
            save_last=True,
    )
    early_stopping_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=10
    )
    csv_logger = CSVLogger(save_dir=log_dir, name="logs")

    wandb_logger = WandbLogger(entity="ai4good", log_model=True, project="segment_from_scratch")

    # set up task
    task = Model_Task(
        segmentation_model=conf["model"]["segmentation_model"],
        encoder_name=conf["model"]["backbone"],
        encoder_weights="imagenet" if conf["model"]["pretrained"] == "True" else "None",
        in_channels=len(var_list),
        num_classes=int(conf["model"]["num_classes"]),
        loss=conf["model"]["loss"],
        ignore_index=None,
        learning_rate=float(conf["model"]["learning_rate"]),
        learning_rate_schedule_patience=int(
            conf["model"]["learning_rate_schedule_patience"]
        ),
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[csv_logger, wandb_logger],
        accelerator="gpu",
        max_epochs=nr_phases*phase_length,
        max_time=conf["trainer"]["max_time"],
        auto_lr_find=conf["trainer"]["auto_lr_find"] == "True",
        auto_scale_batch_size=conf["trainer"]["auto_scale_batch_size"] == "True",
        reload_dataloaders_every_n_epochs=phase_length,
    )


    trainer.fit(task, datamodule=Data())

    trainer.test(model=task, datamodule = Data())