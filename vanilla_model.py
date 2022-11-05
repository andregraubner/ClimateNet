


import argparse
import configparser
import json
import os
import sys
from ctypes import cast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchgeo

import yaml
from decouple import config
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask


import xarray as xr

wandb_logger = WandbLogger(entity="ai4good", log_model="all", project="segment_from_scratch")


DATA_DIR = config("DATA_DIR_A4G")
REPO_DIR = config("REPO_DIR_A4G")

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
print(var_list)
# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, transform=None, target_transform=None):

        # Define the  mask file and the json file for retrieving images
        self.data_dir = DATA_DIR
        self.setname = setname
        assert self.setname in ["train", "test", "val"]

        self.file_names = os.listdir(f'{self.data_dir}{self.setname}/')

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]

        try:
            image = xr.load_dataset(f'{self.data_dir}{self.setname}/{img_name}')
            image = np.concatenate([np.array(image[idx][var]) for var in var_list])
            mask = np.array(image[idx]['LABELS'])
        except:
            return None

    

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return {"image": image, "mask": mask}


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":

    # create datasets
    setname = "train"
    train_data = ImageDataset(setname)
    setname = "val"
    val_data = ImageDataset(setname)
    setname = "test"
    test_data = ImageDataset(setname)

    # DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=int(conf["datamodule"]["batch_size"]),
        shuffle=True,
        num_workers=int(conf["datamodule"]["num_workers"]),
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=int(conf["datamodule"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["datamodule"]["num_workers"]),
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=int(conf["datamodule"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["datamodule"]["num_workers"]),
        collate_fn=collate_fn,
    )

    # set up task
    task = SemanticSegmentationTask(
        logger=wandb_logger,
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
        accelerator="gpu",
        max_epochs=int(conf["trainer"]["max_epochs"]),
        max_time=conf["trainer"]["max_time"],
        logger=wandb_logger,
        auto_lr_find=conf["trainer"]["auto_lr_find"] == "True",
        auto_scale_batch_size=conf["trainer"]["auto_scale_batch_size"] == "True",
    )
    trainer.fit(task, train_dataloader, val_dataloader)

    trainer.test(model=task, dataloaders=test_dataloader, verbose=True)
