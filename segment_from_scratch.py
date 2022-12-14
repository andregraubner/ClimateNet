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
    ClasswiseWrapper,
    ConfusionMatrix,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)
from torchvision.utils import draw_segmentation_masks

import xarray as xr



DATA_DIR = config("DATA_DIR_A4G")





LOG_DIR = config("LOG_DIR_A4G")
REPO_DIR = config("REPO_DIR_A4G")

bg_im = np.array(Image.open(f'{REPO_DIR}climatenet/bluemarble/BM.jpeg').resize((1152,768)))
#bg_im = np.transpose(bg_im, (2, 0, 1))
#print(bg_im.shape)
class_labels = {0: "BG", 1: "TC",  2: "AR"} 

phase_length = 1
nr_phases = 5
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

var_list = conf["cl"]["var_list"].split(',')
event_list = conf['cl']['events'].split(',')

if conf['cl']['extract'] == 'True':
    from utils import cl_prep_2
    cl_prep_2.process_all_images(patch_size= int(conf['cl']['patch_size']), 
                                 stride = int(conf['cl']['stride']), vars = var_list, 
                                 max_exp_patches = int(conf['cl']['max_nr_patches']),
                                 folder_names = event_list)

patch_size = int(conf['cl']['patch_size'])

if patch_size % 32 != 0:
        patch_size += 32 - patch_size % 32

DATA_DIR_CL = f'{DATA_DIR}cl/{patch_size}/'


# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, path, transform=None, target_transform=None):

        # Define the  mask file and the json file for retrieving images
        self.data_dir = path
        self.var_list = var_list
        self.setname = setname
        assert self.setname in ["train", "test", "val"]
        
        self.file_names = os.listdir(f'{self.data_dir}{self.setname}')
        if self.setname =='train':
            self.file_names = os.listdir(f'{self.data_dir}{self.setname}')

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]

        #try:    
        data = xr.load_dataset(f'{self.data_dir}{self.setname}/{img_name}')
        #local = np.full(data[self.var_list[0]].shape, float(img_name[-4]))
        
        image = np.concatenate([np.array(data[var]) for var in self.var_list]).astype(np.float32)
        #image = np.concatenate([image, local]).astype(np.float32)
        mask = np.array(data['LABELS']).astype(np.uint8)
        
        #except:
        #    print(f'skipped image {img_name}')
        #    return None

        

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
        trainer.datamodule.set_phase(epoch)

    def on_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)

class Data(LightningDataModule):

    def train_dataloader(self):

        setname = "train"
        train_data = ImageDataset(setname, DATA_DIR)
        
        train_dataloader = DataLoader(
            train_data,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=True,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
            drop_last=True
        )
        return train_dataloader
    
    def val_dataloader(self):    

        setname = "val"
        val_data = ImageDataset(setname,DATA_DIR_CL)
        val_dataloader = DataLoader(
            val_data,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=False,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
            drop_last=True
        )
        return val_dataloader

    def test_dataloader(self):

        setname = "test"
        test_data = ImageDataset(setname, DATA_DIR_CL)
        
        test_dataloader = DataLoader(
            test_data,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=False,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
            drop_last=True
        )
        return test_dataloader


class Model_Task(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class_labels = ["BG", "TC",'AR']

        self.train_metrics = MetricCollection(
            metrics ={ 
                "accuracy": ClasswiseWrapper(
                    Accuracy(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                    labels=class_labels,
                ),
                'mean accuracy':
                Accuracy(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                    ),
                "jaccard_index": ClasswiseWrapper(
                    JaccardIndex(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                        mdmc_average="global",
                        average="none",
                    ),
                labels=class_labels,
                ),
                'mean jaccard_index':
                JaccardIndex(
                        num_classes=self.hyperparams["num_classes"],
                        ignore_index=self.ignore_index,
                    ),
            },
            prefix="train_",compute_groups=True
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")


    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        x = x.type(torch.float32)
        y_hat = self.forward(x)
        y_hat_int = y_hat.argmax(dim=1)
        

        loss = self.loss(y_hat, y) 
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_int, y)

        y_numpy = y.cpu().numpy()
        y_hat_int_numpy = y_hat_int.cpu().numpy()

        if batch_idx < 10:
      
            
            image = wandb.Image(bg_im.astype(np.uint8), masks={
            "predictions" : {
                "mask_data" : y_hat_int_numpy.astype(np.uint8)[0],
                "class_labels" : class_labels
            },
            "ground_truth" : {
                "mask_data" :y_numpy.astype(np.uint8)[0],
                "class_labels" : class_labels
            }
            })
            wandb.log({"img_with_masks" : image})
            
            

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
    data_module = Data()

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


    trainer.fit(task, datamodule=data_module)
    trainer.test(model=task, datamodule = data_module)
    wandb.finish()
