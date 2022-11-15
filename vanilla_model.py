


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
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask
import wandb
import plotly.express as px

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

from torchvision.utils import draw_segmentation_masks


DATA_DIR = config("DATA_DIR_A4G")
LOG_DIR = config("LOG_DIR_A4G")
REPO_DIR = config("REPO_DIR_A4G")

background_im = Image.open(f'{REPO_DIR}climatenet/bluemarble/BM.jpeg').resize((768,1152))

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
    def __init__(self, setname, transform=None, target_transform=None):

        # Define the  mask file and the json file for retrieving images
        self.data_dir = DATA_DIR
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
     

        mask = torch.Tensor(np.array(data['LABELS']).astype(np.longlong))

    

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return {"image": image, "mask": mask}


class DataModule(LightningDataModule):
    def __init__(self, conf, data_frac=1.0):
        super().__init__()
        self.conf = conf
        self.data_frac = data_frac

    def prepare_data(self) -> None:
        self.train_data, self.val_data, self.test_data = (
            ImageDataset("train"),
            ImageDataset("val"),
            ImageDataset("test"),
        )

    def train_dataloader(self):
        return get_dataloaders(self.conf, self.train_data, data_frac=self.data_frac)[0]

    def val_dataloader(self):
        return get_dataloaders(self.conf, self.val_data, data_frac=self.data_frac)[0]

    def test_dataloader(self):
        return get_dataloaders(self.conf, self.test_data, data_frac=self.data_frac)[0]


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



class SemanticSegmentationTask_metrics(SemanticSegmentationTask):
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
                Precision(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                Recall(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                F1Score(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    mdmc_average="global",
                ),
                ConfusionMatrix(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def log_image(self, image, key, caption=""):
        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def validation_step(self, *args, **kwargs) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["mask"]
        print(torch.unique(y))
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                images = {
                    "image": background_im,
                    "masked": draw_segmentation_masks(
                        background_im.type(torch.uint8),
                        batch["mask"][0].type(torch.bool),
                        alpha=0.5,
                        colors="red",
                    ),
                    "prediction": draw_segmentation_masks(
                        background_im.type(torch.uint8),
                        batch["prediction"][0].type(torch.bool),
                        alpha=0.5,
                        colors="red",
                    ),
                }
                resize = torchvision.transforms.Resize(512)
                image_grid = torchvision.utils.make_grid(
                    [resize(value.float()) for key, value in images.items()],
                    value_range=(0, 255),
                    normalize=True,
                )
                self.log_image(
                    image_grid,
                    key="val_examples (original/groud truth/prediction)",
                    caption="Sample validation images",
                )
                wandb.log(
                    {
                        "pr": wandb.plot.pr_curve(
                            torch.reshape(batch["mask"][0], (-1,)),
                            torch.reshape(
                                y_hat[0].cpu(), (-1, self.hyperparams["num_classes"])
                            ),
                            labels=None,
                            classes_to_plot=None,
                        )
                    }
                )
            except AttributeError:
                pass

    def test_step(self, *args, **kwargs) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

        try:
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            images = {
                    "image": background_im,
                    "masked": draw_segmentation_masks(
                        background_im.type(torch.uint8),
                        batch["mask"][0].type(torch.bool),
                        alpha=0.5,
                        colors="red",
                    ),
                    "prediction": draw_segmentation_masks(
                        background_im.type(torch.uint8),
                        batch["prediction"][0].type(torch.bool),
                        alpha=0.5,
                        colors="red",
                    ),
            }
            resize = torchvision.transforms.Resize(512)
            image_grid = torchvision.utils.make_grid(
                [resize(value.float()) for key, value in images.items()],
                value_range=(0, 255),
                normalize=True,
            )
            self.log_image(
                image_grid,
                key="test_examples (original/groud truth/prediction)",
                caption="Sample test images",
            )
        except AttributeError:
            pass

    def training_epoch_end(self, outputs):
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        computed = self.train_metrics.compute()
        conf_mat = computed["train_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["train_ConfusionMatrix"])
        }
        cm = px.imshow(conf_mat, text_auto=".2f")
        wandb.log({"train_confusion_matrix": cm})
        self.log_dict(new_metrics)
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs):
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        computed = self.val_metrics.compute()
        conf_mat = computed["val_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["val_ConfusionMatrix"])
        }
        cm = px.imshow(conf_mat, text_auto=".2f")
        wandb.log({"val_confusion_matrix": cm})
        self.log_dict(new_metrics)
        self.val_metrics.reset()

    def test_epoch_end(self, outputs):
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        computed = self.test_metrics.compute()
        conf_mat = computed["test_ConfusionMatrix"].cpu().numpy()
        conf_mat = (conf_mat / np.sum(conf_mat)) * 100
        new_metrics = {
            k: computed[k] for k in set(list(computed)) - set(["test_ConfusionMatrix"])
        }
        fig = px.imshow(conf_mat, text_auto=".2f")
        wandb.log({"test_confusion_matrix": fig})
        self.log_dict(new_metrics)
        self.test_metrics.reset()


def get_dataloaders(conf, *datasets, data_frac=1.0):
    if data_frac != 1.0:
        datasets = [
            torch.utils.data.Subset(
                dataset,
                np.random.choice(
                    len(dataset), int(len(dataset) * data_frac), replace=False
                ),
            )
            for dataset in datasets
        ]

    return [
        DataLoader(
            dataset,
            batch_size=int(conf["datamodule"]["batch_size"]),
            shuffle=True,
            num_workers=int(conf["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
        )
        for dataset in datasets
    ]


if __name__ == "__main__":


    wandb.init(config=conf, entity="ai4good", project="segment_from_scratch")
    data_module = DataModule(conf)

    # checkpoints and loggers
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=LOG_DIR + "/checkpoints",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10
    )
    wandb_logger = WandbLogger(project="segment_from_scratch", log_model=True)

   
    # set up task
    task = SemanticSegmentationTask_metrics(
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

    # trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        default_root_dir=LOG_DIR,
        accelerator="gpu",
        max_epochs=int(conf["trainer"]["max_epochs"]),
        max_time=conf["trainer"]["max_time"],
        auto_lr_find=conf["trainer"]["auto_lr_find"] == "True",
        auto_scale_batch_size=conf["trainer"]["auto_scale_batch_size"] == "True",
    )
    trainer.fit(task, datamodule=data_module)

    trainer.test(model=task, datamodule=data_module)

    wandb.finish()