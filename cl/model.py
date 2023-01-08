
#utils
import configparser
import shutil
import os
from decouple import config

#data handling
import numpy as np
import xarray as xr

#model
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader, Dataset

import torchgeo
from torchgeo.trainers import SemanticSegmentationTask

import pytorch_lightning as pl
from pytorch_lightning import Trainer,LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

#logging
import wandb
from torchmetrics import ClasswiseWrapper,JaccardIndex,MetricCollection


#set mode for training, default base (whole image), choose from base, patch, cl
mode = 'base'


#read in config file
config = configparser.ConfigParser()
config.read('config.yaml')


#set path to directories from config
DATA_PATH = config['path']['data_path']
LOG_PATH = config["path"]["log_path"]
REPO_PATH = config["path"]["repo_path"]

#extract lists from config
var_list = config["cl"]["var_list"].split(',')
epoch_lengths = config["trainer"]["max_epochs"].split(',')
epoch_lengths = np.cumsum(np.array([int(i) for i in epoch_lengths]))

#read patch size and round to multiple of 32 (unet requires divisible by 32)
patch_size = int(config['cl']['patch_size'])
if patch_size % 32 != 0:
        patch_size += 32 - patch_size % 32 


# base training on full map
if config['cl']['mode'] == 'base':
    DATA_PATH = f'{DATA_PATH}'

# training on patches
elif config['cl']['mode'] == 'patch':
    DATA_PATH = f'{DATA_PATH}{patch_size}/'

    #only extract data if requested
    if config['cl']['extract'] == 'True':
        from utils import cl_prep
        cl_prep.process_all_images(patch_size= patch_size, 
                                    stride = int(config['cl']['stride']), vars = var_list, 
                                    max_exp_patches = int(config['cl']['max_nr_patches']), mode = 'True')



# cl
elif config['cl']['mode'] == 'cl':
    mode = 'cl'

    #only if new curriculum or different patch size
    if config['cl']['extract'] == 'True':

        #clear folder for new curriculum
        print('Clear folder for new Curriculum')
        shutil.rmtree(f'{DATA_PATH}cl/{patch_size}/')
        print('Folder emptied.')

        #extract data and create training stages
        from utils import cl_prep
        cl_prep.process_all_images(patch_size= patch_size, 
                                    stride = int(config['cl']['stride']), vars = var_list, 
                                    max_exp_patches = int(config['cl']['max_nr_patches']), mode = 'False')



# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, split, path, stage):

        self.data_path = path #path to data
        self.var_list = var_list #channels to train on
        self.split = split #train, test, validation
        self.stage = stage
        
        if mode != 'cl':
            self.file_names = os.listdir(f'{self.data_path}{self.split}')
        
        else: #in cl, extraction is different in train/val (subset of patches) compared to test(all patches)
            if self.split == 'test':
                self.file_names = os.listdir(f'{self.data_path}{self.split}/')
            else:
                self.file_names = os.listdir(f'{self.data_path}{self.split}/stage_{self.stage}')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]

        try:  
            if mode != 'cl':  
                data = xr.load_dataset(f'{self.data_path}{self.split}/{img_name}')
            
            else: #if cl, dataset depends on current stage during training
                if self.split == 'test':
                    data = xr.load_dataset(f'{self.data_path}{self.split}/{img_name}')
                else:
                    data = xr.load_dataset(f'{self.data_path}{self.split}/stage_{self.stage}/{img_name}')
            
            
            image = np.concatenate([np.array(data[var]) for var in self.var_list]).astype(np.float32)
            mask = np.array(data['LABELS']).astype(np.uint8)
            
        except:
            print(f'skipped image {img_name}')
            return None

        return {"image": image, "mask": mask}


#filter Nones from dataset creation
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

'''
class Scheduler(pl.Callback):
    def _prepare_epoch(self, trainer, model, epoch):
        trainer.datamodule.set_phase(epoch)

    def on_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)
'''

#create train, val and test datasets according to stage from Imagedataset
class Data(LightningDataModule):
    
    def __init__(self,  stage = 1):
        super().__init__()
        if mode != 'cl':
            self.path = DATA_PATH
        else:
            self.path = f'{DATA_PATH}cl/{patch_size}/'
        self.stage = stage
      
    def train_dataloader(self):

        split = "train"
        train_data = ImageDataset(split, self.path, self.stage)
        
        train_dataloader = DataLoader(
            train_data,
            batch_size=int(config["datamodule"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
            drop_last=True
        )
        return train_dataloader
    
    def val_dataloader(self):    

        split = "val"
        val_data = ImageDataset(split,self.path, self.stage)
        val_dataloader = DataLoader(
            val_data,
            batch_size=int(config["datamodule"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
            drop_last=True
        )
        return val_dataloader

    def test_dataloader(self):

        split = "test"
        test_data = ImageDataset(split, self.path, self.stage)
        
        test_dataloader = DataLoader(
            test_data,
            batch_size=int(config["datamodule"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["datamodule"]["num_workers"]),
            collate_fn=collate_fn,
            drop_last=True
        )
        return test_dataloader

# plotting function for logging to wandb
def gen_mask_plot(mask):

        #color choices
        r_val = [205,46,100]
        g_val = [92,139,149]
        b_val = [92,87,237]

        r = np.zeros(mask.shape)
        g = np.zeros(mask.shape)
        b = np.zeros(mask.shape)

        for i, r_i, g_i, b_i in zip([2,1,0], r_val, g_val, b_val):
            r += np.where(mask == i, r_i, 0).astype(np.uint8)
            g += np.where(mask == i, g_i, 0).astype(np.uint8)
            b += np.where(mask == i, b_i, 0).astype(np.uint8)

        return np.array([r.astype(np.uint8),g.astype(np.uint8),b.astype(np.uint8)])


# define metrics for logging the performance
class Model_Task(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class_labels = ["BG", "TC",'AR']

        #classwise logging and mean logging of jaccard index
        metrics = MetricCollection({"jaccard_index": ClasswiseWrapper(JaccardIndex(num_classes=self.hyperparams["num_classes"],
                                                                                  ignore_index=self.ignore_index,
                                                                                  mdmc_average="global", average="none"),labels=class_labels),
                                    'mean jaccard_index': JaccardIndex(num_classes=self.hyperparams["num_classes"],
                                                                       ignore_index=self.ignore_index)})
        self.train_metrics = metrics.clone(prefix = 'train_')
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    #overwrite standard validation step to apply custom metrics and log maps to wandb
    def validation_step(self, batch, batch_idx):
        
        x, y = batch['image'], batch['mask']
        x = x.type(torch.float32)
        y_hat = self.forward(x)
        y_hat_int = y_hat.argmax(dim=1) #prediction
        

        #log loss and metrics
        loss = self.loss(y_hat, y) 
        self.log("val_loss", loss)
        self.val_metrics(y_hat_int, y)

        y_numpy = y.cpu().numpy()
        y_hat_int_numpy = y_hat_int.cpu().numpy()
        
        
        #plot every 10th example
        if batch_idx in np.arange(1,100,10):
            
            
            gt_mask = y_numpy.astype(np.uint8)[0] #original mask
            gt_masks = gen_mask_plot(gt_mask) #plotting conversion
            gt_masks = np.transpose(gt_masks,(1,2,0))

            pred_mask = y_hat_int_numpy.astype(np.uint8)[0] #predicted mask
            pred_masks = gen_mask_plot(pred_mask)
            pred_masks = np.transpose(pred_masks,(1,2,0))

            overlay =  0.4 * gt_masks + 0.6 * pred_masks

            #log process
            wandb.log({"True Ground": wandb.Image(gt_masks),
                       "Prediction": wandb.Image(pred_masks),
                       "Overlay": wandb.Image(overlay)}) 


    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        x = x.type(torch.float32)
        y_hat = self.forward(x)
        y_hat_int = y_hat.argmax(dim=1)
        

        loss = self.loss(y_hat, y) 
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_int, y)

        y_numpy = y.cpu().numpy()
        y_hat_int_numpy = y_hat_int.cpu().numpy()

        gt_mask = y_numpy.astype(np.uint8)[0] #ground truth
        gt_masks = gen_mask_plot(gt_mask) 
        gt_masks = np.transpose(gt_masks,(1,2,0))


        pred_mask = y_hat_int_numpy.astype(np.uint8)[0] #prediction
        pred_masks = gen_mask_plot(pred_mask)
        pred_masks = np.transpose(pred_masks,(1,2,0))

        overlay =  0.4 * gt_masks + 0.6 * pred_masks

        wandb.log({"True Ground Test": wandb.Image(gt_masks),
                    "Prediction Test": wandb.Image(pred_masks),
                    "Overlay Test": wandb.Image(overlay)})     
            

    #update metric values
    def training_epoch_end(self, outputs):
        
        metric_values = self.train_metrics.compute()
        keys = set(list(metric_values))
        updated_metrics = {key: metric_values[key] for key in keys}
        
        self.log_dict(updated_metrics)
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs):
       
        metric_values = self.train_metrics.compute()
        keys = set(list(metric_values))
        updated_metrics = {key: metric_values[key] for key in keys}
        
        self.log_dict(updated_metrics)
        self.train_metrics.reset()

    def test_epoch_end(self, outputs):
        
        metric_values = self.train_metrics.compute()
        keys = set(list(metric_values))
        updated_metrics = {key: metric_values[key] for key in keys}
        
        self.log_dict(updated_metrics)
        self.train_metrics.reset()


if __name__ == "__main__":
    
    #set up wandb logging
    wandb.init(entity=config['wandb']['entity'], project=config['wandb']['project'])
    
    #create logging dir
    log_spot = config["logging"]["log_nr"]
    log_dir = f'{LOG_PATH}{log_spot}/'

    if not os.path.exists(log_dir):
        print(f'Create {log_dir}')
        os.makedirs(log_dir)

    # checkpoints and loggers
    checkpoint_callback = ModelCheckpoint(
            monitor=None,#always take last checkpoint, not best for cl
            dirpath=log_dir + "/checkpoints",
            save_top_k=1,
            save_last=True,
    )

    
    wandb_logger = WandbLogger(entity=config['wandb']['entity'], log_model=True, project=config['wandb']['project'])

    #vanilla training
    if mode == 'base':
        early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
        data_module = Data()
        
        # set up task and insert hyperparameters from config file
        task = Model_Task(
        segmentation_model=config["model"]["segmentation_model"],
        encoder_name=config["model"]["backbone"],
        encoder_weights="imagenet" if config["model"]["pretrained"] == "True" else "None",
        in_channels=len(var_list),
        num_classes=int(config["model"]["num_classes"]),
        loss=config["model"]["loss"],
        ignore_index=None,
        learning_rate=float(config["model"]["learning_rate"]),
        learning_rate_schedule_patience=int(
            config["model"]["learning_rate_schedule_patience"]),
        )

        #generate Trainer and fit on data
        trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        accelerator="gpu",
        max_epochs=int(epoch_lengths[0]),
        max_time=config["trainer"]["max_time"],
        auto_lr_find=config["trainer"]["auto_lr_find"] == "True",
        auto_scale_batch_size=config["trainer"]["auto_scale_batch_size"] == "True",
        )

        trainer.fit(task, datamodule=data_module)
        trainer.test(model=task, datamodule = data_module)
        wandb.finish()
    
    #cl learning: loop over stages and read and write from same checkpoint store
    else: 

        nr_stages = int(config["cl"]["nr_stages"])

        for i in range(nr_stages):
            print(f'Starting training round {i}')
            data_module = Data(stage = i+1)
            stage_nr = i+1

    
            # set up task
            task = Model_Task(
            segmentation_model=config["model"]["segmentation_model"],
            encoder_name=config["model"]["backbone"],
            encoder_weights="imagenet" if config["model"]["pretrained"] == "True" else "None",
            in_channels=len(var_list),
            num_classes=int(config["model"]["num_classes"]),
            loss=config["model"]["loss"],
            ignore_index=None,
            learning_rate=float(config["model"]["learning_rate"]),
            learning_rate_schedule_patience=int(
                config["model"]["learning_rate_schedule_patience"]),
            )

            if i == 0: #first round, no checkpoints available yet
                trainer = Trainer(
                callbacks=[checkpoint_callback],
                logger= wandb_logger,
                accelerator="gpu",
                max_epochs=int(epoch_lengths[i]),
                max_time=config["trainer"]["max_time"],
                auto_lr_find=config["trainer"]["auto_lr_find"] == "True",
                auto_scale_batch_size=config["trainer"]["auto_scale_batch_size"] == "True",
                )
            
                trainer.fit(task, datamodule=data_module)


            else: #from second round on read from checkpoint store

                checkpoints = os.listdir(f'{log_dir}checkpoints')
                checkpoint = checkpoints[-1]
                trainer = Trainer(
                callbacks=[checkpoint_callback],
                logger= wandb_logger,
                accelerator="gpu",
                max_epochs=int(epoch_lengths[i]),
                max_time=config["trainer"]["max_time"],
                auto_lr_find=config["trainer"]["auto_lr_find"] == "True",
                auto_scale_batch_size=config["trainer"]["auto_scale_batch_size"] == "True",
                )
                trainer.fit(task, datamodule=data_module, ckpt_path = f'{LOG_PATH}{log_spot}/checkpoints/{checkpoint}')


        #test model and finish wandb
        trainer.test(model=task, datamodule = data_module)
        wandb.finish()


    
