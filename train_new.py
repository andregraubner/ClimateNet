from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet, CGNetModule
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass

from os import path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config
import pandas as pd
import torch
import numpy as np
import torchvision

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import glob
from dask import delayed

from dask.utils import SerializableLock
from dask.diagnostics import ProgressBar

# Create a SerializableLock
lock = SerializableLock()

root_path = "/net/pf-pc69/"

class ClimateDatasetNew(Dataset):
  
    def __init__(self, data_path: str, split="train"):
        self.path: str = data_path
        
        self.label_files_ar: [str] = [f for f in sorted(listdir(self.path+"/AR/")) if f[-3:] == ".nc"]
        self.label_files_tc: [str] = [f for f in sorted(listdir(self.path+"/TC/")) if f[-3:] == ".nc"]

        self.data = xr.open_dataset(f"{root_path}/scratch/lukaska/data/chunks/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_02.nc")#, chunks={'time': '8GB'})

        # Normalisation statistics
        self.mean = self.data.isel(time=slice(0,100)).mean(dim=['time', 'latitude', 'longitude'])
        self.std = self.data.isel(time=slice(0,100)).std(dim=['time', 'latitude', 'longitude'])

        self.labels = []
        self.length = 0
        # Note that this [198:] is suuuper hacky... 
        # just because can't load first half of data. 
        # This seems to work, but might be file system dependent

        #files_class1 = glob.glob(path.join(self.path, "AR",  "*.nc"))[198:]
        #files_class2 = glob.glob(path.join(self.path, "TC",  "*.nc"))[198:]

        files_class1 = [path.join(self.path, "AR",  f) for f in sorted(listdir(self.path+"/AR/")) if f[-3:] == ".nc"]
        files_class2 = [path.join(self.path, "TC",  f) for f in sorted(listdir(self.path+"/TC/")) if f[-3:] == ".nc"]

        datasets_class1 = [xr.open_dataset(f, chunks={'ts': 10}) for f in files_class1]
        datasets_class2 = [xr.open_dataset(f, chunks={'ts': 10}) for f in files_class2]

        # Concatenate the datasets for each class
        ds_class1 = xr.concat(datasets_class1, dim='ts').sortby("ts")
        ds_class2 = xr.concat(datasets_class2, dim='ts').sortby("ts")

        instances = ('ts', np.tile([0, 1], len(ds_class1.ts) // 2))

        ds_class1['instance'] = instances
        ds_class1 = ds_class1.set_index(time=['ts', 'instance'])
        ds_class1 = ds_class1.unstack('time')     

        instances = ('ts', np.tile([0, 1], len(ds_class2.ts) // 2))

        ds_class2['instance'] = instances
        ds_class2 = ds_class2.set_index(time=['ts', 'instance'])   
        ds_class2 = ds_class2.unstack('time')     

        ds = xr.concat([ds_class1, ds_class2], dim="class", join="inner")
        ds = ds.rename({"ts": "time"})

        ds = ds.transpose("time", "class", "instance", "latitude", "longitude")

        # remove time steps that are not in the features
        mask = ds['time'].isin(self.data['time'])  
        # Use the mask to drop the elements in ds1 that are not present in ds2
        ds = ds.where(mask, drop=True)

        #if split == "train":
        #    ds = ds.sel(time=slice(None, '2015-01-01'))
        #elif split == "test":
        #    ds = ds.sel(time=slice('2015-01-01', None)) 

        ds = ds.drop_vars('instance')

        ds["label"] = ds["label"].astype(bool)
        print(ds)
        ds = ds.isel(time=slice(None, 1000))
        ProgressBar().register()
        ds = ds.compute()
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path.join(self.path, "labels.nc"), encoding=encoding)
        print("done!")
        self.length = len(self.labels.time) # TODO: Make sure this is correct when changing instances

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        time_index = idx // len(self.labels.instance)
        instance_index = 0 # TODO: Add instance randomness

        labels = self.labels.isel(time=time_index, instance=instance_index) 

        # Get appropriate data
        timestep = labels.coords['time'].values
        #timestep = pd.to_datetime(str(timestep)).to_pydatetime()

        features = self.data.sel(time=timestep)

        # Normalize data
        features -= self.mean
        features /= self.std

        # TODO: Why are some values nan?
        features = features.fillna(self.mean)

        features = features.to_array().values

        labels = labels.to_array().values[0]

        labels = torch.tensor(labels.astype(bool)).long()
        mask = (labels.sum(0) == 0)[None]
        labels = torch.cat([mask, labels], dim=0)
        
        return torch.tensor(features), labels

        try:
            time_index = idx // len(self.labels.instance)
            instance_index = 0 # TODO: Add instance randomness

            labels = self.labels.isel(time=time_index, instance=instance_index) 

            # Get appropriate data
            timestep = labels.coords['time'].values
            #timestep = pd.to_datetime(str(timestep)).to_pydatetime()

            features = self.data.sel(time=timestep)

            # Normalize data
            features -= self.mean
            features /= self.std

            # TODO: Why are some values nan?
            features = features.fillna(self.mean)

            features = features.to_array().values

            labels = labels.to_array().values[0]
            
            return torch.tensor(features), torch.tensor(labels.astype(bool)).long()

        except:
            print("failed")
            random_index = torch.randint(len(self), size=(1,)).item()
            return self[random_index]


    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return torch.stack(data), torch.stack(labels)

class ClimateDatasetNew2(Dataset):
  
    def __init__(self, data_path: str, split="train"):
        self.path: str = data_path

        self.data = xr.open_dataset(f"{root_path}/scratch/lukaska/data/chunks/chunk_random_1980_1_1_00:00-2023_1_1_00:00_samples_5000_02.nc")#, chunks={'time': '8GB'})

        # Normalisation statistics
        self.mean = self.data.isel(time=slice(0,200)).mean(dim=['time', 'latitude', 'longitude'])
        self.std = self.data.isel(time=slice(0,200)).std(dim=['time', 'latitude', 'longitude'])

        self.labels = xr.load_dataset(f"{root_path}/scratch/andregr/dataset/labels.nc") 

        if split == "train":
            self.labels = self.labels.sel(time=slice(None, '1987-01-01'))
        elif split == "test":
            self.labels = self.labels.sel(time=slice('1986-01-01', None)) 

        self.n_timesteps = len(self.labels.time)
        self.length = len(self.labels.time) * 2 # TODO: Make sure this is correct when changing instances

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        time_index = idx // 2 # TODO: Make sure this is correct
        instance_index = idx % 2 # TODO: Add instance randomness

        labels = self.labels.isel(time=time_index, instance=instance_index) 

        # Get appropriate data
        timestep = labels.coords['time'].values
        features = self.data.sel(time=timestep)

        # Normalize data
        features -= self.mean
        features /= self.std

        # TODO: Why are some values nan?
        features = features.fillna(self.mean)

        features = features.to_array().values
        labels = labels.to_array().values[0]

        labels = torch.tensor(labels.astype(bool)).long()
        mask = (labels.sum(0) == 0)[None]
        labels = torch.cat([mask, labels], dim=0)
        
        return torch.tensor(features), labels


    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return torch.stack(data), torch.stack(labels)

train = ClimateDatasetNew2(f"{root_path}/scratch/andregr/dataset", split="train")
test = ClimateDatasetNew2(f"{root_path}/scratch/andregr/dataset", split="test")

train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=8, prefetch_factor=3)
test_loader = DataLoader(test, batch_size=16, shuffle=True, num_workers=8, prefetch_factor=3)

model = CGNetModule(classes=3, dropout_flag=True)
model = torch.nn.DataParallel(model).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def validate(model):
    model.eval()

    ious = []
    for it, (features, labels) in enumerate(tqdm(test_loader)): 
        features = features.to("cuda")
        labels = labels.to("cuda")

        with torch.no_grad():
            preds = model(features)
            cm = get_cm(preds, labels, n_classes=3)
            ious.append(get_iou_perClass(cm))

    labels = labels.detach().cpu()
    probas = F.softmax(preds, dim=1).detach().cpu()
    plt.imsave(f'new_preds_ar.png', probas[0,1], cmap='viridis', vmin=0, vmax=1)
    plt.imsave(f'new_preds_tc.png', probas[0,2], cmap='viridis', vmin=0, vmax=1)
    plt.imsave(f'new_gt_ar.png', labels[0,1], cmap='viridis', vmin=0, vmax=1)
    plt.imsave(f'new_gt_tc.png', labels[0,2], cmap='viridis', vmin=0, vmax=1)

    ios = np.array(ious)
    print(np.mean(ious, axis=0))

for epoch in range(50):
    validate(model)      
    model.train()

    for it, (features, labels) in enumerate(tqdm(train_loader)):
        features = features.to("cuda")
        labels = labels.to("cuda")

        preds = model(features)
        
        loss = jaccard_loss(preds, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())