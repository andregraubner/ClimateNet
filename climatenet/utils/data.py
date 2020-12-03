from os import listdir, path
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor
import xarray as xr

from netCDF4 import Dataset as ncDataset

class ClimateDataset(Dataset):
  
    def __init__(self, path: str, config: dict):
        self.path: str = path
        self.fields: [str] = config['fields']
        
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)
      
    def __len__(self):
        return self.length

    def normalize(self, features: np.array):

        means = [self.fields[k]['mean'] for k in self.fields]
        stds = [self.fields[k]['std'] for k in self.fields]

        if len(features.shape) == 4:
            # Assume Batch, Variable, Latitude, Longitude
            for i in range(len(self.fields)):
                features[:,i,:,:] -= means[i]
                features[:,i,:,:] /= stds[i]

        elif len(features.shape) == 3:
            # Assume Variable, Latitude, Longitude
            for i in range(len(self.fields)):
                features[i,:,:] -= means[i]
                features[i,:,:] /= stds[i]

        else:
            #TODO: Something likely went wrong here.
            pass 
        

    def get_features(self, file_path: str) -> np.array:
        data = xr.load_dataset(file_path)
        data = data[list(self.fields)].to_array()
        
        data = data.transpose('time', 'variable', 'lat', 'lon')
        data = np.array(data)
        self.normalize(data)

        return data

    def __getitem__(self, idx: int) -> tensor:
        file_path: str = path.join(self.path, self.files[idx]) 
        features = self.get_features(file_path)
        return tensor(features).float()


class ClimateDatasetLabeled(ClimateDataset):

    def get_features(self, file_path: str) -> np.array:
        data = xr.load_dataset(file_path, group='data')
        data = data[list(self.fields)].to_array()
        data = data.transpose('variable', 'lat', 'lon')
        data = np.array(data)
        self.normalize(data)
        return data

    def __getitem__(self, idx: int) -> tensor:
        file_path: str = path.join(self.path, self.files[idx]) 

        features = self.get_features(file_path)

        labels_dataset = xr.load_dataset(file_path, group='labels')
        ar_labels = np.array(labels_dataset['label_0_ar'])
        tc_labels = np.array(labels_dataset['label_0_tc'])
        labels = tc_labels + 2 * ar_labels
        labels[labels==3] = 2 #If both ar and tc

        labels = np.roll(labels, 576, 1)
                
        return tensor(features).float(), tensor(labels).long()