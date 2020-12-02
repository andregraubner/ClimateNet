from os import listdir, path
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor, float32

from netCDF4 import Dataset as ncDataset

class ClimateDataset(Dataset):
  
    def __init__(self, path: str, fields: [str], dowload: bool = False):
        self.path: str = path

        # TODO: DOWNLOAD IF download IS SET AND DOES NOT EXIST AT THIS LOCATION YET

        self.fields: [str] = fields
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)

    def __getitem__(self, idx: int) -> tensor:
      
        file_path: str = path.join(self.path, self.files[idx]) 

        ds = ncDataset(file_path, "r", format="NETCDF4")
        print(ds.variables)
        variables_array = np.array([ds.variables[var] for var in self.fields])
        ar = np.swapaxes(variables_array, 0, 1) 
        print(ar.shape)
        sample = tensor(ar, dtype=float32)
                
        return sample
      
    def __len__(self):
        return self.length
