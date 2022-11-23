from tarfile import TarInfo
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from datetime import datetime
import cartopy.crs as ccrs
from patchify import patchify
import os
import shutil
from PIL import Image
import random

DATA_DIR = 'data/'

# Get Patches
def get_patches(data, shape, stride, vars):
    for i, image in enumerate(tqdm(data)):
        for j, var in enumerate(vars):
            if j == 0:
                im = np.array(image[var])
            else:
                if var =='LABELS':
                    im = np.concatenate((im, np.expand_dims(np.array(image[var]),axis = 0)), axis = 0)
                else:
                    im = np.concatenate((im, np.array(image[var])), axis = 0)

        im = np.transpose(im, (1,2,0))
        patches = np.reshape(patchify(im, (shape,shape,len(vars)), stride), (-1,shape,shape, len(vars)))
    return patches

def save_data(patches, patchsize, setname, file_paths, file_names):

    for i, patch in enumerate(patches):
        classes = np.unique(patch[:,:,-1])
        nr_classes = len(classes)

        if nr_classes == 1:
            if not os.path.exists(os.path.join(DATA_DIR+setname+'background')):
                print('Creating folder for background')
                os.makedirs(os.path.join(DATA_DIR+setname+'background'))
            data = xr.DataArray(patch)
            data.to_netcdf(os.path.join(DATA_DIR+setname+'background/'+str(i)+file_names[i//patchsize]))
        elif nr_classes == 2:
            if 1 in classes:
                if not os.path.exists(os.path.join(DATA_DIR+setname+'single_ac')):
                    print('Creating folder for single_ac')
                    os.makedirs(os.path.join(DATA_DIR+setname+'single_ac'))
                data = xr.DataArray(patch)
                data.to_netcdf(os.path.join(DATA_DIR+setname+'single_ac/'+str(i)+file_names[i//patchsize]))
            else:
                if not os.path.exists(os.path.join(DATA_DIR+setname+'single_tc')):
                    print('Creating folder for single_tc')
                    os.makedirs(os.path.join(DATA_DIR+setname+'single_tc'))
                data = xr.DataArray(patch)
                data.to_netcdf(os.path.join(DATA_DIR+setname+'single_tc/'+str(i)+file_names[i//patchsize]))
        else:
            if not os.path.exists(os.path.join(DATA_DIR+setname+'mixed_events')):
                    print('Creating folder for mixed_events')
                    os.makedirs(os.path.join(DATA_DIR+setname+'mixed_events'))
            data = xr.DataArray(patch)
            data.to_netcdf(os.path.join(DATA_DIR+setname+'mixed_events/'+str(i)+file_names[i//patchsize]))

# data pipeline
train_path = ['data/train/train/'+f for f in listdir('data/train/train/')[60:100] if isfile(join('data/train/train/', f))]
#val_path = ['data/train/test/'+f for f in listdir('data/train/test/') if isfile(join('data/train/test/', f))]
#test_path = ['data/test/'+f for f in listdir('data/test/') if isfile(join('data/test/', f))]

train_names = [f for f in listdir('data/train/train/')[60:100] if isfile(join('data/train/train/', f))]
#val_names = [f for f in listdir('data/train/test/') if isfile(join('data/train/test/', f))]
#test_names = [f for f in listdir('data/test/') if isfile(join('data/test/', f))]

train = [xr.load_dataset(file_path) for file_path in tqdm(train_path)]
#test = [xr.load_dataset(file_path) for file_path in tqdm(test_path)]
#val = [xr.load_dataset(file_path) for file_path in tqdm(val_path)]


train_patches = get_patches(train, 254, 50, vars = ['TMQ','U850','V850','PRECT','LABELS'])

print(train_patches[0].shape)
save_data(train_patches, 254, 'train/train/', train_path, train_names)


brd_path = ['data/train/train/mixed_events/'+f for f in listdir('data/train/train/mixed_events/') if isfile(join('data/train/train/mixed_events/', f))]
bgr = [xr.load_dataset(file_path) for file_path in tqdm(brd_path)]


for i, nr in enumerate(np.random.randint(0, len(bgr), 5, int)):
    for j in range(5):
        data = np.squeeze(np.asarray(bgr[nr].to_array()[:,:,:,j]))
        im = Image.fromarray(data)
        plt.imsave(str(i)+'_'+str(j)+'.jpeg',im, cmap='YlGnBu')
        #plt.savefig(str(i)+'_'+str(j)+'.jpeg', format='jpg', bbox_inches='tight')
      

