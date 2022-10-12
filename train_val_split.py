
from os import listdir
from os.path import isfile, join
import numpy as np
import shutil

DATASET_PATH = '/cluser/work/igp_psr/ai4good/group-1b/data/'
paths = np.array([f for f in listdir(DATASET_PATH + 'train/') if isfile(join(DATASET_PATH + 'train/', f))])
idx = np.arange(len(paths))
np.random.shuffle(idx)
split = round(0.8*len(idx))

train_paths = [DATASET_PATH + '/train/train/' + file for file in paths[idx[:split]]]
val_paths = [DATASET_PATH + '/train/test/' + file for file in paths[idx[split:]]]

dst_path = np.concatenate((train_paths, val_paths))

for i, file in enumerate(paths):
    shutil.move(DATASET_PATH + '/train/'+file, dst_path[i])