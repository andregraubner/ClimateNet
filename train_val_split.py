
from os import listdir, mkdir
from os.path import isfile, join
import numpy as np
import shutil
from pathlib import Path

IS_DATASET_SPLIT = True # True if dataset has already been split, false otherwise

# CAREFUL : this script was already run once, so it won't do anything if you run it again
DATASET_PATH = '/cluster/work/igp_psr/ai4good/group-1b/data/'
paths = np.array([f for f in listdir(DATASET_PATH + 'train/') if isfile(join(DATASET_PATH + 'train/', f))])
idx = np.arange(len(paths))
np.random.shuffle(idx)
split = round(0.8*len(idx))

print('nb of files : ', len(paths))
train_paths = [DATASET_PATH + 'train/' + file for file in paths[idx[:split]]]
val_paths = [DATASET_PATH + 'train/' + file for file in paths[idx[split:]]]
print('nb of train files : ', len(train_paths))
print('nb of val files : ', len(val_paths))

if not IS_DATASET_SPLIT :
    Path(DATASET_PATH + 'val/').mkdir(parents=True, exist_ok=True)
    for file in val_paths:
        shutil.move(file, DATASET_PATH + 'val/')

print('Done')