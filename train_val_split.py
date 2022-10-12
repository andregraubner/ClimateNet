
from os import listdir
from os.path import isfile, join
import numpy as np
import shutil

print('stuff')
paths = np.array([f for f in listdir('data/climatenet_new/train/') if isfile(join('data/train/', f))])
idx = np.arange(len(paths))
np.random.shuffle(idx)
split = round(0.8*len(idx))

train_paths = ['data/train/train/' + file for file in paths[idx[:split]]]
val_paths = ['data/train/test/' + file for file in paths[idx[split:]]]

dst_path = np.concatenate((train_paths, val_paths))

for i, file in enumerate(paths):
    shutil.move('data/train/'+file, dst_path[i])