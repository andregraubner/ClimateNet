import numpy as np
import xarray as xr
from tqdm import tqdm
from patchify import patchify

def get_patches(data, vars, patch_size, stride):

    image_size = np.squeeze(np.array(data[0][vars[0]])).shape
    H = image_size[0] # image height
    W = image_size[1] # image width
    output_H = np.floor((H-patch_size+stride)/stride).astype(int)
    output_W = np.floor((W-patch_size+stride)/stride).astype(int)
    all_patches = np.empty((1, output_H, output_W, len(vars)+1, patch_size, patch_size))

    for i, image in enumerate(tqdm(data)):
        im = np.expand_dims(np.array(image['LABELS']), axis=0)
        for j, var in enumerate(vars):
            im = np.concatenate((im, np.array(image[var])), axis=0)
        patches = patchify(im, (len(vars)+1, patch_size, patch_size), stride) # +1 to include labels in patch
        all_patches = np.concatenate((all_patches, patches), axis=0)

    return all_patches