import math
import numpy as np
import h5py

from PIL import Image
from skimage.color import rgb2ycbcr

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

def load_img(filepath):
    rgb_image = Image.open(filepath)
    ycbcr_image = rgb2ycbcr(np.asarray(rgb_image))
    return Image.fromarray(ycbcr_image.astype('uint8'))

def read_hdf5(path):
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('data'))
        label_ = np.array(hf.get('label'))
        return input_, label_

def compute_psnr(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
