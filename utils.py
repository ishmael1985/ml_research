import math
import numpy as np
import h5py

from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize
from skimage.color import rgb2ycbcr

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg", ".tif"])

def load_img(filepath):
    image = Image.open(filepath)
    if image.mode == 'RGB':
        ycbcr_image = rgb2ycbcr(np.asarray(image))
        return Image.fromarray(ycbcr_image.astype('uint8'), 'YCbCr')
    else:
        return image

def read_hdf5(path):
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('data'))
        label_ = np.array(hf.get('label'))
        return input_, label_

def get_center_crop(image, width, height):
    composed_transform = Compose([CenterCrop(size=(height, width))])
    return composed_transform(image)

def get_interpolated_image(image, scale):
    width = image.size[0] * scale
    height = image.size[1] * scale
    composed_transform = Compose([Resize(size=(height, width),
                                         interpolation=Image.BICUBIC)])
    return composed_transform(image)


