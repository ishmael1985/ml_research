import random
import math
import numpy as np
import h5py
import json
import csv

from os import listdir, makedirs
from os.path import join, exists, basename, splitext, dirname, realpath
from shutil import copy2
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, RandomRotation, Compose


def _get_rotated_rect_max_area(w, h, angle):
    """
    Stolen from here: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr

def rotate_max_area(rotated_image, width, height, angle):
    max_width, max_height = _get_rotated_rect_max_area(width,
                                height, math.radians(angle))
    
    w, h = rotated_image.size
    
    y1 = h//2 - int(max_height/2)
    y2 = y1 + int(max_height)
    x1 = w//2 - int(max_width/2)
    x2 = x1 + int(max_width)
    
    return rotated_image.crop(box=(x1,y1,x2,y2))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

def load_img(filepath):
    return Image.open(filepath).convert('YCbCr')
    
def calculate_cropped_size(width, height, scale_factor):
    cropped_width = width - (width % scale_factor)
    cropped_height = height - (height % scale_factor)
    
    return cropped_width, cropped_height
    

class DatasetFromFolder():
    def __init__(self, image_dir, sample_size=-1, rotation=None, scale=None, hdf5_path='', dataset_csv=''):
        self.image_dir = image_dir
        self.script_dir = dirname(realpath(__file__))
        self.dest_dir = join(self.script_dir, 'generated')
        self.image_filenames = []
        
        if dataset_csv:
            with open(dataset_csv, "r") as csvfile:
                dataset_csv = csv.reader(csvfile)
                for row in dataset_csv:
                    for image_file in row:
                        if exists(join(image_dir, image_file)):
                            self.image_filenames.append(image_file)
                
        if not self.image_filenames:
            self.image_filenames = [x for x in listdir(image_dir) if is_image_file(join(image_dir, x))]
            if sample_size > 0:
                self.image_filenames = random.sample(self.image_filenames,
                                                     sample_size)
        self.scale = scale
        self.rotation = rotation
        self.hdf5_path = hdf5_path
        self.sub_inputs = []
        self.sub_labels = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.hdf5_path:
            inputs = np.asarray(self.sub_inputs)
            labels = np.asarray(self.sub_labels)

            with h5py.File(self.hdf5_path, 'w') as hf:
                hf.create_dataset('data', data=inputs)
                hf.create_dataset('label', data=labels)

    def __getitem__(self, index):
        self.current_image_file = join(self.image_dir, self.image_filenames[index])

        return load_img(self.current_image_file)

    def __len__(self):
        return len(self.image_filenames)

    def _modcrop(self, image, scale):
        input_width, input_height = image.size[0], image.size[1]
        cropped_width, cropped_height = calculate_cropped_size(input_width, input_height, scale)
    
        center_crop = Compose([CenterCrop(size=(cropped_height, cropped_width))])

        return center_crop(image)
    
    def _preprocess(self, image, scale):
        label_ = self._modcrop(image, scale)
    
        cropped_width, cropped_height = label_.size[0], label_.size[1]
        composed_transform = Compose([Resize(size=(cropped_height // scale,
                                                   cropped_width // scale),
                                            interpolation=Image.BICUBIC)])
        input_ = composed_transform(label_)

        return np.array(input_.split()[0]), np.array(label_.split()[0])

    def make_subimages(self, image):
        with open("config.json", "r") as config:
            params = json.load(config)

        block_size = params["block_size"]
        stride = params["stride"]
        scale = params["scale"]
        input_, label_, = self._preprocess(image, scale)
        
        # Make subimages of LR and HR
        # Input
        h, w = input_.shape
        for x in range(0, h - block_size + 1, stride):
            for y in range(0, w - block_size + 1, stride):
                sub_input = input_[x: x + block_size, y: y + block_size] # 17 * 17

                # Reshape sub-inputs
                sub_input = sub_input.reshape([block_size, block_size, 1])
                # Normalize
                sub_input =  sub_input / 255.0
                # Append sub-input to list of sub-inputs for the image 
                self.sub_inputs.append(sub_input)

        # Label
        h, w = label_.shape
        for x in range(0, h - block_size * scale + 1, stride * scale):
            for y in range(0, w - block_size * scale + 1, stride * scale):
                sub_label = label_[x: x + block_size * scale, y: y + block_size * scale] # 17r * 17r
                
                # Reshape sub-label
                sub_label = sub_label.reshape([block_size * scale , block_size * scale, 1])
                # Normalize
                sub_label =  sub_label / 255.0
                # Append sub-label to list of sub-labels for the image 
                self.sub_labels.append(sub_label)

    def transform(self, image):
        image_transforms = []
        input_width, input_height = image.size[0], image.size[1]
        
        if self.rotation:
            image_transforms.append(RandomRotation(degrees=(self.rotation,
                                                            self.rotation),
                                                   expand=True,
                                                   resample=Image.BICUBIC))
        if self.scale:
            cropped_width, cropped_height = calculate_cropped_size(input_width,
                                                                   input_height,
                                                                   self.scale)
            image_transforms.append(CenterCrop(size=(cropped_height,
                                                     cropped_width)))
            image_transforms.append(Resize(size=(cropped_height // self.scale,
                                                 cropped_width // self.scale),
                                           interpolation=Image.BICUBIC))
    
        if image_transforms:
            composed_transform = Compose(image_transforms)
            transformed = composed_transform(image)
            if self.rotation:
                # Get new input size for the maximum area of rotated image if downsampled
                if self.scale:
                    input_width, input_height = transformed.size[0], transformed.size[1]
                transformed = rotate_max_area(transformed, input_width,
                                              input_height, self.rotation)
        else:
            transformed = image.copy()

        return transformed
        
    def save_image(self, image):
        makedirs(self.dest_dir, exist_ok=True)
        output_filename = basename(self.dest_dir + self.current_image_file)
        label =  ''

        if self.rotation:
            label = label + "_rotate-" + str(self.rotation)
        if self.scale:
            label = label + "_scale-" + str(self.scale)

        if label:
            output_filename = splitext(output_filename)[0] + label + '.png'
            output_image = image.convert('RGB')
            output_image.save(join(self.dest_dir, output_filename))
        else:
            copy2(self.current_image_file, self.dest_dir)

        print("Image saved : " + join(self.dest_dir, output_filename))

    def save_dataset(self):
        csvfile = open(join(self.script_dir, "dataset.csv"), "w")
        dataset_csv = csv.writer(csvfile)
        dataset_csv.writerow(self.image_filenames)
        
