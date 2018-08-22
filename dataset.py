import random
import math
import numpy as np
import h5py
import json
import csv
import os
import pathlib

from os import walk, makedirs
from os.path import join, exists, splitext, dirname, basename, realpath
from math import ceil, floor
from shutil import copy2
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Resize,
    RandomRotation,
    Compose
)
from utils import is_image_file, load_img


def _get_rotated_rect_max_area(w, h, angle):
    """
    Stolen from here: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr, hr

def rotate_max_area(rotated_image, width, height, angle):
    max_width, max_height = _get_rotated_rect_max_area(width,
                                height, math.radians(angle))
    
    w, h = rotated_image.size
    
    y1 = h//2 - int(max_height/2)
    y2 = y1 + int(max_height)
    x1 = w//2 - int(max_width/2)
    x2 = x1 + int(max_width)
    
    return rotated_image.crop(box=(x1,y1,x2,y2))
    
def calculate_cropped_size(width, height, scale_factor):
    cropped_width = width - (width % scale_factor)
    cropped_height = height - (height % scale_factor)
    
    return cropped_width, cropped_height

def get_images(image_dir):
    image_files = []
    for root, dirs, files in walk(image_dir):
        for file in files:
            if is_image_file(file):
                p = pathlib.Path(join(root, file))
                image_files.append(str(p.relative_to(image_dir)))

    return image_files


class DatasetFromFolder:
    def __init__(self, image_dir, transforms,
                 sample_size=-1, hdf5_path='', dataset_csv=''):
        self.image_dir = image_dir
        self.script_dir = dirname(realpath(__file__))
        self.dest_dir = join(self.script_dir, 'generated')
        self.image_filenames = []
        self.current_image_file = ''
        self.transformed = None
        self.image_count = 0
        
        if dataset_csv and os.path.isfile(dataset_csv):
            with open(dataset_csv, "r") as csvfile:
                dataset_csv = csv.reader(csvfile)
                for row in dataset_csv:
                    for image_file in row:
                        if exists(join(self.image_dir, image_file)):
                            self.image_filenames.append(image_file)
                
        if not self.image_filenames:
            self.image_filenames = get_images(self.image_dir)
            if sample_size > 0:
                self.image_filenames = random.sample(self.image_filenames,
                                                     sample_size)
        self.transforms = transforms
        self.hdf5_path = hdf5_path
        self.sub_inputs = []
        self.sub_labels = []
        self.params = {}

    def __enter__(self):
        if self.hdf5_path:
            with open("hdf5.json", "r") as config:
                self.params = json.load(config)
            with h5py.File(self.hdf5_path, 'w') as hf:
                hf.create_dataset('data', (0, self.params["channels"],
                                           self.params["input_size"],
                                           self.params["input_size"]),
                                  maxshape=(None, self.params["channels"],
                                            self.params["input_size"],
                                            self.params["input_size"]),
                                  dtype='f4', compression=9,
                                  chunks=(64, self.params["channels"],
                                          self.params["input_size"],
                                          self.params["input_size"]))
                hf.create_dataset('label', (0, self.params["channels"],
                                            self.params["label_size"],
                                            self.params["label_size"]),
                                  maxshape=(None, self.params["channels"],
                                            self.params["label_size"],
                                            self.params["label_size"]),
                                  dtype='f4', compression=9,
                                  chunks=(64, self.params["channels"],
                                          self.params["label_size"],
                                          self.params["label_size"]))
        return self

    def __exit__(self, type, value, traceback):
        if self.hdf5_path:
            # Efficient random shuffling in place
            perm = np.random.permutation(len(self.sub_inputs))
            
            inputs = np.asarray(self.sub_inputs, dtype=np.float32)[perm]
            labels = np.asarray(self.sub_labels, dtype=np.float32)[perm]
            
            with h5py.File(self.hdf5_path, 'a') as hf:
                hf["data"].resize((hf["data"].shape[0] + inputs.shape[0]),
                                  axis = 0)
                hf["data"][-inputs.shape[0]:] = inputs

                hf["label"].resize((hf["label"].shape[0] + labels.shape[0]),
                                    axis = 0)
                hf["label"][-labels.shape[0]:] = labels

    def __getitem__(self, index):
        self.current_image_file = self.image_filenames[index]

        return load_img(join(self.image_dir, self.current_image_file))

    def __len__(self):
        return len(self.image_filenames)

    def _modcrop(self, scale):
        input_width, input_height = self.transformed.size
        cropped_width, cropped_height = calculate_cropped_size(input_width,
                                                               input_height,
                                                               scale)
        center_crop = Compose([CenterCrop(size=(cropped_height, cropped_width))])

        return center_crop(self.transformed)
    
    def _preprocess(self, scale):
        label_ = self._modcrop(scale)
    
        cropped_width, cropped_height = label_.size
        composed_transform = Compose([Resize(size=(int(cropped_height / scale),
                                                   int(cropped_width / scale)),
                                             interpolation=Image.BICUBIC),
                                      Resize(size=(cropped_height, cropped_width),
                                             interpolation=Image.BICUBIC)])
        input_ = composed_transform(label_)
        
        if self.params["channels"] == 3:
            return np.array(input_), np.array(label_)
        else:
            return np.array(input_.split()[0]), np.array(label_.split()[0])

    def _make_subimages(self):
        patch_size = self.params["input_size"]
        label_size = self.params["label_size"]
        stride = self.params["stride"]
        channels = self.params["channels"]
        batch_size = self.params["batch_size"]

        for scale in self.params["upscale_factors"]:
            input_, label_, = self._preprocess(scale)
            
            # Make subimages of LR and HR
            # Input
            if channels == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
                
                    
            for x in range(0, h - patch_size + 1, stride):
                for y in range(0, w - patch_size + 1, stride):
                    sub_input = input_[x: x + patch_size, y: y + patch_size] # 41 * 41
                    sub_label = label_[x: x + label_size, y: y + label_size] # 41 * 41

                    # Normalize
                    sub_input =  sub_input / 255.0
                    sub_label =  sub_label / 255.0

                    # Reshape the subinput and sublabel
                    sub_input = sub_input.reshape([channels, patch_size,
                                                   patch_size])
                    sub_label = sub_label.reshape([channels, label_size,
                                                   label_size])

                    # Add to sequence
                    self.sub_inputs.append(sub_input)
                    self.sub_labels.append(sub_label)

        self.image_count = self.image_count + 1
        if (self.image_count % batch_size) == 0:
            # Efficient random shuffling in place
            perm = np.random.permutation(len(self.sub_inputs))

            inputs = np.asarray(self.sub_inputs, dtype=np.float32)[perm]
            self.sub_inputs = []

            labels = np.asarray(self.sub_labels, dtype=np.float32)[perm]
            self.sub_labels = []

            with h5py.File(self.hdf5_path, 'a') as hf:
                hf["data"].resize((hf["data"].shape[0] + inputs.shape[0]),
                                  axis = 0)
                hf["data"][-inputs.shape[0]:] = inputs

                hf["label"].resize((hf["label"].shape[0] + labels.shape[0]),
                                   axis = 0)
                hf["label"][-labels.shape[0]:] = labels

    
    def get_low_angle_perspective(self, image, angle):
        H = 1
        theta = np.deg2rad(angle)
        w, h = image.size
        x1 = 0
        x2 = w
        y1 = 0
        y2 = h

        original_plane = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        new_plane = []

        for u, v in original_plane:
            s1 = 1/np.cos(theta) + H*np.sin(theta)*(2*(v-h/2)/h)
            s2 = 1/np.cos(theta) + H*np.sin(theta)*(2*(v-(h-1)/2)/(h-1))
            up = (w-1)/2 + (u-(w-1)/2)/s1
            vp = (h-1)/2 + np.cos(theta)*(v-(h-1)/2)/s2
            new_plane.append((up, vp))

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)

        perspective_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
        perspective_coefficients_matrix = np.array(perspective_coefficients_matrix).reshape(8)

        skewed_image = image.transform(image.size,
                                       Image.PERSPECTIVE,
                                       perspective_coefficients_matrix,
                                       resample=Image.BICUBIC)
        
        bounding_box = (ceil(new_plane[0][0]),
                        ceil(new_plane[0][1]),
                        floor(new_plane[1][0]),
                        h if h < new_plane[3][1] else floor(new_plane[3][1]))

        return skewed_image.crop(bounding_box)

    def transform(self, image):
        for transform, value in self.transforms.items():
            input_width, input_height = image.size
            if 'flip_horizontal' in transform:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif 'flip_vertical' in transform:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif 'rotate' in transform:
                image_transforms = [RandomRotation(degrees=(value, value),
                                                   expand=True,
                                                   resample=Image.BICUBIC)]
                composed_transform = Compose(image_transforms)
                image = composed_transform(image)

                image = rotate_max_area(image,
                                        input_width,
                                        input_height,
                                        value)
            elif 'translate' in transform:
                a = 1
                b = 0
                c = value[0]
                d = 0
                e = 1
                f = value[1]
                # For PIL apply inverse of translation matrix
                translate = image.transform(image.size, Image.AFFINE,
                                            (a, b, -c, d, e, -f))
                image = translate.crop(translate.getbbox())
            elif 'tilt_angle' in transform:
                tilt_angle = max(0, min(value, 85))
                image = self.get_low_angle_perspective(image, -tilt_angle)
            elif 'scale' in transform:
                cropped_width, cropped_height = calculate_cropped_size(input_width,
                                                                       input_height,
                                                                       value)
                image_transforms = [CenterCrop(size=(cropped_height,
                                                     cropped_width)),
                                    Resize(size=(int(cropped_height / value),
                                                 int(cropped_width / value)),
                                           interpolation=Image.BICUBIC)]
                composed_transform = Compose(image_transforms)
                image = composed_transform(image)
            elif 'additive_brightness' in transform:
                y, cb, cr = image.split()
                out_y = np.asarray(y) + value
                out_y = out_y.clip(0, 255)
                y = Image.fromarray(np.uint8(out_y), mode='L')
                image = Image.merge('YCbCr', [y, cb, cr])
            elif 'brightness' in transform:
                y, cb, cr = image.split()
                y_ = np.asarray(y)
                out_y = y_ - value[1]
                out_y = y_ + ((value[0] / 255) * out_y)
                out_y = out_y.clip(0, 255)
                y = Image.fromarray(np.uint8(out_y), mode='L')
                image = Image.merge('YCbCr', [y, cb, cr])
            else:
                pass

        self.transformed = image.copy()

        if self.params:
            self._make_subimages()

        return self.transformed
        
    def save_image(self):
        makedirs(self.dest_dir, exist_ok=True)
        output_filename = basename(self.current_image_file)
        label =  ''

        for transform, value in self.transforms.items():
            if 'flip_horizontal' in transform:
                label = label + "_flip_lr"
            elif 'flip_vertical' in transform:
                label = label + "_flip_tb"
            elif 'rotate' in transform:
                label = label + "_rotate_" + str(value)
            elif 'translate' in transform:
                label = (label + "_translate_x_" + str(value[0]) +
                         "_y_" + str(value[1]))
            elif 'tilt_angle' in transform:
                label = label + "_tilt_" + str(value)
            elif 'scale' in transform:
                label = label + "_scale_" + str(value)
            elif 'additive_brightness' in transform:
                label = label + "_brightness_add_" + str(value)
            elif 'brightness' in transform:
                label = (label + "_brightness_mul_" + str(value[0]) +
                         "_" + str(value[1]))
            else:
                pass

        if label:
            output_filename = splitext(output_filename)[0] + label + '.png'
            output_image =  self.transformed.convert('RGB')
            output_image.save(join(self.dest_dir, output_filename))
        else:
            copy2(join(self.image_dir, self.current_image_file), self.dest_dir)

        print("Image saved : " + join(self.dest_dir, output_filename))

    def save_dataset(self, csv_file):
        with open(join(self.script_dir, csv_file), "w") as csvfile:
            dataset_csv = csv.writer(csvfile)
            dataset_csv.writerow(self.image_filenames)
        
