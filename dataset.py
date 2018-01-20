import random

from os import listdir, makedirs
from os.path import join, basename, splitext
from shutil import copy2
from PIL import Image
from utils import rotate_max_area
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    return Image.open(filepath).convert('YCbCr')
    
def calculate_cropped_size(width, height, scale_factor):
    cropped_width = width - (width % scale_factor)
    cropped_height = height - (height % scale_factor)
    return cropped_width, cropped_height

class DatasetFromFolder():
    def __init__(self, image_dir, sample_size, rotation=None, scale=None):
        self.image_dir = image_dir
        self.dest_dir = self.image_dir + 'generated/'
        self.image_filenames = random.sample([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)],
                                            sample_size)
        self.scale = scale
        self.rotation = rotation

    def __getitem__(self, index):
        self.current_image_file = self.image_filenames[index]
        input = load_img(self.image_filenames[index])
        input_width, input_height = input.size[0], input.size[1]
        transformed = None
        image_transforms = []

        if self.rotation:
            image_transforms.append(transforms.RandomRotation(degrees=(self.rotation, self.rotation),
                                                            expand=True,
                                                            resample=Image.BICUBIC))
        if self.scale:
            cropped_width, cropped_height = calculate_cropped_size(input_width, input_height, self.scale)
            image_transforms.append(transforms.CenterCrop(size=(cropped_height, cropped_width)))
            image_transforms.append(transforms.Resize(size=(cropped_height // self.scale, cropped_width // self.scale),
                                                    interpolation=Image.BICUBIC))
    
        if image_transforms:
            composed_transform = transforms.Compose(image_transforms)
            transformed = composed_transform(input)
            if self.rotation:
                # Get new input size for the maximum area of rotated image if downsampled
                if self.scale:
                    input_width, input_height = transformed.size[0], transformed.size[1]
                transformed = rotate_max_area(transformed, input_width, input_height, self.rotation)
        else:
            transformed = input.copy()

        return input, transformed

    def __len__(self):
        return len(self.image_filenames)
        
    def save_image(self, image):
        makedirs(self.dest_dir, exist_ok=True)
        
        output_filename = splitext(basename(self.dest_dir + self.current_image_file))[0] + '_transformed.png'
        output_image = image.convert('RGB')
        output_image.save(self.dest_dir + output_filename)
        print("Image saved : " + self.dest_dir + output_filename)

    def copy_input_image(self):
        makedirs(self.dest_dir, exist_ok=True)
        copy2(self.current_image_file, self.dest_dir)
