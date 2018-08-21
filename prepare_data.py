import argparse, sys

from collections import OrderedDict
from dataset import DatasetFromFolder

parser = argparse.ArgumentParser(description='Data augmentation for super resolution')
parser.add_argument('--image_folder',
                    type=str,
                    required=True,
                    help="path to image dataset")
parser.add_argument('--rotate',
                    type=int,
                    required=False,
                    help="rotation in degrees (-180 to 180)")
parser.add_argument('--scale',
                    type=float,
                    required=False,
                    help="downsampling factor")
parser.add_argument('--translate',
                    type=int,
                    nargs='+',
                    required=False,
                    help="translate image specifying x and y offsets")
parser.add_argument('--flip_horizontal',
                    action='store_true',
                    help="flip image left to right")
parser.add_argument('--flip_vertical',
                    action='store_true',
                    help="flip image top to bottom")
parser.add_argument('--tilt_angle',
                    type=float,
                    required=False,
                    help="tilt angle")
parser.add_argument('--additive_brightness',
                    type=int,
                    required=False,
                    help="change brightness with additive pixel value")
parser.add_argument('--brightness',
                    type=float,
                    nargs='+',
                    required=False,
                    help="change brightness with factor and offset")
parser.add_argument('--sample_size',
                    type=int,
                    required=False,
                    default=-1,
                    help="number of samples to generate")
parser.add_argument('--dataset_csv',
                    type=str,
                    required=False,
                    help="specified dataset to load")
parser.add_argument('--hdf5_path',
                    type=str,
                    required=False,
                    help="output hdf5 file")
parser.add_argument('--save_images',
                    action='store_true',
                    help="save images")
parser.add_argument('--save_dataset',
                    action='store_true',
                    help="save dataset csv")

def compose_transforms(opt, args):
    transforms = OrderedDict()
    
    for arg in args:
        if 'rotate' in arg:
            transforms['rotate'] = opt.rotate
        elif 'scale' in arg:
            transforms['scale'] = opt.scale
        elif 'flip_horizontal' in  arg:
            transforms['flip_horizontal'] = None
        elif 'flip_vertical' in arg:
            transforms['flip_vertical'] = None
        elif 'tilt_angle' in arg:
            transforms['tilt_angle'] = opt.tilt_angle
        elif 'translate' in arg:
            transforms['translate'] = opt.translate
        elif 'additive_brightness' in arg:
            transforms['additive_brightness'] = opt.additive_brightness
        elif 'brightness' in arg:
            transforms['brightness'] = opt.brightness
        else:
            pass

    return transforms
    

def main(args):
    opt = parser.parse_args(args)

    with DatasetFromFolder(image_dir=opt.image_folder,
                           sample_size=opt.sample_size,
                           transforms=compose_transforms(opt, args),
                           dataset_csv=opt.dataset_csv,
                           hdf5_path=opt.hdf5_path) as sampled_dataset:
        for input_image in sampled_dataset:
            transformed_image = sampled_dataset.transform(input_image)
            if opt.save_images:
                sampled_dataset.save_image()

        if opt.save_dataset:
            sampled_dataset.save_dataset("dataset.csv")

if __name__ == "__main__":
    main(sys.argv[1:])
