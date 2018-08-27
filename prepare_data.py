import argparse, sys

from dataset import DatasetFromFolder

parser = argparse.ArgumentParser(description='Data augmentation for super resolution')
parser.add_argument('--image_folder',
                    type=str,
                    required=True,
                    help="path to image dataset")
parser.add_argument('--rotate',
                    type=int,
                    required=False,
                    action='append',
                    help="rotation in degrees")
parser.add_argument('--scale',
                    type=float,
                    required=False,
                    action='append',
                    help="downsampling factor")
parser.add_argument('--translate',
                    type=float,
                    nargs=2,
                    required=False,
                    action='append',
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
                    action='append',
                    help="tilt angle")
parser.add_argument('--additive_brightness',
                    type=int,
                    required=False,
                    action='append',
                    help="change brightness with additive pixel value")
parser.add_argument('--brightness',
                    type=float,
                    nargs=2,
                    action='append',
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
                    help="save images in RGB format")
parser.add_argument('--save_grayscale',
                    action='store_true',
                    help="save images in lossless grayscale format")
parser.add_argument('--save_dataset',
                    action='store_true',
                    help="save dataset csv")

def compose_transforms(opt, args):
    transforms = []
    
    for arg in args:
        if arg == '--rotate':
            transforms.append(('rotate', opt.rotate.pop(0)))
        elif arg == '--scale':
            transforms.append(('scale', opt.scale.pop(0)))
        elif arg == '--flip_horizontal':
            transforms.append(('flip_horizontal', None))
        elif arg == '--flip_vertical':
            transforms.append(('flip_vertical', None))
        elif arg == '--tilt_angle':
            transforms.append(('tilt_angle', opt.tilt_angle.pop(0)))
        elif arg == '--translate':
            transforms.append(('translate', opt.translate.pop(0)))
        elif arg == '--additive_brightness':
            transforms.append(('additive_brightness',
                               opt.additive_brightness.pop(0)))
        elif arg == '--brightness':
            transforms.append(('brightness', opt.brightness.pop(0)))
        else:
            pass

    return transforms
    

def main(args):
    opt = parser.parse_args(args)

    with DatasetFromFolder(image_dir=opt.image_folder,
                           sample_size=opt.sample_size,
                           dataset_csv=opt.dataset_csv,
                           hdf5_path=opt.hdf5_path) as sampled_dataset:
        transforms = compose_transforms(opt, args)
        for input_image in sampled_dataset:
            sampled_dataset.transform(input_image, transforms)
            if opt.save_images or opt.save_grayscale:
                sampled_dataset.save_image(opt.save_grayscale)

        if opt.save_dataset:
            sampled_dataset.save_dataset("dataset.csv")

if __name__ == "__main__":
    main(sys.argv[1:])
