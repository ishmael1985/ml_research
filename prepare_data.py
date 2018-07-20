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
                    help="rotation in degrees (-180 to 180)")
parser.add_argument('--scale',
                    type=float,
                    required=False,
                    help="downsampling factor")
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

def main(args):
    opt = parser.parse_args(args)

    with DatasetFromFolder(image_dir=opt.image_folder,
                           sample_size=opt.sample_size,
                           rotation=opt.rotate,
                           scale=opt.scale,
                           flip_horizontal=opt.flip_horizontal,
                           flip_vertical=opt.flip_vertical,
                           tilt_angle=opt.tilt_angle,
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
