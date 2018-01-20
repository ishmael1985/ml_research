import argparse

from dataset import DatasetFromFolder

parser = argparse.ArgumentParser(description='Data augmentation for super resolution')
parser.add_argument('--rotate', type=int, required=False, help="rotation in degrees (-180 to 180)")
parser.add_argument('--scale', type=int, required=False, help="downsampling factor")
#parser.add_argument('--brightness', type=float, required=False, help="brightness factor")
parser.add_argument('--sample_size', type=int, required=False, default=16, help="number of samples to generate")
parser.add_argument('--image_folder', required=True, help="number of samples to generate")

opt = parser.parse_args()

if __name__ == "__main__":
    sampled_dataset = DatasetFromFolder(image_dir=opt.image_folder,
                                        sample_size=opt.sample_size,
                                        rotation=opt.rotate,
                                        scale=opt.scale)
                                        
    for input_image, transformed_image in sampled_dataset:
        sampled_dataset.save_image(transformed_image)
        sampled_dataset.copy_input_image()

