import argparse

from dataset import DatasetFromFolder

parser = argparse.ArgumentParser(description='Data augmentation for super resolution')
parser.add_argument('--rotate', type=int, required=False, help="rotation in degrees (-180 to 180)")
parser.add_argument('--scale', type=int, required=False, help="downsampling factor")
#parser.add_argument('--brightness', type=float, required=False, help="brightness factor")
parser.add_argument('--sample_size', type=int, required=False, default=-1, help="number of samples to generate")
parser.add_argument('--image_folder', type=str, required=True, help="number of samples to generate")
parser.add_argument('--dataset_csv', type=str, required=False, help="specified dataset to load")
parser.add_argument('--hdf5_path', type=str, required=False, help="output hdf5 file")
parser.add_argument('--save_images', action='store_true', help="save images")
parser.add_argument('--save_dataset', action='store_true', help="save dataset csv")

opt = parser.parse_args()

if __name__ == "__main__":
    with DatasetFromFolder(image_dir=opt.image_folder,
                           sample_size=opt.sample_size,
                           rotation=opt.rotate,
                           scale=opt.scale,
                           dataset_csv=opt.dataset_csv,
                           hdf5_path=opt.hdf5_path) as sampled_dataset:
        for input_image in sampled_dataset:
            transformed_image = sampled_dataset.transform(input_image)
            if opt.hdf5_path:
                sampled_dataset.make_subimages(transformed_image)
            if opt.save_images:
                sampled_dataset.save_image(transformed_image)

        if opt.save_dataset:
            sampled_dataset.save_dataset()
