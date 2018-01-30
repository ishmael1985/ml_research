import tensorflow as tf
import numpy as np
import argparse
import csv

from torchvision.transforms import CenterCrop, Compose
from model import ESPCN
from dataset import DatasetFromFolder
from utils import compute_psnr
from PIL import Image

parser = argparse.ArgumentParser(description='Super resolution test')
parser.add_argument('--image_folder',
                    type=str,
                    required=True,
                    help="path to image folder containing test images")
parser.add_argument('--checkpoint_dir',
                    type=str,
                    required=False,
                    default='checkpoint',
                    help="Name of checkpoint directory")
parser.add_argument('--scale',
                    type=int,
                    required=False,
                    default=3,
                    help="downsampling factor")
parser.add_argument('--sample_size',
                    type=int,
                    required=False,
                    default=10,
                    help="number of samples to generate")
parser.add_argument('--load_test',
                    type=str,
                    required=False,
                    help="path to csv file containing specified set of images to load")
parser.add_argument('--save_test',
                    action='store_true',
                    help="save set of images used for testing to specified csv file")
parser.add_argument('--save_result',
                    action='store_true',
                    help="save PSNR results")


def get_center_crop(image, width, height):
    composed_transform = Compose([CenterCrop(size=(height, width))])
    return composed_transform(image)

def main():
    opt = parser.parse_args()
    
    average_psnr = 0
    sampled_dataset = DatasetFromFolder(image_dir=opt.image_folder,
                                        sample_size=opt.sample_size,
                                        dataset_csv=opt.load_test,
                                        scale=opt.scale)
    with tf.Session() as session:
        espcn = ESPCN(session)

        if opt.save_result:
            results_file = open("results.csv", "w")
            results_csv = csv.writer(results_file)
            
        for ground_truth in sampled_dataset:
            downsampled_image = sampled_dataset.transform(ground_truth)
            out_img = espcn.test(opt.checkpoint_dir, downsampled_image)
            
            out_img *= 255.0
            out_img = out_img.clip(0, 255)
            
            height, width = out_img.shape[:2]
            ground_truth = get_center_crop(ground_truth, width, height)
            gt_img = np.asarray(ground_truth.split()[0])

            #img = Image.fromarray(out_img.astype('uint8'), 'L')
            #img.show()

            psnr = compute_psnr(out_img, gt_img)

            if opt.save_result:
                results_csv.writerow([sampled_dataset.current_image_file, psnr])

            average_psnr += compute_psnr(out_img, gt_img)
            
    if opt.save_test:
        sampled_dataset.save_dataset("test.csv")

    sample_size = len(sampled_dataset.image_filenames)
    print("Upscale factor = ", opt.scale)
    print("Average PSNR for {} samples = {}".format(sample_size,
                                                    average_psnr / sample_size))

if __name__ == "__main__":
    main()
