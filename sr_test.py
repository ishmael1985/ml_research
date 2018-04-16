import tensorflow as tf
import numpy as np
import argparse
import csv

from torchvision.transforms import CenterCrop, Compose, Resize
from fsrcnn import FSRCNN
from dataset import DatasetFromFolder
from PIL import Image
from skimage import img_as_ubyte, img_as_float
from skimage.measure import compare_psnr, compare_ssim

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
                    help="number of randomly sampled test images")
parser.add_argument('--load_test',
                    type=str,
                    required=False,
                    help="load csv file containing specified set of test images")
parser.add_argument('--save_test',
                    action='store_true',
                    help="save set of images used for testing")
parser.add_argument('--save_result',
                    action='store_true',
                    help="save IQA results")


def get_center_crop(image, width, height):
    composed_transform = Compose([CenterCrop(size=(height, width))])
    return composed_transform(image)

def get_interpolated_image(image, scale):
    width = image.size[0] * scale
    height = image.size[1] * scale
    composed_transform = Compose([Resize(size=(height, width),
                                         interpolation=Image.BICUBIC)])
    return composed_transform(image)

def main():
    opt = parser.parse_args()

    average_psnr = 0
    average_bicubic_psnr = 0
    average_ssim = 0
    average_bicubic_ssim = 0
    sampled_dataset = DatasetFromFolder(image_dir=opt.image_folder,
                                        sample_size=opt.sample_size,
                                        dataset_csv=opt.load_test,
                                        scale=opt.scale)

    if opt.save_result:
        results_file = open("results.csv", "w")
        results_csv = csv.writer(results_file)

    for ground_truth in sampled_dataset:
        downsampled_image = sampled_dataset.transform(ground_truth)

        with tf.Session() as session:
            fsrcnn = FSRCNN(session, opt.checkpoint_dir)
            out_img_y = fsrcnn.test(downsampled_image)
            
            out_img_y = out_img_y.clip(0, 1)
            
            height, width = out_img_y.shape[:2]
            gt_img = get_center_crop(ground_truth, width, height)
            gt_img = img_as_float(np.asarray(gt_img.split()[0]))
            
            y_interpolated = get_interpolated_image(downsampled_image.split()[0], opt.scale)
            y_interpolated = get_center_crop(y_interpolated, width, height)
            interpolated = img_as_float(np.asarray(y_interpolated))

            predicted = img_as_float(out_img_y)
            
##            img = Image.fromarray(img_as_ubyte(out_img_y), 'L')
##            gt = Image.fromarray(img_as_ubyte(gt_img), 'L')
##            img.show()
##            gt.show()

            psnr = compare_psnr(gt_img, predicted)
            ssim = compare_ssim(gt_img, predicted,
                                data_range=gt_img.max() - gt_img.min())
            bicubic_psnr = compare_psnr(gt_img, interpolated)
            bicubic_ssim = compare_ssim(gt_img, interpolated,
                                        data_range=gt_img.max() - gt_img.min())
        
            if opt.save_result:
                results_csv.writerow([sampled_dataset.current_image_file, scale,
                                      psnr, ssim])

            average_psnr += psnr
            average_ssim += ssim
            average_bicubic_psnr += bicubic_psnr
            average_bicubic_ssim += bicubic_ssim

        tf.reset_default_graph()


    sample_size = len(sampled_dataset.image_filenames)
    print("Upscale factor = ", opt.scale)
    print("Average PSNR for {} samples = {}".format(sample_size,
                                                    average_psnr / sample_size))
    print("Average SSIM for {} samples = {}".format(sample_size,
                                                    average_ssim / sample_size))
    print("Average bicubic PSNR for {} samples = {}".format(sample_size,
                                                            average_bicubic_psnr / sample_size))
    print("Average bicubic SSIM for {} samples = {}".format(sample_size,
                                                            average_bicubic_ssim / sample_size))

    if opt.save_test:
        sampled_dataset.save_dataset("test.csv")

if __name__ == "__main__":
    main()
