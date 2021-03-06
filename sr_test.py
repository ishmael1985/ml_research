import torch
import numpy as np
import argparse, sys
import csv

from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torch.autograd import Variable
from dataset import DatasetFromFolder
from utils import get_center_crop, get_interpolated_image
from PIL import Image
from skimage import img_as_ubyte, img_as_float
from skimage.measure import compare_psnr, compare_ssim

parser = argparse.ArgumentParser(description='Super resolution test')
parser.add_argument('--image_folder',
                    type=str,
                    required=True,
                    help="path to image folder containing test images")
parser.add_argument("--cuda",
                    action="store_true",
                    help="use cuda")
parser.add_argument("--model",
                    type=str,
                    default="checkpoint/model_epoch_50.pth",
                    help="path to saved model")
parser.add_argument('--scale',
                    type=int,
                    required=False,
                    nargs='+',
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
                    help="csv file containing specified set of images to load")
parser.add_argument('--save_test',
                    action='store_true',
                    help="save images used for testing to specified csv file")
parser.add_argument('--save_result',
                    action='store_true',
                    help="save PSNR results")

def main(args):
    opt = parser.parse_args(args)
    model = torch.load(opt.model,
                       map_location=lambda storage, loc: storage)["model"]
    
    if opt.save_result:
        results_file = open("results.csv", "w")
        results_csv = csv.writer(results_file)

    sampled_dataset = DatasetFromFolder(image_dir=opt.image_folder,
                                        sample_size=opt.sample_size,
                                        dataset_csv=opt.load_test)
    for scale in opt.scale:
        average_psnr = 0
        average_bicubic_psnr = 0
        average_ssim = 0
        average_bicubic_ssim = 0
        scale_transform = [('scale', scale)]

        for ground_truth in sampled_dataset:
            downsampled_image = sampled_dataset.transform(ground_truth,
                                                          scale_transform)
            y = get_interpolated_image(downsampled_image.split()[0], scale)
            
            input_image = Variable(ToTensor()(y),
                                   volatile=True).view(1, -1,
                                                       y.size[1], y.size[0])
            if opt.cuda:
                model = model.cuda()
                input_image = input_image.cuda()
            else:
                model = model.cpu()

            out_img = model(input_image)
                    
            out_img = out_img.cpu()
            out_img_y = out_img.data[0].numpy().clip(0, 1)
            out_img_y = out_img_y[0, :, :]
            
            height, width = out_img_y.shape
            ground_truth = get_center_crop(ground_truth, width, height)
            
            gt_img = img_as_float(np.asarray(ground_truth.split()[0]))
            predicted = img_as_float(out_img_y)
            interpolated = img_as_float(np.asarray(y))

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


        sample_size = len(sampled_dataset.image_filenames)
        print("Upscale factor = ", scale)
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
    main(sys.argv[1:])
