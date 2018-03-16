import torch
import numpy as np
import argparse
import csv

from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torch.autograd import Variable
from dataset import DatasetFromFolder
#from utils import compute_psnr
from PIL import Image
from skimage import img_as_ubyte
from skimage.measure import compare_psnr
#from imresize import imresize

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

def get_interpolated_image(image, scale):
    width = image.size[0] * scale
    height = image.size[1] * scale
    composed_transform = Compose([Resize(size=(height, width),
                                         interpolation=Image.BICUBIC)])
    return composed_transform(image)

def main():
    opt = parser.parse_args()

    #model = torch.load(opt.model)["model"].module
    model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
    
    if opt.save_result:
        results_file = open("results.csv", "w")
        results_csv = csv.writer(results_file)

    for scale in opt.scale:
        average_psnr = 0
        average_bicubic_psnr = 0
        sampled_dataset = DatasetFromFolder(image_dir=opt.image_folder,
                                            sample_size=opt.sample_size,
                                            dataset_csv=opt.load_test,
                                            scale=scale)
        for ground_truth in sampled_dataset:
            downsampled_image = sampled_dataset.transform(ground_truth)
            y = get_interpolated_image(downsampled_image.split()[0], scale)

##            gt_img = np.asarray(ground_truth.split()[0])
##            downsampled_image = imresize(gt_img, scalar_scale=1/scale)
##            y = imresize(downsampled_image, scalar_scale=scale)
            
            input_image = Variable(ToTensor()(y)).view(1, -1, y.size[1],
                                                       y.size[0])
##            input_y = np.asarray(y).astype(float) / 255.
##            input_image = Variable(torch.from_numpy(input_y).float()).view(1, -1, input_y.shape[0], input_y.shape[1])
            
            if opt.cuda:
                model = model.cuda()
                input_image = input_image.cuda()
            else:
                model = model.cpu()

            out_img = model(input_image)
                    
            out_img = out_img.cpu()
            out_img_y = out_img.data[0].numpy().clip(0, 1)
            out_img_y = out_img_y[0,:,:]
            
            height, width = out_img_y.shape
            ground_truth = get_center_crop(ground_truth, width, height)
            gt_img = img_as_ubyte(np.asarray(ground_truth.split()[0]))

##            img = Image.fromarray(img_as_ubyte(out_img_y), 'L')
##            gt = Image.fromarray(gt_img, 'L')
##            img.show()
##            y.show()
##            gt.show()

            psnr = compare_psnr(gt_img, img_as_ubyte(out_img_y))
            bicubic_psnr = compare_psnr(gt_img, img_as_ubyte(np.asarray(y)))
            
            if opt.save_result:
                results_csv.writerow([sampled_dataset.current_image_file, scale, psnr])

            average_psnr += psnr
            average_bicubic_psnr += bicubic_psnr

        sample_size = len(sampled_dataset.image_filenames)
        print("Upscale factor = ", scale)
        print("Average PSNR for {} samples = {}".format(sample_size,
                                                        average_psnr / sample_size))
        print("Average bicubic PSNR for {} samples = {}".format(sample_size,
                                                                average_bicubic_psnr / sample_size))

    if opt.save_test:
        sampled_dataset.save_dataset("test.csv")

if __name__ == "__main__":
    main()
