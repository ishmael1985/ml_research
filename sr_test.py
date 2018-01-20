import argparse
import torch
import math
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Compose, CenterCrop

from dataset import DatasetFromFolder

parser = argparse.ArgumentParser(description='PyTorch Super Resolution Example')
parser.add_argument('--image_folder', type=str, required=True, help='image_folder containing test images')
parser.add_argument('--sample_size', type=int, required=False, default=10, help="number of random samples to test")
parser.add_argument('--model', type=str, required=True, help='model file to use')
#parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument("--scale", default=3, type=int, help="downsampling factor, Default: 3")
opt = parser.parse_args()

print(opt)

def get_center_crop(image, width, height):
    composed_transform = Compose([CenterCrop(size=(height, width))])
    return composed_transform(image)
    
def compute_psnr(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

if __name__ == "__main__":
    average_psnr = 0
    sampled_dataset = DatasetFromFolder(image_dir=opt.image_folder,
                                        sample_size=opt.sample_size,
                                        scale=opt.scale)
    
    for ground_truth, downsampled_image in sampled_dataset:
        model = torch.load(opt.model)
        in_img_y = downsampled_image.split()[0]
        input = Variable(ToTensor()(in_img_y)).view(1, -1, in_img_y.size[1], in_img_y.size[0])

        #if opt.cuda:
        #    model = model.cuda()
        #    input = input.cuda()

        out = model(input)
        out = out.cpu()

        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y[0,:,:]
        
        height, width = out_img_y.shape[:2]
        ground_truth = get_center_crop(ground_truth, width, height)
        gt_img_y = np.asarray(ground_truth.split()[0])

        average_psnr += compute_psnr(out_img_y, gt_img_y)

    print("Upscale factor = ", opt.scale)
    print("Average PSNR for {} samples = {}".format(opt.sample_size, average_psnr / opt.sample_size))


