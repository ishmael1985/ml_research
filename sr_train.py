import argparse, os, sys
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import json

from collections import deque
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataseth5 import DatasetFromHdf5
from dataset import DatasetFromFolder
from utils import get_center_crop, get_interpolated_image
from skimage import img_as_float
from skimage.measure import compare_psnr

parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize",
                    type=int,
                    default=128,
                    help="Training batch size")
parser.add_argument("--nEpochs",
                    type=int,
                    default=50,
                    help="Number of epochs to train for")
parser.add_argument("--lr",
                    type=float,
                    default=0.1,
                    help="Learning Rate. Default=0.1")
parser.add_argument("--step",
                    type=int,
                    default=10,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda",
                    action="store_true",
                    help="Use cuda?")
parser.add_argument("--resume",
                    default="",
                    type=str,
                    help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch",
                    default=1,
                    type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip",
                    type=float,
                    default=0.4,
                    help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads",
                    type=int,
                    default=1,
                    help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum",
                    default=0.9,
                    type=float,
                    help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay",
                    "--wd",
                    default=1e-4,
                    type=float,
                    help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained',
                    default='',
                    type=str,
                    help='path to pretrained model (default: none)')
parser.add_argument("--gpus",
                    default="0",
                    type=str,
                    help="gpu ids (default: 0)")
parser.add_argument('--test_images',
                    type=str,
                    required=False,
                    help="path to image folder containing test images")
parser.add_argument('--sample_size',
                    type=int,
                    required=False,
                    default=10,
                    help="number of randomly selected images used for testing")


class VDSRModel:
    window_size = 5
    c_limit = 5
    delta_limit = 0.001
    
    def __init__(self, lr=0.1, step=10, clip=0.4, dataset='', sample_size=-1):
        # Training parameters
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.lr = lr
        self.step = step
        self.clip = clip

        # Validation parameters
        self.moving_windows = {}
        self.means = {}
        self.test_results = {}
        self.c = 0
        if dataset:
            self.test_dataset = DatasetFromFolder(image_dir=dataset,
                                                  sample_size=sample_size)
            with open("hdf5.json", "r") as config:
                self.upscale_factors = json.load(config)["upscale_factors"]

            for scale in self.upscale_factors:
                self.moving_windows[scale] = deque(maxlen=self.window_size)
                self.means[scale] = 0

        # Environment variables
        self.cuda = False

    def prepare_model(self, cuda=False, gpus='0', pretrained_model=''):
        self.cuda = cuda
        
        if self.cuda:
            print("=> use gpu id: '{}'".format(gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            if not torch.cuda.is_available():
                raise Exception(
                    "No GPU found or wrong gpu id, please run without --cuda")

        seed = random.randint(1, 10000)
        print("Random Seed: ", seed)
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

        cudnn.benchmark = True

        print("===> Building model")
        self.model = Net()
        self.criterion = nn.MSELoss(size_average=False)

        print("===> Setting GPU")
        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        else:
            self.model = self.model.cpu()

        # optionally copy weights from a checkpoint
        if pretrained_model:
            if os.path.isfile(pretrained_model):
                print("=> loading model '{}'".format(pretrained_model))
                weights = torch.load(pretrained_model)
                self.model.load_state_dict(weights['model'].state_dict())
            else:
                print("=> no model found at '{}'".format(pretrained_model))

    def initialize_optimizer(self, momentum=0.9, weight_decay=1e-4):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                   momentum=momentum, weight_decay=weight_decay)

    def get_resume_point(self, resume):
        start_epoch = 1
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(resume))

        return start_epoch

    def train_model(self, training_data_loader, epoch):
        # Adjust the learning rate to the initial LR decayed by 10 every 10 epochs
        lr = self.lr * (0.1 ** (epoch // self.step))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        print("Epoch = {}, lr = {}".format(
            epoch, self.optimizer.param_groups[0]["lr"]))

        self.model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1],
                                                         requires_grad=False)

            if self.cuda:
                input = input.cuda()
                target = target.cuda()

            loss = self.criterion(self.model(input), target)
            self.optimizer.zero_grad()
            loss.backward() 
            nn.utils.clip_grad_norm(self.model.parameters(), self.clip) 
            self.optimizer.step()

            if iteration%100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(training_data_loader), loss.data[0]))


    def validate_model(self):
        for scale in self.upscale_factors:
            average_psnr = 0
            scale_transform = [('scale', scale)]

            for ground_truth in self.test_dataset:
                downsampled_image = self.test_dataset.transform(ground_truth,
                                                                scale_transform)
                y = get_interpolated_image(downsampled_image.split()[0], scale)
                
                input_image = Variable(ToTensor()(y),
                                       volatile=True).view(1, -1,
                                                           y.size[1], y.size[0])

                if self.cuda:
                    input_image = input_image.cuda()

                out_img = self.model(input_image)
                
                out_img = out_img.cpu()
                out_img_y = out_img.data[0].numpy().clip(0, 1)
                out_img_y = out_img_y[0, :, :]
                
                height, width = out_img_y.shape
                ground_truth = get_center_crop(ground_truth, width, height)
                
                gt_img = img_as_float(np.asarray(ground_truth.split()[0]))
                predicted = img_as_float(out_img_y)
                interpolated = img_as_float(np.asarray(y))
                
                psnr = compare_psnr(gt_img, predicted)
                average_psnr += psnr

            sample_size = len(self.test_dataset.image_filenames)
            self.moving_windows[scale].appendleft(average_psnr / sample_size)
            if len(self.moving_windows[scale]) == self.window_size:
                new_mean = np.mean(list(self.moving_windows[scale]))
                if self.means[scale]:
                    delta = new_mean - self.means[scale]
                    if delta < self.delta_limit:
                        self.c = self.c + 1
                self.means[scale] = new_mean
                if self.c > self.c_limit:
                    self.test_results[scale] = True
                    continue

            self.test_results[scale] = False

        if False in self.test_results.values():
            return False
        else:
            return True

    def save_checkpoint(self, epoch):
        model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch , "model": self.model}
        if not os.path.exists("checkpoint/"):
            os.makedirs("checkpoint/")

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


def main(args):
    opt = parser.parse_args(args)

    sr_model = VDSRModel(lr=opt.lr, step=opt.step, clip=opt.clip,
                         dataset=opt.test_images, sample_size=opt.sample_size)

    sr_model.prepare_model(opt.cuda, opt.gpus, opt.pretrained)
    
    if opt.resume:
        start_epoch = sr_model.get_resume_point(opt.resume)
    else:
        start_epoch = opt.start_epoch

    print("===> Setting Optimizer")
    sr_model.initialize_optimizer(opt.momentum, opt.weight_decay)

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("train.h5")
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True)
    print("===> Training")        
    for epoch in range(start_epoch, opt.nEpochs + 1):
        sr_model.train_model(training_data_loader, epoch)
        sr_model.save_checkpoint(epoch)

        if opt.test_images and sr_model.validate_model():
            print("Stopped training at {} epochs".format(epoch))
            break
        
if __name__ == "__main__":
    main(sys.argv[1:])
