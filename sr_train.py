import argparse, os, sys
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from collections import deque
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataseth5 import DatasetFromHdf5
from dataset import DatasetFromFolder
from utils import get_center_crop, get_interpolated_image
from skimage import img_as_ubyte, img_as_float
from skimage.measure import compare_psnr

# Training settings
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
parser.add_argument('--scale',
                    type=int,
                    required=False,
                    nargs='+',
                    default=3,
                    help="downsampling factor")

moving_windows = {}
means = {}
test_results = {}
window_size = 5
c_limit = 5
delta_limit = 0.001
c = 0

def prepare_environment():
    global model, criterion, weights
    
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
            
def is_sufficiently_trained():
    global moving_windows, means, test_results, c
    global opt, model
    
    for scale in opt.scale:
        if not scale in moving_windows:
            moving_windows[scale] = deque(maxlen=window_size)
        if not scale in means:
            means[scale] = 0
            
        average_psnr = 0
        sampled_dataset = DatasetFromFolder(image_dir=opt.test_images,
                                            dataset_csv='validation.csv',
                                            sample_size=opt.sample_size,
                                            scale=scale)

        for ground_truth in sampled_dataset:
            downsampled_image = sampled_dataset.transform(ground_truth)
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
            average_psnr += psnr

        sample_size = len(sampled_dataset.image_filenames)
        moving_windows[scale].appendleft(average_psnr / sample_size)
        if len(moving_windows[scale]) == window_size:
            new_mean = np.mean(list(moving_windows[scale]))
            if means[scale]:
                delta = new_mean - means[scale]
                if delta < delta_limit:
                    c = c + 1
            means[scale] = new_mean
            if c > c_limit:
                test_results[scale] = True
                continue

        test_results[scale] = False

    # This should only ever be executed once
    if not os.path.isfile("validation.csv"):
        sampled_dataset.save_dataset("validation.csv")

    if False in test_results.values():
        return False
    else:
        os.remove("validation.csv")
        return True

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def main(args):
    global moving_windows, means, test_results, c
    global opt
    opt = parser.parse_args(args)

    prepare_environment()
    
    print("===> Training")
    print("===> Loading datasets")
    train_set = DatasetFromHdf5("train.h5")
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True)
    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, epoch)
        save_checkpoint(epoch)

        if opt.test_images and opt.scale:
            if is_sufficiently_trained():
                print("Stopped training at {} epochs".format(epoch))
                break

    # Reset global variables related to early termination
    moving_windows = {}
    means = {}
    test_results = {}
    c = 0

if __name__ == "__main__":
    main(sys.argv[1:])
