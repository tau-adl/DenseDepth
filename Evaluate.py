import time
import argparse
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from Model import FullModel
from Loss import ssim
from Data import getTrainingTestingData
from Utilities import AverageMeter, DepthNorm, colorize, saveImages


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--path', default="TrainedModel/EntireModel/model_batch_2_epochs_20.pt", type=str, help='model path')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    args = parser.parse_args()

    if args.epochs != 20:
        args.path = "TrainedModel/EntireModel/model_batch_2_epochs_{:}.pt".format(args.epochs)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=args.bs)

    model = torch.load(args.path)
    l1_criterion = nn.L1Loss()

    for i, sample_batched in enumerate(test_loader):
        torch.cuda.empty_cache()
        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        # Normalize depth
        depth_n = DepthNorm(depth)

        output = model(image)

        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
        loss = (1.0 * l_ssim) + (0.1 * l_depth)
        if i % 100 == 0:
            print("saving image..")
            print("loss depth = {:}, loss ssim = {:}, total loss = {:}".format(l_depth, l_ssim, loss))
            saveImages("{:}.png".format(i), output, image, depth_n)
    return


if __name__ == '__main__':
    main()
