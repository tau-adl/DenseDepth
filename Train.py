import argparse
import datetime
import time

import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from Data import *
from Model import FullModel
from Utilities import AverageMeter, DepthNorm, colorize, logEpoch


def main():
    modality_names = CustomDataLoader.modality_names
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--path', default='../data/nyudepthv2', help='path')
    parser.add_argument('--data', default='nyudepthv2', help='model')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    args = parser.parse_args()

    # Create model
    torch.cuda.empty_cache()
    model = FullModel().cuda()
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = args.batch_size
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = createDataLoaders(args)

    torch.cuda.empty_cache()
    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-batch_size{}'.format(prefix, args.lr, args.epochs, args.batch_size),
                           flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()

    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, (image, depth) in enumerate(train_loader):
            optimizer.zero_grad()
            # Prepare sample and target
            image, depth = image.cuda(), depth.cuda()
            torch.cuda.synchronize()
            # Normalize depth
            depth_n = DepthNorm(depth)
            # Predict
            output = model(image)
            # Compute the loss
            criterion = criteria.MaskedL1Loss().cuda()
            loss = criterion.forward(output, depth_n)
            # l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

            # loss = (1.0 * l_ssim) + (0.1 * loss)

            # Update step
            # losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            result = Result()
            result.evaluate(output.data, depth_n.data)
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            # Log progress
            niter = epoch * N + i

            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss} RMSE {rmse}'
                      .format(epoch, i, N, batch_time=batch_time, loss=loss, rmse=result.rmse, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter)

        # Record epoch's intermediate results
        logEpoch(epoch, losses.val, "TrainOutput.txt")
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
    # save the final model
    base_dir = "TrainedModel"
    entire_model_dir = os.path.join(base_dir, "EntireModel")
    model_param_dir = os.path.join(base_dir, "ModelParameters")
    if not os.path.exists("TrainedModel"):
        os.mkdir(base_dir)
        if not os.path.exists(os.path.join(base_dir, "EntireModel")):
            os.mkdir(entire_model_dir)
        if not os.path.exists(os.path.join(base_dir, "ModelParameters")):
            os.mkdir(model_param_dir)
    torch_model_name = "model_batch_{:}_epochs_{:}.pt".format(args.batch_size, args.epochs)
    torch.save(model, os.path.join(entire_model_dir, torch_model_name))
    torch.save(model.state_dict(), os.path.join(model_param_dir, torch_model_name))
    print('done :-)')


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    (image, depth) = next(iter(sequential))
    image = torch.autograd.Variable(image.cuda())
    depth = torch.autograd.Variable(depth.cuda(non_blocking=True))
    if epoch == 0:
        writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0:
        writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = model(image)
    depth = DepthNorm(depth)
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff',
                     colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output


if __name__ == '__main__':
    main()
