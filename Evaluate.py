import argparse
import csv
import os
import time
import numpy as np
import torch
import utils

from metrics import AverageMeter, Result

from nyu import NYUDataset


def validate(val_loader, model, epoch, output_directory=""):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()
    
    eval_file = output_directory + '/evaluation.csv'
    f = open(eval_file, "w+")
    f.write("Max_Error,Depth,RMSE,GPU_TIME,Number_Of_Frame\r\n")
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # normalization for the model
        input = input[:, :, ::2, ::2]
        target = target[:, :, ::2, ::2]

        abs_err = (target.data - pred.data).abs().cpu()
        max_err_ind = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)

        max_err_depth = target.data[max_err_ind]
        max_err = abs_err[max_err_ind]
        f.write(f'{max_err}  {max_err_depth}   \r\n')

        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
        
        f.write(f'{max_err},{max_err_depth},{result.rmse:.2f},{gpu_time},{i+1}\r\n')

        # save 8 images for visualization
        skip = 10
        output_directory = os.path.abspath(os.path.dirname(__file__))

        if i == 0:
            img_merge = utils.merge_into_row_with_gt(input, target, pred, (target-pred).abs())
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row_with_gt(input, target, pred, (target-pred).abs())
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i + 1) % skip == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))
    f.close()
    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))
    return avg, img_merge


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--path', default="TrainedModel/EntireModel/model_batch_2_epochs_20.pt", type=str,
                        help='model path')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--data', metavar='DATA', default='nyudepthv2',
                        help='dataset:')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                        help='modality: ')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    args = parser.parse_args()

    # Data loading code
    print("=> creating data loaders...")
    valdir = os.path.join('..', 'data', args.data, 'val')

    val_dataset = NYUDataset(valdir, split='val', modality=args.modality)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")

    assert os.path.isfile(args.path), "=> no model found at '{}'".format(args.path)
    print("=> loading model '{}'".format(args.path))
    checkpoint = torch.load(args.path)
    if type(checkpoint) is dict:
        args.start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    else:
        model = checkpoint
        args.start_epoch = 0
    output_directory = os.path.dirname(os.path.abspath(__file__))
    print(output_directory)
    validate(val_loader, model, args.start_epoch, output_directory)
    return


if __name__ == '__main__':
    main()
