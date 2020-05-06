import matplotlib
import matplotlib.cm
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
from PIL import Image

cmap = plt.cm.viridis

def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def logEpoch(num_epoch, loss, file_name):
    f = open(file_name, "a+")
    s = "Epoch [{:}]\t loss value is: {:}\n".format(num_epoch, loss)
    f.write("******************************************\n")
    f.write(s)
    f.close()
    return

def toMultichannel(img):
    # img = img[:,:,0]
    return np.stack((img,img,img), axis=2)

def transformToImageScale(im):
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')

def createImages(pred, img, gt, is_colormap=True, is_rescale=True):
    img = img.cpu().numpy().squeeze()
    img = img.transpose((1, 2, 0))
    gt = gt.cpu().numpy().squeeze()
    pred = pred.cpu().detach().numpy().squeeze()

    plasma = plt.get_cmap('plasma')
    shape = (pred.shape[0], pred.shape[1], 3)
    imgs = []
    x = resize(img, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
    x = transformToImageScale(x)
    imgs.append(Image.fromarray(x))
    x = resize(gt, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
    x = transformToImageScale(x)
    imgs.append(Image.fromarray(x))

    x = toMultichannel(pred)
    x = transformToImageScale(x)
    imgs.append(Image.fromarray(x))

    all_images = imgs

    return all_images

def saveImages(filename, pred, img, gt, is_colormap=True, is_rescale=False):
    montage =  createImages(pred, img, gt, is_colormap, is_rescale)
    widths, heights = zip(*(i.size for i in montage))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in montage:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(filename)
    return
