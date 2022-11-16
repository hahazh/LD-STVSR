from matplotlib.pyplot import sca
from numpy.core.fromnumeric import size
import scipy.ndimage.filters as fi
import numpy as np
import torch.nn.functional as F
import torch
from torchvision.transforms.transforms import RandomAffine
import utils.core_bicubic as core
from utils.data_util import imresize_np
def gaussian_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code
    Args:
        x (Tensor, [C, T, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = gkern(13, 0.8)  # 13 and 0.8 for x2
    elif scale == 3:
        h = gkern(13, 1.2)  # 13 and 1.2 for x3
    elif scale == 4:
        h = gkern(13, 1.6)  # 13 and 1.6 for x4
    else:
        print('Invalid upscaling factor: {} (Must be one of 2, 3, 4)'.format(R))
        exit(1)

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W) # depth convolution (channel-wise convolution)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0

    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)

    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')
    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    # please keep the operation same as training.
    # if  downsample to 32 on training time, use the below code.
    x = x[:, :, 2:-2, 2:-2]
    # if downsample to 28 on training time, use the below code.
    #x = x[:,:,scale:-scale,scale:-scale]
    x = x.view(C, T, x.size(2), x.size(3))
    return x
def bicubic_downsample(x, scale=4):
    C, T, H, W = x.size()
    # x  = x.permute(1,0,2,3)
    ret_ls = []
    for i in range(T):
        img = x[:,i,:,:]
        img = core.imresize(img,sizes=(int(H*scale),int(W*scale)))
        # print('img shape',img.shape)
        ret_ls.append(img.unsqueeze(1))
    
    ret = torch.cat(ret_ls,dim=1)
    # print(ret.shape)
    return ret
def bicubic_downsample_4(x, scale=4):
    C, T, H, W = x.size()
    # x  = x.permute(1,0,2,3)
    ret_ls = []
    for i in range(T):
        img = x[:,i,:,:]
        img = core.imresize(img,sizes=(H//4,W//4))
        # print('img shape',img.shape)
        ret_ls.append(img.unsqueeze(1))
    
    ret = torch.cat(ret_ls,dim=1)
    # print(ret.shape)
    return ret
def bicubic_downsample_new(x, ratio):
    # print('shape is',shape)
    C, T, H, W = x.size()
    # x  = x.permute(1,0,2,3)
    ret_ls = []
    for i in range(T):
        img = x[:,i,:,:]
        # img = torch.from_numpy(imresize_np(img.numpy(),ratio)).permute(2,0,1)
        img =  core.imresize(img,sizes=(ratio,ratio))
        # print('img shape',img.shape)
        ret_ls.append(img.unsqueeze(1))
    
    ret = torch.cat(ret_ls,dim=1)
    # print(ret.shape)
    return ret