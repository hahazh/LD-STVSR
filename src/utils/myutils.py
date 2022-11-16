from .pytorch_msssim import ssim_matlab as calc_ssim
import math
import os
import torch
import shutil
import numpy as np
from collections import OrderedDict
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims
def calculate_psnr(prediction, target):
    # prediction and target have range [0, 255]
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    odd_psnr = 0
    even_psnr = 0
    odd_cnt = 0
    even_cnt = 0
    for b in range(gt.size(0)):
        avg = 0
        for t in range(gt.size(1)):
            
            psnr = calc_psnr(output[b][t], gt[b][t])
            # print('pnsr: ',psnr)
            avg+=psnr
            # print(str(t)+' psnr : '+str(psnr))
            psnrs.update(psnr)
            if t%2==0:
                odd_psnr+=psnr
                odd_cnt+=1
            else:
                even_psnr+=psnr
                even_cnt+=1
    print(f'odd psnr: {odd_psnr/odd_cnt :.3f} even psnr: {even_psnr/even_cnt :.3f}')

            # ssim = calc_ssim(output[b][t].unsqueeze(0).clamp(0,1), gt[b][t].unsqueeze(0).clamp(0,1) , val_range=1.)
            # ssims.update(ssim)
        # print('avg   :',avg/7)
        # return avg/7
def eval_metrics_test(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        avg = 0
        for t in range(7):
            
            psnr = calc_psnr(output[b][t], gt[b][t])
            avg+=psnr
            print(str(t)+' psnr : '+str(psnr))
            psnrs.update(psnr)

            ssim = calc_ssim(output[b][t].unsqueeze(0).clamp(0,1), gt[b][t].unsqueeze(0).clamp(0,1) , val_range=1.)
            ssims.update(ssim)
        print('avg   :',avg/7)
        return avg/7
def eval_metrics_vid(output, gt, psnrs, ssims,length,tag_ls):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        avg = 0
        avg_ssim = 0
        for t in range(length):
            print(tag_ls[t])
            psnr = calc_psnr(output[b][t], gt[b][t])
            avg+=psnr
            print(str(t)+' psnr : '+str(psnr))
            psnrs.update(psnr)

            ssim = calc_ssim(output[b][t].unsqueeze(0).clamp(0,1), gt[b][t].unsqueeze(0).clamp(0,1) , val_range=1.)
            avg_ssim+=ssim
            print(str(t)+' ssim : '+str(ssim))
            ssims.update(ssim)
        print('avg   :',avg/length)
        print('avg   :',avg_ssim/length)
        return avg/length

def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def save_checkpoint(state, directory, is_best, exp_name, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory , filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory , 'model_best.pth'))

def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar('Loss/%s/%s' % mode, loss, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)
