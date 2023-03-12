from pickle import FALSE
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
import sys
from utils.gaussian_downsample import gaussian_downsample,bicubic_downsample,bicubic_downsample_4
import torch
import math
import cv2

class Vid4(Dataset):  # load train dataset
    def __init__(self, data_dir,seq_name, scale, transform, seq_length=14,mode='mirror'):
        super(Vid4, self).__init__()
        self.mode = mode
        self.vid4_dir = data_dir + '/' + seq_name
        self.total_len = len(os.listdir(self.vid4_dir))
        self.seq_length = seq_length
        self.alist = self.split_seq() # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.scale = scale

        self.transform = transform  # To_tensor
        self.LRindex = [int(i) for i in range(seq_length) if i%2==0]
        
        self.HRindex = [0, 2, 3, 4, 6]
    def split_seq(self):
        ret_ls = []
        append_ix = reversed([ix for ix in range(self.total_len-(self.seq_length-self.total_len%self.seq_length)-1,self.total_len-1)])
        sub_num = math.ceil(self.total_len/self.seq_length)
        if self.mode=='mirror':

            for seq_ix in range(sub_num):
                tmp_ix_ls = []
                if seq_ix==sub_num-1 :
                    for sub_ix in range(self.total_len%self.seq_length):
                        tmp_ix_ls.append(seq_ix*self.seq_length+sub_ix)
                    tmp_ix_ls+=append_ix
                else:
                    for sub_ix in range(self.seq_length):
                        tmp_ix_ls.append(seq_ix*self.seq_length+sub_ix)
                ret_ls.append(tmp_ix_ls)
        elif self.mode=='overlap':
            for seq_ix in range(sub_num):
                tmp_ix_ls = []
                if seq_ix==sub_num-1:
                    tmp_ix_ls = [int(i) for i in range(self.total_len-self.seq_length,self.total_len)]

                else:
                    for sub_ix in range(self.seq_length):
                        tmp_ix_ls.append(seq_ix*self.seq_length+sub_ix)
                ret_ls.append(tmp_ix_ls)
        return ret_ls
    def load_img(self, image_path):

        HR = []
        for img_num in image_path:
            
            img_gt = Image.open(os.path.join(self.vid4_dir, str(img_num).zfill(8)+'.png')).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        
        return HR

    def __getitem__(self, index):

     
        GT = self.load_img(self.alist[index])

       
       
        GT = np.asarray(GT)  
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        h_raw = h
        w_raw = w

        if h%4!=0:
            padded_h1 = (4-h%4)//2
            padded_h2 = (4-h%4)-padded_h1
            h = h+(4-h%4)
    
        else:
            padded_h1,padded_h2 =0,0
        if w%4!=0:
            padded_w1 = (4-w%4)//2
            padded_w2 = (4-w%4)-padded_w1
            w = w+(4-w%4)
        else:
            padded_w1,padded_w2 =0,0

        crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
       

        GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)), mode='reflect')
      
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]

        self.scale = 1/4.0
        LR = bicubic_downsample(GT[:, self.LRindex],self.scale)

        LR = LR.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  

     
       
        return GT,LR,self.alist[index],crop_shape

    def __len__(self):
        return len(self.alist)  



def get_vid4_loader(data_root,scale,seq_name,seq_length, batch_size, num_workers):

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vid4(data_root, seq_name,scale, transform,seq_length,mode='overlap')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)




