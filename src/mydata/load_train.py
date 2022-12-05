from pickle import FALSE
from torch.utils.data import DataLoader, Dataset
import numpy as np
# from mydata.dataset_config import Train_comfig
import os

import random
from PIL import Image
from torchvision import transforms
import sys

from utils.gaussian_downsample import gaussian_downsample,bicubic_downsample,bicubic_downsample_4
# import config as config
import torch

import math
import cv2


class Vimeo_SepTuplet_ST(Dataset):  # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform, load_mode='train'):
        super(Vimeo_SepTuplet_ST, self).__init__()
        self.load_mode = load_mode
        alist = [line.rstrip() for line in open(os.path.join(image_dir,
                                                             file_list))]  # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.image_filenames = [os.path.join(image_dir, x) for x in alist]  # get image path list
        # fake_num = 40
        # self.image_filenames = [str(i) for i in range(fake_num)]
        self.scale = scale

        self.transform = transform  # To_tensor
        self.data_augmentation = data_augmentation  # flip and rotate
        self.LRindex = [0, 2, 4, 6]
        self.mindex = [1,3,5]
        self.HRindex = [0, 2, 3, 4, 6]

    def train_process(self, GT, flip_h=True, rot=True, flip_v=True, converse=True):  # input:list, target:PIL.Image
        if random.random() < 0.5 and flip_v:
            GT = [LR[::-1, :, :].copy() for LR in GT]
        if random.random() < 0.5 and flip_h:
            GT = [LR[:, ::-1, :].copy() for LR in GT]
        if rot and random.random() < 0.5:
            GT = [LR.transpose(1, 0, 2).copy() for LR in GT]
        if converse and random.random() < 0.5:
            GT = GT.copy()[::-1]
        return GT

    def load_img(self, image_path, scale):

        def crop_256(img_list, H, W):
            crop_size = 256
            rnd_h = random.randint(0, H - crop_size)
            rnd_w = random.randint(0, W - crop_size)

            img_gt_crop = [v[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :] for v in img_list]
            return img_gt_crop

        HR = []
        for img_num in range(7):
            # GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            # h,w,c = 720,1280,3
            # print(os.path.join(image_path,'im{}.png'.format(img_num+1)))
            img_gt = Image.open(os.path.join(image_path, 'im{}.png'.format(img_num + 1))).convert('RGB')

            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        H, W = HR[0].shape[:2]
        if self.load_mode == 'train':
            HR = crop_256(HR, H, W)
        # else:
        #     #县令史加上
        #     HR = crop_256(HR,H,W)
        # 对每个group进行crop

        return HR

    def __getitem__(self, index):

        # GT shape 长度为7的list
        GT = self.load_img(self.image_filenames[index], self.scale)

        if self.load_mode == 'train':
            if self.data_augmentation:
                GT = self.train_process(GT)

        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
        # if self.scale == 4:
        #     GT = np.lib.pad(GT, pad_width=((0,0),(2*self.scale,2*self.scale),(2*self.scale,2*self.scale),(0,0)), mode='reflect')
        # for i in range(len(GT)):
        #     # print(GT[i].shape)
        #     cv2.imwrite('./test_in/GT/'+str(i)+'.png',GT[i])
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
      
        if self.load_mode=='test':
            if h%64!=0:
                padded_h1 = (64-h%64)//2
                padded_h2 = (64-h%64)-padded_h1
                h = h+(64-h%64)
    
            else:
                padded_h1,padded_h2 =0,0
            if w%64!=0:
                padded_w1 = (64-w%64)//2
                padded_w2 = (64-w%64)-padded_w1
                w = w+(64-w%64)
                
            else:
                padded_w1,padded_w2 =0,0
        # padded_h1,padded_h2,padded_w1,padded_w2 =  padded_h1+64,padded_h2+64,padded_w1+64,padded_w2+64
        # h,w = h+128,w+128
            crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
            if self.scale == 4:
                GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)), mode='reflect')
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        # LR = gaussian_downsample(GT, self.scale)
        self.scale = 1/4.0
        LR = bicubic_downsample_4(GT,self.scale)

        LR_in = LR[:, self.LRindex]
        LR_mid = LR[:, self.mindex ]

        LR_in = LR_in.permute(1, 0, 2, 3)
        LR_mid = LR_mid.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        long = True
        if long:

            GT_long = torch.cat((GT,torch.flip(GT[:-1],dims = [0]) ),dim=0) 
            LR_long = torch.cat((LR_in,torch.flip(LR_in[:-1],dims = [0]) ),dim=0)
            LR_mid_long = torch.cat((LR_mid,torch.flip(LR_mid,dims = [0]) ),dim=0)
            GT,LR_in,LR_mid = GT_long,LR_long,LR_mid_long
        
        if self.load_mode == 'test':
        
            return GT,LR_in,self.image_filenames[index],crop_shape
        elif self.load_mode == 'mask':
            
            return GT,LR_in,LR_mid
        else:
            return GT,LR_in

    def __len__(self):
        return len(self.image_filenames)  # total video number. not image number


def get_loader(mode, data_root, train_file_name, test_file_name, batch_size, shuffle, num_workers, test_mode=None):
    scale = 4
    if mode == 'train':
        data_augmentation = True
        file_list = train_file_name
    else:
        data_augmentation = False
        file_list = test_file_name
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vimeo_SepTuplet_ST(data_root, scale, data_augmentation, file_list, transform, load_mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)




class Vid4(Dataset):  # load train dataset
    def __init__(self, data_dir,seq_name, scale, transform, seq_length=14,mode='mirror'):
        super(Vid4, self).__init__()
        self.mode = mode
        self.vid4_dir = data_dir + '/' + seq_name
        self.total_len = len(os.listdir(self.vid4_dir))
        self.seq_length = seq_length
        self.alist = self.split_seq() # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
      
        # fake_num = 40
        # self.image_filenames = [str(i) for i in range(fake_num)]
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
            # GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            # h,w,c = 720,1280,3
            # print(os.path.join(image_path,'im{}.png'.format(img_num+1)))
            img_gt = Image.open(os.path.join(self.vid4_dir, str(img_num).zfill(8)+'.png')).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        
        return HR

    def __getitem__(self, index):

        # GT shape 长度为7的list
        # return self.alist[index]
        GT = self.load_img(self.alist[index])

       
       
        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        h_raw = h
        w_raw = w
        # print('raw shape',GT.shape)
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
        # padded_h1,padded_h2,padded_w1,padded_w2 =  padded_h1+64,padded_h2+64,padded_w1+64,padded_w2+64
        # h,w = h+128,w+128
        crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
        # print('padded_h1 ',padded_h1)
        # print('padded h2', padded_h2)
        # print('padded_w1 ',padded_w1)
        # print('padded w2', padded_w2)
    
        # if self.scale == 4.0:
        GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)), mode='reflect')
      
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        # LR = gaussian_downsample(GT[:, self.LRindex], self.scale)
        self.scale = 1/4.0
        LR = bicubic_downsample(GT[:, self.LRindex],self.scale)

        LR = LR.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]

        long = False
        if long:

            GT_long = torch.cat((GT,torch.flip(GT[:-1],dims = [0]) ),dim=0) 
            LR_long = torch.cat((LR,torch.flip(LR[:-1],dims = [0]) ),dim=0)
            GT,LR = GT_long,LR_long
       
        return GT,LR,self.alist[index],crop_shape

    def __len__(self):
        return len(self.alist)  # total video number. not image number



class REDS(Dataset):  # load train dataset
    def __init__(self, data_dir, scale, transform, seq_length=14,mode='mirror',load_mode = 'train'):
        super(REDS, self).__init__()
        self.mode = mode
        self.load_mode = load_mode
        if load_mode != 'test':
            self.reds_seq_dir = sorted([os.path.join(data_dir,each.strip()) for each in open(os.path.join(data_dir,'train_list.txt')).readlines()])
        else:
            self.reds_seq_dir = sorted([os.path.join(data_dir,each.strip()) for each in open(os.path.join(data_dir,'test_list.txt')).readlines()])
        
        # length of reds
        self.total_len = 100
        self.seq_length = seq_length
        self.alist = self.split_seq() # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.total_list = self.generate_total_list()
       
        # fake_num = 40
        # self.image_filenames = [str(i) for i in range(fake_num)]
        self.scale = scale

        self.transform = transform  # To_tensor
        self.LRindex = [int(i) for i in range(seq_length) if i%2==0]
        self.mindex = [int(i) for i in range(seq_length) if i%2!=0]
        
        self.HRindex = [0, 2, 3, 4, 6]
    def train_process(self, GT, flip_h=True, rot=True, flip_v=True, converse=True):  # input:list, target:PIL.Image
        if random.random() < 0.5 and flip_v:
            GT = [LR[::-1, :, :].copy() for LR in GT]
        if random.random() < 0.5 and flip_h:
            GT = [LR[:, ::-1, :].copy() for LR in GT]
        if rot and random.random() < 0.5:
            GT = [LR.transpose(1, 0, 2).copy() for LR in GT]
        if converse and random.random() < 0.5:
            GT = GT.copy()[::-1]
        return GT
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
    def generate_total_list(self):
        total_ls = []
        for f_dir in self.reds_seq_dir:
            for s_dir in self.alist:
                tmp_ls  = []
                for im_d in s_dir:
                    im_d = str(im_d).zfill(8)+'.png'
                    this_dir = os.path.join(f_dir,im_d)
                    tmp_ls.append(this_dir)
                    # print(this_dir)
                total_ls.append(tmp_ls)
            

        return total_ls
        # print(len(total_ls))
            
        # print(total_ls[-1])
    def load_img(self, image_path):
        def crop_256(img_list, H, W):
            crop_size = 256
            rnd_h = random.randint(0, H - crop_size)
            rnd_w = random.randint(0, W - crop_size)
            # rnd_h = 300
            # rnd_w = 300
            img_gt_crop = [v[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :] for v in img_list]
            return img_gt_crop
        HR = []
        for img_name in image_path:
            W,H = 256,256
            # GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            # h,w,c = 720,1280,3
            # print(os.path.join(image_path,'im{}.png'.format(img_num+1)))
            img_gt = Image.open(img_name).convert('RGB')
            # if self.load_mode == 'test':
            #     img_gt = np.asarray(img_gt)
            #     img_gt = cv2.resize(src=img_gt,dsize = (W,H))
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
       
        H, W = HR[0].shape[:2]
        if self.load_mode != 'test':
            HR = crop_256(HR, H, W)

        # print('img raw shape',HR[0].shape)
        return HR

    def __getitem__(self, index):

        # GT shape 长度为7的list
        # return self.alist[index]
        GT = self.load_img(self.total_list[index])
        
        if self.load_mode!='test':
             GT = self.train_process(GT)
        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
      
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        if self.load_mode=='test':
            h_raw = h
            w_raw = w
            # print('raw shape',GT.shape)
            if h%32!=0:
                padded_h1 = (32-h%32)//2
                padded_h2 = (32-h%32)-padded_h1
                h = h+(32-h%32)
        
            else:
                padded_h1,padded_h2 =0,0
            if w%32!=0:
                padded_w1 = (32-w%32)//2
                padded_w2 = (32-w%32)-padded_w1
                w = w+(32-w%32)
            else:
                padded_w1,padded_w2 =0,0
            padded_h1,padded_h2,padded_w1,padded_w2 =  padded_h1+16,padded_h2+16,padded_w1+16,padded_w2+16
            h,w = h+32,w+32
            crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
            if self.scale == 4:
                GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)), mode='reflect')
                # print(GT.shape)

        # print(GT.shape)
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        self.scale = 1/4.0
        LR = bicubic_downsample(GT, self.scale)
        LR_in =  LR[:, self.LRindex]
        LR_m = LR[:,self.mindex]
        LR_in =  LR_in.permute(1, 0, 2, 3)
        LR_m = LR_m.permute(1, 0, 2, 3)
      
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        # print(GT.shape)
        if self.load_mode=='test':
            return GT,LR_in,self.total_list[index],crop_shape
        elif self.load_mode=='mask':
            return GT,LR_in,LR_m,self.total_list[index]
        else:
            return GT,LR_in,self.total_list[index]

    def __len__(self):
        return len(self.total_list)  # total video number. not image number


    # reds = REDS()
















class REDS_test(Dataset):  # load train dataset
    def __init__(self, data_dir, scale, transform, seq_length=14,mode='mirror',load_mode = 'train'):
        super(REDS_test, self).__init__()
        self.mode = mode
        self.load_mode = load_mode
        if load_mode != 'test':
            self.reds_seq_dir = sorted([os.path.join(data_dir,each.strip()) for each in open(os.path.join(data_dir,'test_list.txt')).readlines()])
        else:
            self.reds_seq_dir = sorted([os.path.join(data_dir,each.strip()) for each in open(os.path.join(data_dir,'test_list.txt')).readlines()])
        
        # length of reds
        self.total_len = 100
        self.seq_length = seq_length
        self.alist = self.split_seq() # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.total_list = self.generate_total_list()
       
        # fake_num = 40
        # self.image_filenames = [str(i) for i in range(fake_num)]
        self.scale = scale

        self.transform = transform  # To_tensor
        self.LRindex = [int(i) for i in range(seq_length) if i%2==0]
        self.mindex = [int(i) for i in range(seq_length) if i%2!=0]
        
        self.HRindex = [0, 2, 3, 4, 6]
    def train_process(self, GT, flip_h=True, rot=True, flip_v=True, converse=True):  # input:list, target:PIL.Image
       
        return GT
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
    def generate_total_list(self):
        total_ls = []
        for f_dir in self.reds_seq_dir:
            for s_dir in self.alist:
                tmp_ls  = []
                for im_d in s_dir:
                    im_d = str(im_d).zfill(8)+'.png'
                    this_dir = os.path.join(f_dir,im_d)
                    tmp_ls.append(this_dir)
                    # print(this_dir)
                total_ls.append(tmp_ls)
            

        return total_ls
        # print(len(total_ls))
            
        # print(total_ls[-1])
    def load_img(self, image_path):
        crop_W,crop_H = 1024,512
        def crop_256(img_list, H, W):
            
            rnd_h = random.randint(0, H - crop_H)
            rnd_w = random.randint(0, W - crop_W)
            rnd_h = 0
            rnd_w = 0
            img_gt_crop = [v[rnd_h:rnd_h + crop_H, rnd_w:rnd_w + crop_W, :] for v in img_list]
            return img_gt_crop
        HR = []
        for img_name in image_path:
           
            # GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
            # h,w,c = 720,1280,3
            # print(os.path.join(image_path,'im{}.png'.format(img_num+1)))
            img_gt = Image.open(img_name).convert('RGB')
            
            # if self.load_mode == 'test':
            # img_gt = np.asarray(img_gt)
            # img_gt = cv2.resize(src=img_gt,dsize = (W,H))
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
       
        H, W = HR[0].shape[:2]
        # if self.load_mode != 'test':
        #     HR = crop_256(HR, H, W)

        # print('img raw shape',HR[0].shape)
        return HR

    def __getitem__(self, index):

        # GT shape 长度为7的list
        # return self.alist[index]
        GT = self.load_img(self.total_list[index])
        
        if self.load_mode!='test':
             GT = self.train_process(GT)
        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
      
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
     
        h_raw = h
        w_raw = w
        # print('raw shape',GT.shape)
        if h%128!=0:
            padded_h1 = (128-h%128)//2
            padded_h2 = (128-h%128)-padded_h1
            h = h+(128-h%128)
    
        else:
            padded_h1,padded_h2 =0,0
        if w%128!=0:
            padded_w1 = (128-w%128)//2
            padded_w2 = (128-w%128)-padded_w1
            w = w+(128-w%128)
        else:
            padded_w1,padded_w2 =0,0
        # padded_h1,padded_h2,padded_w1,padded_w2 =  padded_h1+64,padded_h2+64,padded_w1+64,padded_w2+64
        # h,w = h+128,w+128
        crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
        if self.scale == 4:
            # GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)),mode='constant',constant_values=0)
             GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)),mode='reflect')
 # (512, 512, 1)

        print(GT.shape)

        # print(GT.shape)
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
  
        LR = bicubic_downsample_4(GT, self.scale)
        LR_in =  LR[:, self.LRindex]
        LR_m = LR[:,self.mindex]
        LR_in =  LR_in.permute(1, 0, 2, 3)
        LR_m = LR_m.permute(1, 0, 2, 3)
      
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        # print(GT.shape)
        if self.load_mode=='test':
            return GT,LR_in,self.total_list[index],crop_shape
        elif self.load_mode=='mask':
            return GT,LR_in,LR_m,self.total_list[index]
        else:
            return GT,LR_in,self.total_list[index]

    def __len__(self):
        return len(self.total_list)  # total video number. not image number


def get_loader(mode, data_root, train_file_name, test_file_name, batch_size, shuffle, num_workers, test_mode=None):
    scale = 4
    if mode == 'train':
        data_augmentation = True
        file_list = train_file_name
    else:
        data_augmentation = False
        file_list = test_file_name
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vimeo_SepTuplet_ST(data_root, scale, data_augmentation, file_list, transform, load_mode=mode)
    # return dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def get_vid4_loader(data_root,scale,seq_name,seq_length, batch_size, num_workers):

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vid4(data_root, seq_name,scale, transform,seq_length,mode='overlap')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def get_REDS_loader(data_root,scale,seq_length, batch_size, num_workers,shuffle=True,mode='overlap',load_mode='train'):

    transform = transforms.Compose([transforms.ToTensor()])
   
    reds = REDS(data_root,scale,transform,seq_length,mode,load_mode)
    return DataLoader(reds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)



def get_REDS_loader_test(data_root,scale,seq_length, batch_size, num_workers,shuffle=True,mode='overlap',load_mode='test'):

    # data_root = '/home/zhangyuantong/dataset/REDS/mysplit/'
    # scale = 4
    # seq_length = 7
    transform = transforms.Compose([transforms.ToTensor()])
    # mode = 'overlap'
    # load_mode = 'train'
    # f = open('test_list.txt','w')
    # for each in sorted(os.listdir(train_data_dir)):
    #     f.write(os.path.join('test',each)+'\n')
    reds = REDS_test(data_root,scale,transform,seq_length,mode,load_mode)
    return DataLoader(reds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)







if __name__=='__main__':
    
    train_data_dir = '/home/zhangyuantong/dataset/REDS/mysplit/'
    scale = 4
    seq_length = 7
    transform = transforms.Compose([transforms.ToTensor()])
    mode = 'overlap'
    load_mode = 'test'
    # f = open('test_list.txt','w')
    # for each in sorted(os.listdir(train_data_dir)):
    #     f.write(os.path.join('test',each)+'\n')
    test = REDS(train_data_dir,scale,transform,seq_length,mode,load_mode)
    for each in test:
      
         GT,LR,index,crop_shape = each
         print(GT.shape)
         print(LR.shape)
        #  print(LR_m.shape)
         print(crop_shape)
         
         break
    # for each in test:
    #     img_HR_gt,LR_gt = each
    #     print(img_HR_gt.shape)
    #     print(LR_gt.shape)
    #
    #     break
    # args, unparsed = config.get_args()
    # train_loader = get_loader('test', args.vimeo_root, args.train_file_name, args.test_file_name, args.batch_size,
    #                           shuffle=True,
    #                           num_workers=args.num_workers)

    # for ix,data in enumerate(train_loader):
    #     img_HR_gt, LR_gt = data
    #     print(img_HR_gt.shape)
    #     print(LR_gt.shape)
    #     break
    # seq_name_tuple = ('calendar','city','foliage','walk')
    # data_root =  '/home/zhangyuantong/dataset/Vid4/GT/'
    # scale = 4
    # seq_length = 15
    # transform = transforms.Compose([transforms.ToTensor()])
    # mode = 'overlap'
    # for i in range(4):
    #     test_vid4 = Vid4(data_root, seq_name_tuple[i],scale, transform,seq_length,mode='overlap')
    #     print(' now %s len %d '%(seq_name_tuple[i],len(test_vid4)))
    #     for each in test_vid4:
    #         GT,LR = each
    #         print(LR.shape)
    #         print(GT.shape)
        # print(each)
       