import os

import time
import copy
import shutil
import random
import pdb
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from arbitrary_shape_ps1 import Unstrained_ST as Unstrained_ST_ps1

import utils.myutils as utils

# from bak.best_arg_now import IconSTSR as IconSTSR_if
from mydata.load_train import get_vid4_loader


# from icon_STSR_IFnet  import IconSTSR as IconSTSR_if


##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"]='0'

cwd = os.getcwd()
seq_name_tuple = ('calendar','city','foliage','walk')
seq_length_tuple = (41,33,49,47)
data_root =  '/home/zhangyuantong/dataset/Vid4/GT/'
scale = 4
seq_length = 31
batch_size = 1
mode = 'mirror'

# from model.FLAVR_arch import UNet_3D_3D




# model = torch.nn.DataParallel(model).to(device)



def test():
    # from mydata.load_data_arbitrary import get_vid4_loader
    # from arbitrary_pcd_forward import IconSTSR as pcd_forward
    # from arbitrary_shape import IconSTSR as pcd_forward_shape
    # from arbitrary_time_spy import IconSTSR as pcd_forward_time
    from model_fix import Unstrained_ST
    # for l in range(38,44):
    out_dir = '/home/zhangyuantong/code/ST-SR/unstrained_STSR/out_img/unstrained_ST/vid'
    load_from = '/home/zhangyuantong/code/MyOpenSource/LD-STVSR/weight/model_fix.pth'
    # args.load_from = '/home/zhangyuantong/code/ST-SR/ThreeBArbitrary/save/saved_models_final/vimeo90K_septuplet/arbitrary_ST_fix_flow_simple/50_edvr.pth'


    # model_dict = model.state_dict()
    scale,scale2 = 4.0,4.0
    model = Unstrained_ST(is_train=True,scale = scale,scale2 = scale2)
    model = model.cuda()
    print("#params" , sum([p.numel() for p in model.parameters()]))
    model_dict =torch.load(load_from)
    # model_dict =  torch.load(load_from)["state_dict"]
    model.load_state_dict(model_dict , strict=True)


    # model = IconSTSR_raw()
    # model = IconSTSR_if()
    # model = softnet_f(insert_shape=(192,192))
    
    losses, psnrs, ssims = utils.init_meters('1*CharbonnierLoss')
    model.eval()

    psnr_list = []
    total = 0
    cnt = 0
    
    with torch.no_grad():
        for q in range(4):
            tag = seq_name_tuple[q]
            seq_length = seq_length_tuple[q]
            vid4_dataloader = get_vid4_loader(data_root,scale,tag,seq_length,batch_size,0)
            for i, (gt_image,images,tag_ix,crop_shape) in enumerate(tqdm(vid4_dataloader)):
                print(i)
                # tag = tag[0].split('/')[-2]+'/'+tag[0].split('/')[-1]
                
                images = images.cuda()
                gt = gt_image.cuda()

                torch.cuda.synchronize()
                start_time = time.time()
                out= model(images,scale,scale2)
                torch.cuda.synchronize()
                this_time = time.time() - start_time
                print(this_time)
             
                img_p = out_dir+'/'+str(scale)+'/'+tag
                if   not os.path.exists(img_p):
                    os.makedirs(img_p)
            
                # print(out.shape)
                out_cp = out.squeeze(0)
                # out_compile = (out_cp[:7,:,:,:] +torch.flip(out_cp[6:,:,:,:],dims=[1]) )/2.0
                
                # out_cp = torch.stack([out_cp[-1],out_compile[1],out_cp[-3],out_compile[3],out_cp[-5],out_compile[5],out_cp[-7]],dim=0) 
                # out_cp = out_compile
                # out[0,:7,:,:,:]  = out_cp
                for j in range(seq_length):
                    img = out_cp[j].permute(1,2,0).detach().cpu().clamp(0,1).numpy()*255.0
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    x1,x2,y1,y2 = crop_shape
                    
                    img = img[x1:x2,y1:y2,:]
                    # print('write shape',img.shape)
                    cv2.imwrite(img_p+'/'+str(int(tag_ix[j])).zfill(8)+'uns.png',img)
                    cnt+=1
                print(img_p)
                # total+= utils.eval_metrics_vid(out, gt, psnrs, ssims,seq_length)
                # print(tag)
                # print('now avg: ',total/(cnt))
                

            # print("PSNR: %f, SSIM: %fn" %
                # (psnrs.avg, ssims.avg))
    # print("Time , " , 1/(sum(time_taken)/cnt))

    return psnrs.avg
def cal_cri():
      
    from basicsr.metrics import calculate_psnr, calculate_ssim
    from basicsr.utils.matlab_functions import bgr2ycbcr
    GT_base_dir = data_root
  
    pred_base_dir = '/home/zhangyuantong/code/MyOpenSource/LD-STVSR/out/vid4'
    total_psnr = 0
    total_ssim = 0
    cnt = 0
    for q in range(4):
        tag = seq_name_tuple[q]
        img_gt_dir_ls = sorted([os.path.join(GT_base_dir,tag,each) for each in os.listdir(os.path.join(GT_base_dir,tag))])
        img_pred_dir_ls = sorted([os.path.join(pred_base_dir,tag,each) for each in os.listdir(os.path.join(pred_base_dir,tag))])
        seq_psnr = 0
        seq_ssim = 0
        print("scene: %s "%(tag))
        for ix,data in enumerate(img_gt_dir_ls):
            print("%d / %d "%(ix,len(img_gt_dir_ls)))
            GT_img = cv2.imread(img_gt_dir_ls[ix])
            pred_img = cv2.imread(img_pred_dir_ls[ix])
            prediction_Y = bgr2ycbcr(GT_img,y_only=True)

            target_Y = bgr2ycbcr(pred_img,y_only=True)
            prediction_Y = prediction_Y * 255
            target_Y = target_Y * 255
            psnr = calculate_psnr( target_Y,prediction_Y,crop_border=0)
            ssim = calculate_ssim(target_Y,prediction_Y,crop_border=0)
            total_psnr+=psnr
            total_ssim+=ssim
            seq_psnr+=psnr
            seq_ssim+=ssim
            cnt+=1
            print('psnr  '+str(psnr)+'  ssim  '+str(ssim))
            
        print(" for scene %s avg psnr Y %f"%(tag,seq_psnr/len(img_pred_dir_ls)))
        print(" for scene %s avg ssim Y %f"%(tag,seq_ssim/len(img_pred_dir_ls)))
    # print('model is ',l)
    print(" total avg psnr Y %f"%(total_psnr/cnt))
    print(" total avg ssim Y %f"%(total_ssim/cnt))
           

""" Entry Point """
def main():
  
    test()


if __name__ == "__main__":
    # main(args)
    main()
    cal_cri()
    # GT_img = cv2.imread('/home/zhangyuantong/dataset/REDS/mysplit/test//020/00000001.png')
   
    # pred_img = cv2.imread('/home/zhangyuantong/code/ST-SR/flow_base_STSR/lr_b_insert_f_dyn/020/00000001.png')
    # psnr = utils.calculate_psnr(pred_img, GT_img)
    # print('psnr :',psnr)Building model: iconstsr
#params 19803334
