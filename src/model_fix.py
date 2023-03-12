
from numpy.core.numeric import outer
from  modules.arbitrary_shape_module.arbitrary_upsample import Arb_upsample_cas
import torch

import torch.nn as nn
import torch.nn.functional as F
from modules.general_module import ResidualBlocksWithInputConv, block

from modules.general_module import flow_warp,ref_attention
import os
from modules.edvr_net_new import Forward_warp_guided_pcd,FirstOrderDeformableAlignment
from modules.flow_module import SPyNet




class Unstrained_ST(nn.Module):
   

    def __init__(self,
                 scale = 4.0,
                 scale2 = 4.0,
                 mid_channels=64,
                 num_blocks=30,
                 keyframe_stride=1,
                 padding=2,
                 time_step = [0.2,0.4,0.6,0.8]):

        super().__init__()
        self.scale = scale
        self.scale2 = scale2
        self.mid_channels = mid_channels
        self.padding = padding
        self.keyframe_stride = keyframe_stride
        self.time_step = time_step
        self.spy = SPyNet(pretrained=None) 
        self.insert = Forward_warp_guided_pcd()
       

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels, mid_channels, num_blocks)

        self.forward_resblocks = ResidualBlocksWithInputConv(
             mid_channels , mid_channels, num_blocks)
        self.forward_resblocks_insert = ResidualBlocksWithInputConv(
             mid_channels, mid_channels, num_blocks)
        self.arb_upample = Arb_upsample_cas(scale,scale2)
        self.lr_feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.deform_align = FirstOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=10)
        self.atten_fuse = TFA()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def spatial_padding(self, lrs):
      
        n, t, c, h, w = lrs.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.view(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(n, t, c, h + pad_h, w + pad_w)

    def check_if_mirror_extended(self, lrs):
       

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def compute_flow(self, lrs):
        

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spy(lrs_2,lrs_1)
        flows_backward_mid = nn.functional.interpolate(input=flows_backward, size=(h//2, w//2),
                                                            mode='bilinear', align_corners=False)
        flows_backward_small = nn.functional.interpolate(input=flows_backward, size=(h//4, w//4),
                                                            mode='bilinear', align_corners=False)
        flows_backward_large = flows_backward.view(n, (t - 1), 2, h, w)
        flows_backward_mid = (1.0/2.0)*flows_backward_mid.view(n, (t - 1), 2, h//2, w//2)
        flows_backward_small = (1.0/4.0)*flows_backward_small.view(n, (t - 1), 2, h//4, w//4)
        flows_backward_pri = [flows_backward_large,flows_backward_mid,flows_backward_small]

        if self.is_mirror_extended:
          
            flows_forward = None
            flows_forward_large = None
        else:
            flows_forward = self.spy(lrs_1,lrs_2)
            flows_forward_mid = nn.functional.interpolate(input=flows_forward, size=(h//2, w//2),
                                                            mode='bilinear', align_corners=False)
            flows_forward_small = nn.functional.interpolate(input=flows_forward, size=(h//4, w//4),
                                                            mode='bilinear', align_corners=False)
            flows_forward_large = flows_forward.view(n, (t - 1), 2, h, w)
            flows_forward_mid = (1.0/2.0)*flows_forward_mid.view(n, (t - 1), 2, h//2, w//2)
            flows_forward_small = (1.0/4.0)*flows_forward_small.view(n, (t - 1), 2, h//4, w//4)
            
            flows_forward_pri = [flows_forward_large,flows_forward_mid,flows_forward_small]
    
        return flows_forward_large, flows_backward_large,flows_forward_pri,flows_backward_pri


    def forward(self, lrs,scale1,scale2):
        
        n, t, c, h_input, w_input = lrs.size()

        self.check_if_mirror_extended(lrs)

     
        h, w = lrs.size(3), lrs.size(4)

        
        keyframe_idx = list(range(0, t, self.keyframe_stride))
        if keyframe_idx[-1] != t - 1:
            keyframe_idx.append(t - 1)  # the last frame must be a keyframe
        flows_forward_large, flows_backward_large,flows_forward_pri,flows_backward_pri = self.compute_flow(lrs)
    
        # backward-time propgation
        outputs = []
        mix_out = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)

        feat_current_ls = []
  
        for i in range(t - 1, -1, -1):
            # ----------------for lrs ---------------
            lr_curr = lrs[:, i, :, :, :]
            feat_current = self.lr_feat_extract(lr_curr)
            feat_current_ls.append(feat_current)
            if i < t - 1:  # no warping for the last timestep
                flow_n1 = flows_backward_large[:, i, :, :, :].contiguous()
              
                    
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current], dim=1)

                feat_prop = torch.cat([feat_prop, feat_current_ls[-2]], dim=1)
              
                feat_prop = self.deform_align(feat_prop, cond, flow_n1)
                    

            feat_prop += feat_current
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)

        for ix in range(2 * t - 1):
            if ix % 2 == 0:
                mix_out.append(outputs[ix // 2])
            else:
                mix_out.append(None)
        outputs = mix_out[::-1]
        feat_current_ls = feat_current_ls[::-1]
   
        feat_prop = torch.zeros_like(feat_prop)
        outputs_large =  [None for _ in range(2*t-1)]
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            feat_current = feat_current_ls[i]
            if i > 0:  # no warping for the first timestep
                if flows_forward_large is not None:
                    flow_n1  = flows_forward_large[:, i - 1, :, :, :].contiguous()
                else:
                    flow = flows_backward_large[:, -i, :, :, :].contiguous()
                
                cond = flow_warp(feat_prop,  flow_n1.permute(0, 2, 3, 1))

                cond = torch.cat([cond, feat_current], dim=1)
                feat_prop = torch.cat([feat_prop, feat_current_ls[i-1]], dim=1)
                feat_prop = self.deform_align(feat_prop, cond, flow_n1)
        
            stacked_feat =torch.stack([ outputs[2 * i], feat_current,feat_prop]).permute(1,2,0,3,4)
            feat_prop = self.atten_fuse(stacked_feat)
            feat_prop = self.forward_resblocks(feat_prop)

            outputs[i * 2] = feat_prop
            if scale1 is not None:
                out = self.arb_upample(feat_prop,scale1,scale2)

        
            outputs_large[2 * i ] = out
           
            # ----------------for inserted lrs ---------------
            if i > 0:
                img0 = lrs[:, i-1, :, :, :]
                img1 = lrs[:, i, :, :, :]
                c0,c1 = outputs[2*i-2],outputs[2*i]
                flow_1to2_pyri,flow_2to1_pyri = [[],[],[]],[[],[],[]]
                for j in range(3):
                    flow_1to2_pyri[j] = flows_forward_pri[j][:,i-1,:,:,:]
                    flow_2to1_pyri[j] = flows_backward_pri[j][:,i-1,:,:,:]
                intermidate_ls = []
                for each in  self.time_step:

                    intermidate_ls.append(self.insert(img0,img1,c0,c1,flow_1to2_pyri,flow_2to1_pyri,each))
                for ix,sub in enumerate(intermidate_ls):
                    insert_feat_prop = sub
                    insert_feat_prop = self.forward_resblocks_insert(insert_feat_prop)
                    if  scale1 is not None:
                        out_insert = self.arb_upample(insert_feat_prop,scale1,scale2)
                    else:
                        out_insert = self.arb_upample(insert_feat_prop,self.scale,self.scale2)
                    intermidate_ls[ix] = out_insert
               
                outputs_large[2 * i - 1] = intermidate_ls
        final_out = []
        for ix,each in enumerate( outputs_large):
            if isinstance(each,list):
                for sub in each:
                    final_out.append(sub)
                  
            else:
                final_out.append(each)
                
       
        return torch.stack(final_out, dim=1)
       
class TFA(nn.Module):
    
    def __init__(self,nf = 64):
        super(TFA, self).__init__()
        self.conv3d_1 = nn.Conv3d(nf, nf, (1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.conv3d_2 = nn.Conv3d(nf, nf, (1,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv3d_3 = nn.Conv3d(2*nf, nf, (1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.conv3d_4 = nn.Conv3d(nf, nf, (1,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv3d_fuse1 = nn.Conv3d(2*nf, nf, (1,1,1), stride=(1,1,1), padding=(0,0,0))
        
        self.repeat_hybrid = self.make_layer_for_2x3(block,3)
        self.short_cut_out =  nn.Conv2d(3*nf, nf, 3, padding=1)
        self.att =  ref_attention()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def make_layer_for_2x3(self,block, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        #x b,c,t,h,w
        b,c,t,h,w = x.size()
        x1 = self.conv3d_1(self.lrelu(x))
        x1 = self.conv3d_2(self.lrelu(x1))
        x1 = torch.cat((x,x1), 1)
        x1 = self.conv3d_3(self.lrelu(x1))
        x2 = self.conv3d_4(self.lrelu((x1)))
        x2 = torch.cat((x1,x2),1)
        x3 = self.conv3d_fuse1(self.lrelu(x2))
        x4 = self.repeat_hybrid(x3)
        out = self.lrelu(self.short_cut_out(x.contiguous().view(b,c*t,h,w))) + self.att(x4)
        return out




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    # (n, t, c, h, w) = 1, 5, 3, 64,64
    scale,scale2 = 4.0,4.0
    model = Unstrained_ST(scale = scale,scale2 = scale2)
    

   
    