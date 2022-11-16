


# @Time : 2021/1/22 17:53 

# @Author : xx

# @File : .py 

# @Software: PyCharm

# @description=''
import torch.nn as nn

from modules.softsplatting import softsplat
from modules.softsplatting.run import backwarp
import torch
from modules.other_modules import context_extractor_layer , Matric_UNet,light_context_extractor_layer
from modules.gridnet.net_module import GridNet,Light_GridNet
from modules.pwc.utils.flow_utils import show_compare
import cv2
from torch.nn.modules.utils import _pair
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import ConvModule, constant_init, kaiming_init

class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.
    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


# insert_2 这种方式 即 让 hidden_state 单独拉出来卷积获得三层再跟原来的img提取的feature cat 是不合理的
#谨慎使用吧
class Insert_net_2(nn.Module):
    def __init__(self,shape,is_train,time_step):
        super(Insert_net_2, self).__init__()
        #直接把原始图片的分辨率传进来，方便确认图片大小
        self.shape = shape
        # 从第一张到第二张的extracor
        self.feature_extractor = context_extractor_layer(3)
      
        
        #注意，这个参数是可以学出来的！！！
        self.alpha = nn.Parameter(-torch.ones(1))
      
        self.Matric_UNet = Matric_UNet()
        self.grid_net = Light_GridNet()
        self.mix_f_b = nn.Conv2d(128,64,3,1,1)
        self.is_train = is_train
        if self.is_train:
            self.insert_rate = 0.5
        else:
            self.insert_time_ls = time_step
    #把估计的flow 尺寸也减小
    def scale_flow(self,flow):
        #注意 pwc-net  那边传回来的flow的分辨率是原始大小的1/4
        intHeight,intWidth = self.shape
        #到底应该乘几是个谜我觉得确实应该乘的是 2,4
        # https://github.com/sniklaus/softmax-splatting/issues/12
        flow_scale_half =(20.0/2.0) * nn.functional.interpolate(input=flow,
                                                                     size=(int(intHeight/2), int(intWidth /2)),
                                                                     mode='bilinear', align_corners=False)
        flow_scale_raw =(20.0)* nn.functional.interpolate(input=flow,size=(int(intHeight), int(intWidth)),
                                                                     mode='bilinear', align_corners=False)
        return [flow_scale_raw,flow_scale_half,flow*(20/4.0)]

    def scale_tenMetric(self, tenMetric):
        intHeight, intWidth = self.shape
        tenMetric_scale_half =  nn.functional.interpolate(input=tenMetric,size=(int(intHeight / 2), int(intWidth / 2)), mode='bilinear', align_corners=False)
        tenMetric_scale_quarter = nn.functional.interpolate(input=tenMetric, size=(int(intHeight/4), int(intWidth/4)),
                                                            mode='bilinear', align_corners=False)
        return [tenMetric,tenMetric_scale_half,tenMetric_scale_quarter ]
    def insert_time_t(self,insert_t,img1,img2,flow_1to2_pyri,flow_2to1_pyri,feature_pyrr1,feature_pyrr2,tenMetric_ls_1to2,tenMetric_ls_2to1,hidden1,hidden2):
        self.insert_rate = insert_t
    # 我们插中间的 所以是0.5 嗷 这是对图片的warp
        warped_img1 = softsplat.FunctionSoftsplat(tenInput=img1, tenFlow=flow_1to2_pyri[0] *(1- self.insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')  # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter
        # print('beta 1',self.alpha)
        # print('beta 2', self.beta2)
        # warped_img1_out = warped_img1.squeeze().cpu().detach().numpy().transpose(1,2,0)
        # cv2.imshow(warped_img1_out)
        # cv2.waitKey(0)
        warped_hidden1 =  softsplat.FunctionSoftsplat(tenInput=hidden1, tenFlow=flow_1to2_pyri[0] *(1- self.insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')

        warped_pyri1_1 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[0], tenFlow=flow_1to2_pyri[0] * (1- self.insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')
        warped_pyri1_2 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[1], tenFlow=flow_1to2_pyri[1] * (1- self.insert_rate),
                                                    tenMetric=self.alpha * tenMetric_ls_1to2[1],
                                                    strType='softmax')
        warped_pyri1_3 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr1[2], tenFlow=flow_1to2_pyri[2] * (1- self.insert_rate),
                                                    tenMetric=self.alpha * tenMetric_ls_1to2[2],
                                                    strType='softmax')
        
      
        # 我们插中间的 所以是0.5 嗷 这是对图片的warp
    
        warped_img2 = softsplat.FunctionSoftsplat(tenInput=img2, tenFlow=flow_2to1_pyri[0] *self.insert_rate,
                                                tenMetric=self.alpha * tenMetric_ls_2to1[0],
                                                strType='softmax')  # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter
        warped_hidden2 =  softsplat.FunctionSoftsplat(tenInput=hidden2, tenFlow=flow_2to1_pyri[0] *self.insert_rate,
                                                tenMetric=self.alpha* tenMetric_ls_2to1[0],
                                                strType='softmax') 
        out_feat_mix = self.mix_f_b(torch.cat((warped_hidden1,warped_hidden2),dim = 1))
        warped_pyri2_1= softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[0], tenFlow=flow_2to1_pyri[0] *self.insert_rate,
                                                    tenMetric=self.alpha * tenMetric_ls_2to1[0],
                                                    strType='softmax')
        warped_pyri2_2 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[1], tenFlow=flow_2to1_pyri[1] *self.insert_rate,
                                                    tenMetric=self.alpha * tenMetric_ls_2to1[1],
                                                    strType='softmax')
        warped_pyri2_3 = softsplat.FunctionSoftsplat(tenInput=feature_pyrr2[2], tenFlow=flow_2to1_pyri[2] *self.insert_rate,
                                                    tenMetric=self.alpha * tenMetric_ls_2to1[2],
                                                    strType='softmax')
    
        grid_input_l1 = torch.cat([warped_img1, warped_pyri1_1,warped_hidden1,warped_img2,warped_pyri2_1,warped_hidden2], dim=1)

        grid_input_l2 = torch.cat([ warped_pyri1_2,warped_pyri2_2], dim=1)

        grid_input_l3 = torch.cat([ warped_pyri1_3,warped_pyri2_3], dim=1)

        out_im = self.grid_net(grid_input_l1,grid_input_l2,grid_input_l3)
        return out_feat_mix,out_im
    def forward(self,img1,img2,hidden1,hidden2,pwc_flow):
        feature_pyrr1 = self.feature_extractor(img1)
        feature_pyrr2 = self.feature_extractor(img2)

        flow_1to2 = pwc_flow['0to1']
       

        flow_1to2_pyri = self.scale_flow(flow_1to2)
        #可以实时的查看光流  因为我们没有gt 就把预测的都传进去了
        # show_compare(flow_1to2_pyri[0].squeeze().cpu().detach().numpy().transpose(1,2,0), flow_1to2_pyri[0].squeeze().cpu().detach().numpy().transpose(1,2,0))
        flow_2to1 = pwc_flow['1to0']
        flow_2to1_pyri = self.scale_flow(flow_2to1)

        #这个是important metric的输入之一 注意结果一定是单通道的 对应原论文的公式15
        # 1 to 2
        tenMetric_1to2 = nn.functional.l1_loss(input=img1, target=backwarp(tenInput=img2, tenFlow=flow_1to2_pyri[0]),
                                                reduction='none').mean(1, True)
        tenMetric_1to2 = self.Matric_UNet(tenMetric_1to2,img1)
        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
        # 2 to 1
        tenMetric_2to1 = nn.functional.l1_loss(input=img2, target=backwarp(tenInput=img1, tenFlow=flow_2to1_pyri[0]),
                                            reduction='none').mean(1, True)
        tenMetric_2to1 = self.Matric_UNet(tenMetric_2to1, img2)
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)
        if self.is_train:
            out_feat_mix,out_im = self.insert_time_t(self.insert_rate,img1,img2,flow_1to2_pyri,flow_2to1_pyri,feature_pyrr1,feature_pyrr2,tenMetric_ls_1to2,tenMetric_ls_2to1,hidden1,hidden2)
            return out_feat_mix,out_im
        else:
            intermediate_list = []
            for insert_t in self.insert_time_ls:
                out_feat_mix,out_im = self.insert_time_t(insert_t,img1,img2,flow_1to2_pyri,flow_2to1_pyri,feature_pyrr1,feature_pyrr2,tenMetric_ls_1to2,tenMetric_ls_2to1,hidden1,hidden2)
                intermediate_list.append({'feat':out_feat_mix,'frame':out_im})
            return intermediate_list



       
if __name__=='__main__':
    W = 448
    H = 256
    N = 1
    tenFirst = torch.rand(size=(N, 3, H, W)).cuda()
    tenSecond = torch.rand(size=(N, 3, H, W)).cuda()
    hidden1 = torch.rand(size=(N, 64, H, W)).cuda()
    hidden2 = torch.rand(size=(N, 64, H, W)).cuda()
    pwc_flow = {'0to1':torch.rand(size=(N, 2, H//4, W//4)).cuda(),'1to0':torch.rand(size=(N, 2, H//4, W//4)).cuda()} 
    model = Insert_net_2(tenFirst.shape).cuda()

    out_f,out_im = model(tenFirst,tenSecond,hidden1,hidden2,pwc_flow)
    print(out_f.shape)
    print(out_im.shape)










