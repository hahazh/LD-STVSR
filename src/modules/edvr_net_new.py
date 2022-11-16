import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair
from modules.softsplatting.run import backwarp
from modules.general_module import (PixelShufflePack, ResidualBlockNoBN,
                                  make_layer)
from modules.softsplatting import softsplat
import torch.nn.functional as F
import numpy as np
# from mmedit.models.registry import BACKBONES
# from mmedit.utils import get_root_logger


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


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVRNet.
    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        deform_groups (int): Deformable groups. Defaults: 8.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 deform_groups=8,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
            if i == 3:
                self.offset_conv2[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            else:
                self.offset_conv2[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg)
                self.offset_conv3[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            self.dcn_pack[level] = ModulatedDCNPack(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=deform_groups)

            if i < 3:
                act_cfg_ = act_cfg if i == 2 else None
                self.feat_conv[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg_)

        # Cascading DCN
        self.cas_offset_conv1 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_offset_conv2 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(
            mid_channels,
            mid_channels,
            3,
            padding=1,
            deform_groups=deform_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neighbor_feats, ref_feats):
        """Forward function for PCDAlignment.
        Align neighboring frames to the reference frame in the feature level.
        Args:
            neighbor_feats (list[Tensor]): List of neighboring features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).
            ref_feats (list[Tensor]): List of reference features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # The number of pyramid levels is 3.
        assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
            'The length of neighbor_feats and ref_feats must be both 3, '
            f'but got {len(neighbor_feats)} and {len(ref_feats)}')

        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([neighbor_feats[i - 1], ref_feats[i - 1]],
                               dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](
                    torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)

            feat = self.dcn_pack[level](neighbor_feats[i - 1], offset)
            if i == 3:
                feat = self.lrelu(feat)
            else:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))

            if i > 1:
                # upsample offset and features
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feats[0]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat
#这里不改一下吗？？？
class Metric(torch.nn.Module):
    def __init__(self):
        super(Metric, self).__init__()

        self.paramScale = torch.nn.Parameter(-torch.ones(1, 1, 1, 1))

    def forward(self, tenFirst, tenSecond, tenFlow):
        
        return self.paramScale * torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenSecond, tenFlow),reduction='none').mean(1,True)
# class Small_UNet(nn.Module):
#     def __init__(self):
#         super(Small_UNet, self).__init__()

#         class Decoder(nn.Module):
#             def __init__(self, l_num):
#                 super(Decoder, self).__init__()

#                 self.conv_relu = nn.Sequential(
#                     nn.ReLU(inplace=False),
#                     nn.Conv2d(in_channels=l_num * 32 * 2, out_channels=l_num * 32, kernel_size=3, stride=1, padding=1),
#                     nn.ReLU(inplace=False),
#                     nn.Conv2d(in_channels=l_num * 32, out_channels=l_num * 32, kernel_size=3, stride=1, padding=1),

#                 )

#             def forward(self, x1, x2):
#                 x1 = torch.cat((x1, x2), dim=1)
#                 x1 = self.conv_relu(x1)
#                 return x1
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

#         self.down_l1 = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#         )
#         self.down_l2 = nn.Sequential(
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#         )

#         self.middle = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False)
#         )
#         self.up_l2 = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False)
#         )
#         self.up_l1 = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#         )

#         self.Decoder2 = Decoder(2)
#         self.Decoder1 = Decoder(2)

#     def forward(self, ten):
#         conv_1 = self.conv1(ten)
#         ten_d_l1 = self.down_l1(conv_1)
#         ten_d_l2 = self.down_l2(ten_d_l1)

#         ten_mid = self.middle(ten_d_l2)

#         ten_u_l2 = self.Decoder2(ten_d_l2, ten_mid)
#         ten_u_l1 = self.up_l2(ten_u_l2)

#         ten_u_l1 = self.Decoder1(ten_d_l1, ten_u_l1)
#         ten_out = self.up_l1(ten_u_l1)

#         return ten_out
# from  https://github.com/JunHeum/ABME/blob/d9f04d160d6806204a384b29dc6a4821152bb77b/model/SynthesisNet.py


class Small_UNet(nn.Module):
    def __init__(self):
        super(Small_UNet, self).__init__()

        class Decoder(nn.Module):
            def __init__(self, l_num):
                super(Decoder, self).__init__()

                self.conv_relu = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=l_num * 32 * 2, out_channels=l_num * 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=l_num * 32, out_channels=l_num * 32, kernel_size=3, stride=1, padding=1),

                )

            def forward(self, x1, x2):
                x1 = torch.cat((x1, x2), dim=1)
                x1 = self.conv_relu(x1)
                return x1
        self.conv_img = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.down_l1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.down_l2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        self.up_l2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        self.up_l1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        )

        self.Decoder2 = Decoder(2)
        self.Decoder1 = Decoder(1)

    def forward(self, img1):
        conv_img = self.conv_img(img1)
        ten_d_l1 = self.down_l1(conv_img)
        ten_d_l2 = self.down_l2(ten_d_l1)

        ten_mid = self.middle(ten_d_l2)

        ten_u_l2 = self.Decoder2(ten_d_l2, ten_mid)
        ten_u_l1 = self.up_l2(ten_u_l2)

        ten_u_l1 = self.Decoder1(ten_d_l1, ten_u_l1)
        ten_out = self.up_l1(ten_u_l1)

        return ten_out

class DynFilter(nn.Module):
    def __init__(self, kernel_size=(3,3), padding=1, DDP=False):
        super(DynFilter, self).__init__()

        self.padding = padding
        
        filter_localexpand_np = np.reshape(np.eye(np.prod(kernel_size), np.prod(kernel_size)), (np.prod(kernel_size), 1, kernel_size[0], kernel_size[1]))
        if DDP:
            self.register_buffer('filter_localexpand', torch.FloatTensor(filter_localexpand_np)) # for DDP model
        else:
            self.filter_localexpand = torch.FloatTensor(filter_localexpand_np).cuda() # for single model

    def forward(self, x, filter):
        x_localexpand = []

        for c in range(x.size(1)):
            x_localexpand.append(F.conv2d(x[:, c:c + 1, :, :], self.filter_localexpand, padding=self.padding))

        x_localexpand = torch.cat(x_localexpand, dim=1)
        x = torch.sum(torch.mul(x_localexpand, filter), dim=1).unsqueeze(1)

        return x
class Forward_warp_guided_no_pcd(nn.Module):
    def __init__(self, mid_channels=64, groups=8, act_cfg=dict(type='LeakyReLU', negative_slope=0.1),use_feat = True):
        super(Forward_warp_guided_no_pcd, self).__init__()
        # fea1
        # L3: level 3, 1/4 spatial size
        self.alpha = -20.0
        # self.L2_fea_conv_1 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=act_cfg)
        # L1: level 1, original spatial size
      
        # self.L1_fea_conv_1 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=None)
        # fea2
         # L3: level 3, 1/4 spatial size
        # self.L2_fea_conv_2 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=act_cfg)
        # L1: level 1, original spatial size
        # self.L1_fea_conv_2 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=None)
        # use for downsampling feat and img
        self.fea_L1_conv  = nn.Conv2d(3, mid_channels, 3, 1, 1, bias=True)
        self.fea_L2_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.fea_L3_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        
        self.hs_L2_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.hs_L3_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        # self.img_L2_conv1 = nn.Conv2d(3, mid_channels, 3, 2, 1, bias=True)
       
        # self.img_L3_conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.metric = Metric()
        # self.blend = nn.Conv2d(mid_channels*2+32, mid_channels, 1, 1, 0, bias=True)
        # self.small_u = Small_UNet()
        self.reduce1 = nn.Conv2d(2*mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.reduce2 = nn.Conv2d(3*mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.reduce3 = nn.Conv2d(2*mid_channels+3, mid_channels, 3, 1, 1, bias=True)
        
        self.mask_l3 = nn.Conv2d(2*mid_channels, 1, 3, 1, 1, bias=True)
        self.mask_l2 = nn.Conv2d(2*mid_channels, 1, 3, 1, 1, bias=True)
        self.mask_l1 = nn.Conv2d(2*mid_channels, 1, 3, 1, 1, bias=True)
        self.short_cut = nn.Conv2d(2*mid_channels+3,mid_channels , 3, 1, 1, bias=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        
    def scale_tenMetric(self, tenMetric):
        intHeight, intWidth = tenMetric.shape[2:]
        tenMetric_scale_half =  nn.functional.interpolate(input=tenMetric,size=(int(intHeight / 2), int(intWidth / 2)), mode='bilinear', align_corners=False)
        tenMetric_scale_quarter = nn.functional.interpolate(input=tenMetric, size=(int(intHeight/4), int(intWidth/4)),
                                                            mode='bilinear', align_corners=False)
        return [tenMetric,tenMetric_scale_half,tenMetric_scale_quarter ]
    def get_pyrr_feats(self,feat1,feat2):
        L2_feat_1  = self.hs_L2_conv1(feat1)
        L2_feat_2 =  self.hs_L2_conv1(feat2)
        L3_feat_1 = self.hs_L3_conv2(L2_feat_1)
        L3_feat_2 = self.hs_L3_conv2(L2_feat_2)
        return [[feat1,L2_feat_1,L3_feat_1],[feat2,L2_feat_2,L3_feat_2]]

    def forward(self,img1,img2, feat1,feat2,flow_1to2_pyri,flow_2to1_pyri,insert_time):
        
        insert_rate = insert_time
        
        tenMetric_1to2 = self.metric(img1,img2,flow_1to2_pyri[0])
        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
        tenMetric_2to1 = self.metric(img2,img1,flow_2to1_pyri[0])
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)
        #first calculate anchor img
        warped_content_feat1_ =  softsplat.FunctionSoftsplat(tenInput=img1, tenFlow=flow_1to2_pyri[0] *(insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')
        warped_content_feat2_ =  softsplat.FunctionSoftsplat(tenInput=img2, tenFlow=flow_2to1_pyri[0] *(1-insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_2to1[0],
                                                strType='softmax')
        warped_content_feat1_ = torch.where(warped_content_feat1_>0,warped_content_feat1_,warped_content_feat2_)
        warped_content_feat2_ = torch.where(warped_content_feat2_>0,warped_content_feat2_,warped_content_feat1_)
        anchor_img_l1 = (warped_content_feat1_+warped_content_feat2_)/2.0
        
        # warped_feat2_l2 = torch.where(warped_feat2_l2>0,warped_feat2_l2,warped_feat1_l2)

        # extract feat
        L1_feat1 = self.lrelu(self.fea_L1_conv(img1))
        L1_feat2 = self.lrelu(self.fea_L1_conv(img2))
        L2_feat1 =  self.fea_L2_conv1(L1_feat1)
        L2_feat2 =  self.fea_L2_conv1(L1_feat2)
        L3_feat1 = self.fea_L3_conv2(L2_feat1)
        L3_feat2 = self.fea_L3_conv2(L2_feat2)
        
        #L2
        warped_feat1_l2 =  softsplat.FunctionSoftsplat(tenInput=L2_feat1, tenFlow=flow_1to2_pyri[1] *(insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[1],
                                                strType='softmax')
        warped_feat2_l2 =  softsplat.FunctionSoftsplat(tenInput=L2_feat2, tenFlow=flow_2to1_pyri[1] *(1-insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_2to1[1],
                                                strType='softmax')
        warped_feat1_l2 = torch.where(warped_feat1_l2>0,warped_feat1_l2,warped_feat2_l2)
        warped_feat2_l2 = torch.where(warped_feat2_l2>0,warped_feat2_l2,warped_feat1_l2)
        anchor_feat_l2 = (warped_feat1_l2+warped_feat2_l2)/2.0
        
        #L3
        warped_feat1_l3 =  softsplat.FunctionSoftsplat(tenInput=L3_feat1, tenFlow=flow_1to2_pyri[2] *(insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[2],
                                                strType='softmax')
        warped_feat2_l3 =  softsplat.FunctionSoftsplat(tenInput=L3_feat2, tenFlow=flow_2to1_pyri[2] *(1-insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_2to1[2],
                                                strType='softmax')
        warped_feat1_l3 = torch.where(warped_feat1_l3>0,warped_feat1_l3,warped_feat2_l3)
        warped_feat2_l3 = torch.where(warped_feat2_l3>0,warped_feat2_l3,warped_feat1_l3)
        
        anchor_feat_l3 = (warped_feat1_l3+warped_feat2_l3)/2.0
        
        hs_feat1_pr,hs_feat2_pr = self.get_pyrr_feats(feat1,feat2)
        # from low resolution to high resolution
        hs_feate1_l3 = softsplat.FunctionSoftsplat(tenInput=hs_feat1_pr[2], tenFlow=flow_1to2_pyri[2] *(insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[2],
                                                strType='softmax')
        hs_feate2_l3 = softsplat.FunctionSoftsplat(tenInput=hs_feat2_pr[2], tenFlow=flow_2to1_pyri[2] *(1-insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_2to1[2],
                                                strType='softmax')
        mask_l3 = torch.sigmoid(self.mask_l3(torch.cat([hs_feate1_l3,hs_feate2_l3],dim=1))) 
        hs_feate_l3 = mask_l3*(hs_feate1_l3)+(1-mask_l3)*hs_feate2_l3
        
        hs_feat_l3 = torch.cat([anchor_feat_l3,hs_feate_l3],dim=1)
        hs_feat_l3 = self.reduce1(hs_feat_l3)
        hs_feat_l3 =  self.upsample(hs_feat_l3)
        
        
        hs_feate1_l2 = softsplat.FunctionSoftsplat(tenInput=hs_feat1_pr[1], tenFlow=flow_1to2_pyri[1] *(insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[1],
                                                strType='softmax')
        hs_feate2_l2 = softsplat.FunctionSoftsplat(tenInput=hs_feat2_pr[1], tenFlow=flow_2to1_pyri[1] *(1-insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_2to1[1],
                                                strType='softmax')
        
        mask_l2 = torch.sigmoid(self.mask_l2(torch.cat([hs_feate1_l2,hs_feate2_l2],dim=1))) 
        hs_feat_l2 = mask_l2*(hs_feate1_l2)+(1-mask_l2)*hs_feate2_l2
        
        hs_feat_l2 = torch.cat([anchor_feat_l2,hs_feat_l3,hs_feat_l2],dim=1)
        hs_feat_l2 = self.reduce2(hs_feat_l2)
        hs_feat_l2 =  self.upsample(hs_feat_l2)
        
        
        hs_feate1_l1 = softsplat.FunctionSoftsplat(tenInput=hs_feat1_pr[0], tenFlow=flow_1to2_pyri[0] *(insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')
        hs_feate2_l1 = softsplat.FunctionSoftsplat(tenInput=hs_feat2_pr[0], tenFlow=flow_2to1_pyri[0] *(1-insert_rate),
                                                tenMetric=self.alpha* tenMetric_ls_2to1[0],
                                                strType='softmax')
        mask_l1 = torch.sigmoid(self.mask_l1(torch.cat([hs_feate1_l1,hs_feate2_l1],dim=1))) 
        hs_feat_l1 = mask_l1*(hs_feate1_l1)+(1-mask_l1)*hs_feate2_l1
        hs_feat_l1 = torch.cat([anchor_img_l1,hs_feat_l2,hs_feat_l1],dim=1)
        
        hs_feat_l1 = self.reduce3(hs_feat_l1)
        
        # refined_feat = self.small_u(hs_feat_l1)
        
        anchor_2 = self.upsample(anchor_feat_l2)
        anchor_3 = self.upsample(self.upsample(anchor_feat_l3)) 
        anchr_feat = self.lrelu(self.short_cut(torch.cat([anchor_img_l1,anchor_2,anchor_3],dim=1)))
        return anchr_feat+hs_feat_l1

class Forward_warp_guided_pcd(nn.Module):
    def __init__(self, mid_channels=64, groups=8, act_cfg=dict(type='LeakyReLU', negative_slope=0.1),use_feat = True):
        super(Forward_warp_guided_pcd, self).__init__()
        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1  = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L3_offset_conv2_1 = ConvModule (mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L3_dcnpack_1 = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1,
                                    deform_groups=groups)
          # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1  = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)# concat for diff
        self.L2_offset_conv2_1 = ConvModule (2*mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)# concat for offset
        self.L2_offset_conv3_1 = ConvModule (mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)

        self.L2_dcnpack_1 = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1,
                                    deform_groups=groups)
        self.L2_fea_conv_1 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=act_cfg)
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L1_offset_conv2_1 = ConvModule (2*mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L1_offset_conv3_1 = ConvModule (mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L1_dcnpack_1 = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1,
                                    deform_groups=groups)
        self.L1_fea_conv_1 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=None)
        # fea2
         # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2  = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L3_offset_conv2_2 = ConvModule (mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L3_dcnpack_2 = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1,
                                    deform_groups=groups)
          # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2  = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)# concat for diff
        self.L2_offset_conv2_2 = ConvModule (2*mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)# concat for offset
        self.L2_offset_conv3_2 = ConvModule (mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)

        self.L2_dcnpack_2 = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1,
                                    deform_groups=groups)
        self.L2_fea_conv_2 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=act_cfg)
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L1_offset_conv2_2 = ConvModule (2*mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L1_offset_conv3_2 = ConvModule (mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.L1_dcnpack_2 = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1,
                                    deform_groups=groups)
        self.L1_fea_conv_2 = ConvModule( mid_channels * 2,mid_channels, 3,padding=1,act_cfg=None)
        # use for downsampling feat and img
        self.fea_L1_conv  = nn.Conv2d(mid_channels+3, mid_channels, 3, 1, 1, bias=True)
        self.fea_L2_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.fea_L3_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
        )
        # self.img_L2_conv1 = nn.Conv2d(3, mid_channels, 3, 2, 1, bias=True)
       
        # self.img_L3_conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.metric = Metric()
        self.blend = nn.Conv2d(mid_channels*2+32, mid_channels, 1, 1, 0, bias=True)
        self.small_u = Small_UNet()
        self.mask_l1 = nn.Conv2d(2*mid_channels, 1, 3, 1, 1, bias=True)
    def scale_tenMetric(self, tenMetric):
        intHeight, intWidth = tenMetric.shape[2:]
        tenMetric_scale_half =  nn.functional.interpolate(input=tenMetric,size=(int(intHeight / 2), int(intWidth / 2)), mode='bilinear', align_corners=False)
        tenMetric_scale_quarter = nn.functional.interpolate(input=tenMetric, size=(int(intHeight/4), int(intWidth/4)),
                                                            mode='bilinear', align_corners=False)
        return [tenMetric,tenMetric_scale_half,tenMetric_scale_quarter ]
    def get_pyrr_feats(self,feat1,feat2,img1,img2):
        L2_feat_1  = self.lrelu(self.fea_L2_conv1(feat1))

    def forward(self, feat1,feat2,img1,img2,flow_1to2_pyri,flow_2to1_pyri,insert_time):
        # first forward warping
        insert_rate = insert_time
        alpha = -20.0
        y = []
        tenMetric_1to2 = self.metric(feat1,feat2,flow_1to2_pyri[0])
        tenMetric_ls_1to2 = self.scale_tenMetric(tenMetric_1to2)
        tenMetric_2to1 = self.metric(feat2,feat1,flow_2to1_pyri[0])
        tenMetric_ls_2to1 = self.scale_tenMetric(tenMetric_2to1)
        #content map
        warped_content_feat1_ =  softsplat.FunctionSoftsplat(tenInput=feat1, tenFlow=flow_1to2_pyri[0] *(insert_rate),
                                                tenMetric=alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')
        warped_content_feat2_ =  softsplat.FunctionSoftsplat(tenInput=feat2, tenFlow=flow_2to1_pyri[0] *(1-insert_rate),
                                                tenMetric=alpha* tenMetric_ls_2to1[0],
                                                strType='softmax')
        warped_content_feat1_ = torch.where(warped_content_feat1_>0,warped_content_feat1_,warped_content_feat2_)
        warped_content_feat2_ = torch.where(warped_content_feat2_>0,warped_content_feat2_,warped_content_feat1_)
        avg_content_map = (warped_content_feat1_+warped_content_feat2_)/2.0
        avg_content_map = self.small_u(avg_content_map)
        # warped_feat2_l2 = torch.where(warped_feat2_l2>0,warped_feat2_l2,warped_feat1_l2)

        # extract feat
        L1_feat1 = self.lrelu(self.fea_L1_conv(torch.cat([feat1,img1],dim=1)))
        L1_feat2 = self.lrelu(self.fea_L1_conv(torch.cat([feat2,img2],dim=1)))
        L2_feat1 =  self.fea_L2_conv1(L1_feat1)
        L2_feat2 =  self.fea_L2_conv1(L1_feat2)
        L3_feat1 = self.fea_L3_conv2(L2_feat1)
        L3_feat2 = self.fea_L3_conv2(L2_feat2)
        #L3
        warped_feat1_l3 =  softsplat.FunctionSoftsplat(tenInput=L3_feat1, tenFlow=flow_1to2_pyri[2] *(insert_rate),
                                                tenMetric=alpha* tenMetric_ls_1to2[2],
                                                strType='softmax')
        warped_feat2_l3 =  softsplat.FunctionSoftsplat(tenInput=L3_feat2, tenFlow=flow_2to1_pyri[2] *(1-insert_rate),
                                                tenMetric=alpha* tenMetric_ls_2to1[2],
                                                strType='softmax')
        warped_feat1_l3 = torch.where(warped_feat1_l3>0,warped_feat1_l3,warped_feat2_l3)
        warped_feat2_l3 = torch.where(warped_feat2_l3>0,warped_feat2_l3,warped_feat1_l3)
        
        L3_offset = torch.cat([warped_feat1_l3, warped_feat2_l3], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(warped_feat1_l3, L3_offset))
        #L2
        warped_feat1_l2 =  softsplat.FunctionSoftsplat(tenInput=L2_feat1, tenFlow=flow_1to2_pyri[1] *(insert_rate),
                                                tenMetric=alpha* tenMetric_ls_1to2[1],
                                                strType='softmax')
        warped_feat2_l2 =  softsplat.FunctionSoftsplat(tenInput=L2_feat2, tenFlow=flow_2to1_pyri[1] *(1-insert_rate),
                                                tenMetric=alpha* tenMetric_ls_2to1[1],
                                                strType='softmax')
        # warped_feat1_l2 = torch.where(warped_feat1_l2>0,warped_feat1_l2,warped_feat2_l2)
        # warped_feat2_l2 = torch.where(warped_feat2_l2>0,warped_feat2_l2,warped_feat1_l2)

        L2_offset = torch.cat([warped_feat1_l2,warped_feat2_l2], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(
            torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(warped_feat1_l2, L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(
            torch.cat([L2_fea, L3_fea], dim=1)))
        #L1 
        warped_feat1_l1 =  softsplat.FunctionSoftsplat(tenInput=L1_feat1, tenFlow=flow_1to2_pyri[0] *(insert_rate),
                                                tenMetric=alpha* tenMetric_ls_1to2[0],
                                                strType='softmax')
        warped_feat2_l1 =  softsplat.FunctionSoftsplat(tenInput=L1_feat2, tenFlow=flow_2to1_pyri[0] *(1-insert_rate),
                                                tenMetric=alpha* tenMetric_ls_2to1[0],
                                                strType='softmax')
        # warped_feat1_l1 = torch.where(warped_feat1_l1>0,warped_feat1_l1,warped_feat2_l1)
        # warped_feat2_l1 = torch.where(warped_feat2_l1>0,warped_feat2_l1,warped_feat1_l1)

        L1_offset = torch.cat([warped_feat1_l1, warped_feat2_l1], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(
            torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(warped_feat1_l1, L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
########------------------- do not share weights---------------------------------###############################
        # L3
        L3_offset = torch.cat([warped_feat2_l3, warped_feat1_l3], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(warped_feat2_l3, L3_offset))
        # L2
        L2_offset = torch.cat([warped_feat2_l2, warped_feat1_l2], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(
            torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(warped_feat2_l2, L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(
            torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([warped_feat2_l1,warped_feat1_l1], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(
            torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(warped_feat2_l1, L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2,
                               mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        mask = torch.sigmoid(self.mask_l1(torch.cat(y,dim=1))) 
        mask_res = mask*(y[0])+(1-mask)*y[1]
        y = torch.cat(y, dim=1)

       
        
        y = self.blend(torch.cat([y,avg_content_map],dim=1))+mask_res
 
        return y

       
class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.
    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        num_frames (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 num_frames=5,
                 center_frame_idx=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(
            mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """Forward function for TSAFusion.
        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (n, h, w)
            corr_l.append(corr.unsqueeze(1))  # (n, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (n, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob

        # fusion
        feat = self.feat_fusion(aligned_feat)

        # spatial attention
        attn = self.spatial_attn1(aligned_feat)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.
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
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

class FirstOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.
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
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        # self.conv_to_2 =  nn.Conv2d( 18 * self.deform_groups, 2, 3, 1, 1)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        extra_feat = torch.cat([extra_feat, flow_1], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(offset)
       
        offset = offset + flow_1.flip(1).repeat(1,offset.size(1) // 2, 1,1)
        # ret_flow = self.conv_to_2(offset.clone().flip(1))
        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

class EDVRNet(nn.Module):
    """EDVR network structure for video super-resolution.
    Now only support X4 upsampling factor.
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN,
            num_blocks_extraction,
            mid_channels=mid_channels)

        # generate pyramid features
        self.feat_l2_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        # pcd alignment
        self.pcd_alignment = PCDAlignment(
            mid_channels=mid_channels, deform_groups=deform_groups)
        # fusion
        if self.with_tsa:
            self.fusion = TSAFusion(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1,
                                    1)

        # reconstruction
        self.reconstruction = make_layer(
            ResidualBlockNoBN,
            num_blocks_reconstruction,
            mid_channels=mid_channels)
        # upsample
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        # we fix the output channels in the last few layers to 64.
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """Forward function for EDVRNet.
        Args:
            x (Tensor): Input tensor with shape (n, t, c, h, w).
        Returns:
            Tensor: SR center frame with shape (n, c, h, w).
        """
        n, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, (
            'The height and width of inputs should be a multiple of 4, '
            f'but got {h} and {w}.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        # L2
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        # L3
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        # pcd alignment
        ref_feats = [  # reference feature list
            l1_feat[:, self.center_frame_idx, :, :, :].clone(),
            l2_feat[:, self.center_frame_idx, :, :, :].clone(),
            l3_feat[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [
                l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(),
                l3_feat[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)

        # reconstruction
        out = self.reconstruction(feat)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(x_center)
        out += base
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            if self.with_tsa:
                for module in [
                        self.fusion.feat_fusion, self.fusion.spatial_attn1,
                        self.fusion.spatial_attn2, self.fusion.spatial_attn3,
                        self.fusion.spatial_attn4, self.fusion.spatial_attn_l1,
                        self.fusion.spatial_attn_l2,
                        self.fusion.spatial_attn_l3,
                        self.fusion.spatial_attn_add1
                ]:
                    kaiming_init(
                        module.conv,
                        a=0.1,
                        mode='fan_out',
                        nonlinearity='leaky_relu',
                        bias=0,
                        distribution='uniform')
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
