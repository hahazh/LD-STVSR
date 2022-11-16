import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from turtle import pen
from matplotlib.pyplot import sca
# from  modules.arbitrary_shape_module.sepconv_opt import sepconv_func 
# from  modules.arbitrary_shape_module.shuffle_att import sa_layer 
from src.utils.data_util import imresize_np
from src.modules.general_module import PixelShufflePack
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math

class Arb_upsample_cas(nn.Module):
    def __init__(self,scale,scale2):
        super(Arb_upsample_cas, self).__init__()
      
        self.ps_num = 3
        self.x_shape = []
        self.y_shape = []
        for i in range(self.ps_num):
            self.x_shape.append(scale/(2**(i+1)))
            self.y_shape.append(scale2/(2**(i+1)))
        self.sa_adapt = SA_adapt(32)
        self.spacial_adaptive_conv = SA_conv(64,64)
        # self.reduce_0 =  torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.reduce =  torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.cascade_ps_result = Cascade_ps()
        # self.sa_up = SA_upsample_light()
        # self.st_att = ST_att(inplanes  =32,planes = 32)
        # self.soft_average = Mlp_GEGLU(in_features=64, hidden_features=64, act_layer=nn.GELU)
        #显存瓶颈不在这儿
        self.soft_average = nn.Sequential(nn.Conv2d(96,32,kernel_size = 1,groups=32),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32,32,kernel_size = 1),
                                           nn.ReLU(inplace=True),) 
  

        
    def forward(self,aligned_fea,scale, scale2):
        # print('scale is :',scale)
        scale_aware_fea = self.spacial_adaptive_conv(aligned_fea,scale, scale2)
        cascade_ps_result = self.cascade_ps_result(scale_aware_fea)
        std_shape_ls = []
        for i in range(self.ps_num):
            x_ratio = self.x_shape[i]
            y_ratio = self.y_shape[i]
            # print('相对',i,' 倍','下采样',scale/(2**(i+1)))
            cascade_ps_result_std_shape = std_grid_sample(cascade_ps_result[i],scale/(2**(i+1)),scale2/(2**(i+1)))
            # print(f'cascade_ps_result_std_shape {i}',cascade_ps_result_std_shape.shape)
            std_shape_ls.append(cascade_ps_result_std_shape)
        std_shape_fea = torch.stack(std_shape_ls,dim=1)
       
       
        b,d,c,h,w = std_shape_fea.shape
        # std_shape_fea = std_shape_fea.view(b,3,-1,h,w)
        std_shape_fea = std_shape_fea.transpose(1,2)
        std_shape_fea = std_shape_fea.reshape(b,d*c,h,w)
        # std_shape_fea = std_shape_fea.squeeze(3).squeeze(4).view(b,h*w,c)
        # std_shape_fea = self.soft_average(std_shape_fea)
        std_shape_fea = self.soft_average(std_shape_fea)
        # std_shape_fea = self.st_att(std_shape_fea)
        std_shape_fea = self.sa_adapt(std_shape_fea,scale,scale2)  
        std_shape_fea = self.reduce(std_shape_fea)
       
       
        return  std_shape_fea

def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    if round(scale2*w)%2!=0:
        tar_w = round(scale2*w)-1
    else:
        tar_w = round(scale2*w)
    if round(scale*h)%2!=0:
        tar_h = round(scale*h)-1
    else:
        tar_h = round(scale*h)    
    grid = np.meshgrid(range(tar_w), range(tar_h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    offset = torch.cat((offset_0, offset_1),1)
    offset = F.interpolate(offset,size=(tar_h,tar_w),mode='bilinear',align_corners=False)
    grid = grid + offset
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros',align_corners=True)

    return output


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=2, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 32, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(32, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(32, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)
        #这里是注意力融合的部分，看看能不能改了
        weight_expand = self.weight_expand.view(self.num_experts, -1)
        # weight_expand (4,512) routing_weights (262144,4)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        #等于说是先 upsampling 并加上偏移offset，（这时候已经跟目标大小一样大了）然后再乘以路由的权重（类似于卷积的形式）
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0
    
class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=2, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 32, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(32, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(32, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)
        #这里是注意力融合的部分，看看能不能改了
        weight_expand = self.weight_expand.view(self.num_experts, -1)
        # weight_expand (4,512) routing_weights (262144,4)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        #等于说是先 upsampling 并加上偏移offset，（这时候已经跟目标大小一样大了）然后再乘以路由的权重（类似于卷积的形式）
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0
class SA_upsample_light(nn.Module):
    def __init__(self, channels, num_experts=2, bias=False):
        super(SA_upsample_light, self).__init__()
        
        self.channels = channels

        # experts
        # weight_compress = []
        # for i in range(num_experts):
        #     weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
        #     nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        # self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        # weight_expand = []
        # for i in range(num_experts):
        #     weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
        #     nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        # self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 32, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        # self.routing = nn.Sequential(
        #     nn.Conv2d(32, num_experts, 1, 1, 0, bias=True),
        #     nn.Sigmoid()
        # )
        # offset head
        self.offset = nn.Conv2d(32, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size() 
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)


        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

      
        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        # fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        ## spatially varying filtering
        # out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        # out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return  fea0
class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=2):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale, scale2):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale
        scale2 = torch.ones(1, 1).to(x.device) / scale2
        routing_weights = self.routing(torch.cat((scale, scale2), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out
def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias.data, 0.0)
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv_d1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_s1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_s2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv_d1, self.conv_d2, self.conv_s1, self.conv_s2], 0.1)
    def forward(self, hd, hs):
        identity_d = hd
        out_d = F.relu(self.conv_d1(hd), inplace=False)
        out_d = self.conv_d2(out_d)

        identity_s = hs
        out_s = F.relu(self.conv_s1(hs), inplace=False)
        out_s = self.conv_s2(out_s)
     
        hsd = out_d + out_s
        return hsd + identity_d, hsd + identity_s
class Cascade_ps(nn.Module):
    def __init__(self):
        super(Cascade_ps, self).__init__()
        self.ps_num = 3
        for i in range(self.ps_num):
            #mid_channels, mid_channels, 2, upsample_kernel=3
            if i==0:
                setattr(self, f'ps_x{2**(i+1)}',PixelShufflePack(64,32,2,3))
            else:
                setattr(self, f'ps_x{2**(i+1)}',PixelShufflePack(32,32,2,3))
          
    def forward(self, x):
        fea_list = [[] for _ in range(self.ps_num)]
        # print(fea_list)
        last_level = x
        for i in range(self.ps_num):
            last_level = self.__getattr__('ps_x'+str(2**(i+1)))(last_level)
            fea_list[i]=last_level
            # print(last_level.shape)
        return fea_list
    
class Cascade_ps_seprate(nn.Module):
    def __init__(self,psnum = 3):
        super(Cascade_ps_seprate, self).__init__()
        self.ps_num = psnum
        for i in range(self.ps_num):
            #mid_channels, mid_channels, 2, upsample_kernel=3
          
            setattr(self, f'ps_x{2**(i+1)}',PixelShufflePack(64,32,2**(i+1),3))
          
    def forward(self, x):
        fea_list = [[] for _ in range(self.ps_num)]
        # print(fea_list)
        # last_level = x
        for i in range(self.ps_num):
            last_level = self.__getattr__('ps_x'+str(2**(i+1)))(x)
            fea_list[i]=last_level
            # print(last_level.shape)
        return fea_list
    
class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".
    Args:
        x: (B, D, H, W, C)
    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc11(x)
        x = self.act(x)
        x = self.act(self.fc12(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x
def std_grid_sample(x, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    if round(scale2*w)%2!=0:
        tar_w = round(scale2*w)-1
    else:
        tar_w = round(scale2*w)
    if round(scale*h)%2!=0:
        tar_h = round(scale*h)-1
    else:
        tar_h = round(scale*h)    
    grid = np.meshgrid(range(tar_w), range(tar_h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    # offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    # offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    # grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros',align_corners=True)

    return output
class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
           
            nn.Conv2d(channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale2)

        return x + adapted * mask
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
#from https://github.com/luuuyi/CBAM.PyTorch/blob/83d3312c8c542d71dfbb60ee3a15454ba253a2b0/model/resnet_cbam.py
class ST_att(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ST_att, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Arb_upsample(nn.Module):
    def __init__(self,scale,scale2):
        super(Arb_upsample, self).__init__()
      
        self.ps_num = 3
        self.x_shape = []
        self.y_shape = []
        for i in range(self.ps_num):
            self.x_shape.append(scale/(2**(i+1)))
            self.y_shape.append(scale2/(2**(i+1)))
        self.sa_adapt = SA_adapt(32)
        self.spacial_adaptive_conv = SA_adapt(64)
        # self.reduce_0 =  torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.reduce =  torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.cascade_ps_result = Cascade_ps()
        self.cascade_ps_result = Cascade_ps_seprate()
        self.sa_up = SA_upsample_light(32)
        # self.st_att = ST_att(inplanes  =32,planes = 32)
        # self.soft_average = Mlp_GEGLU(in_features=64, hidden_features=64, act_layer=nn.GELU)
        #显存瓶颈不在这儿
        self.soft_average = nn.Sequential(nn.Conv2d(96,32,kernel_size = 1,groups=32),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32,32,kernel_size = 1),
                                           nn.ReLU(inplace=True),) 
  

        
    def forward(self,aligned_fea,scale, scale2):

        scale_aware_fea = self.spacial_adaptive_conv(aligned_fea,scale, scale2)
        cascade_ps_result = self.cascade_ps_result(scale_aware_fea)
        std_shape_ls = []
        for i in range(self.ps_num):
            x_ratio = scale/(2**(i+1))
            y_ratio = scale2/(2**(i+1))
            # cascade_ps_result_std_shape = std_grid_sample(cascade_ps_result[i],x_ratio,y_ratio)
            cascade_ps_result_std_shape  =  self.sa_up(cascade_ps_result[i],x_ratio,y_ratio)
            # print(f'cascade_ps_result_std_shape {i}',cascade_ps_result_std_shape.shape)
            std_shape_ls.append(cascade_ps_result_std_shape)
        std_shape_fea = torch.stack(std_shape_ls,dim=1)
       
        # std_shape_fea = torch.cat(std_shape_ls,dim=1)
        # std_shape_fea = self.st_att(std_shape_fea)
        # std_shape_fea = self.reduce_0(std_shape_fea)
        # std_shape_fea = self.sa_adapt(std_shape_fea,scale,scale2)  
        # b,c,h,w = std_shape_fea.shape
        # std_shape_fea = std_shape_fea.contiguous().view(b,c,h*w).permute(0,2,1)
        
        # std_shape_fea = std_shape_fea.reshape(b,h,1,w,1,c).permute(0, 1, 3, 2, 4,5)
        # std_shape_fea = std_shape_fea.squeeze(3).squeeze(4).view(b,h*w,c)
        # std_shape_fea = self.soft_average(std_shape_fea)
        # std_shape_fea = std_shape_fea.view(b,c,h,w)
        # std_shape_fea = self.reduce(std_shape_fea)
        b,d,c,h,w = std_shape_fea.shape
        # std_shape_fea = std_shape_fea.view(b,3,-1,h,w)
        std_shape_fea = std_shape_fea.transpose(1,2)
        std_shape_fea = std_shape_fea.reshape(b,d*c,h,w)
        # std_shape_fea = std_shape_fea.squeeze(3).squeeze(4).view(b,h*w,c)
        # std_shape_fea = self.soft_average(std_shape_fea)
        std_shape_fea = self.soft_average(std_shape_fea)
        # std_shape_fea = self.st_att(std_shape_fea)
        std_shape_fea = self.sa_adapt(std_shape_fea,scale,scale2)  
        std_shape_fea = self.reduce(std_shape_fea)
       
       
        return  std_shape_fea

class Arb_upsample_ps1(nn.Module):
    def __init__(self,scale,scale2):
        super(Arb_upsample_ps1, self).__init__()
      
        self.ps_num = 1
        self.x_shape = []
        self.y_shape = []
        for i in range(self.ps_num):
            self.x_shape.append(scale/(2**(i+1)))
            self.y_shape.append(scale2/(2**(i+1)))
        self.sa_adapt = SA_adapt(32)
        self.spacial_adaptive_conv = SA_adapt(64)
        # self.reduce_0 =  torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.reduce =  torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.cascade_ps_result = Cascade_ps()
        self.cascade_ps_result = Cascade_ps_seprate(self.ps_num)
        self.sa_up = SA_upsample_light(32)
        # self.st_att = ST_att(inplanes  =32,planes = 32)
        # self.soft_average = Mlp_GEGLU(in_features=64, hidden_features=64, act_layer=nn.GELU)
        #显存瓶颈不在这儿
        self.soft_average = nn.Sequential(nn.Conv2d(32,32,kernel_size = 1,groups=32),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32,32,kernel_size = 1),
                                           nn.ReLU(inplace=True),) 
       

        
    def forward(self,aligned_fea,scale, scale2):

        scale_aware_fea = self.spacial_adaptive_conv(aligned_fea,scale, scale2)
        cascade_ps_result = self.cascade_ps_result(scale_aware_fea)
        std_shape_ls = []
        for i in range(self.ps_num):
            x_ratio = scale/(2**(i+1))
            y_ratio = scale2/(2**(i+1))
            # cascade_ps_result_std_shape = std_grid_sample(cascade_ps_result[i],x_ratio,y_ratio)
            cascade_ps_result_std_shape  =  self.sa_up(cascade_ps_result[i],x_ratio,y_ratio)
            # print(f'cascade_ps_result_std_shape {i}',cascade_ps_result_std_shape.shape)
            std_shape_ls.append(cascade_ps_result_std_shape)
        std_shape_fea = torch.stack(std_shape_ls,dim=1)
       
    
        b,d,c,h,w = std_shape_fea.shape
        # std_shape_fea = std_shape_fea.view(b,3,-1,h,w)
        std_shape_fea = std_shape_fea.transpose(1,2)
        std_shape_fea = std_shape_fea.reshape(b,d*c,h,w)
        # std_shape_fea = std_shape_fea.squeeze(3).squeeze(4).view(b,h*w,c)
        # std_shape_fea = self.soft_average(std_shape_fea)
        std_shape_fea = self.soft_average(std_shape_fea)
        # std_shape_fea = self.st_att(std_shape_fea)
        std_shape_fea = self.sa_adapt(std_shape_fea,scale,scale2)  
        std_shape_fea = self.reduce(std_shape_fea)
       
       
        return  std_shape_fea

class Arb_bilinear(nn.Module):
    def __init__(self,scale,scale2):
        super(Arb_bilinear, self).__init__()
      
        self.conv1 =  nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.reduce = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        
    def forward(self,aligned_fea,scale, scale2):
        
        B,T, H,W = aligned_fea.shape
        if round(scale2*W)%2!=0:
            tar_w = round(scale2*W)-1
        else:
            tar_w = round(scale2*W)
        if round(scale*H)%2!=0:
            tar_h = round(scale*H)-1
        else:
            tar_h = round(scale*H)    
        out_fea = F.interpolate(aligned_fea,size=(tar_h,tar_w),mode='bilinear',align_corners=False)
        out_fea = self.relu(self.conv1(out_fea))
        out = self.reduce(out_fea)

        return  out


if __name__=='__main__':
    s = [1.2, 1.4 ,1.6 ,1.8 ,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]
    fea = torch.rand(1,64,64,64).cuda()
    test_np_data = torch.rand((256,256,3)).numpy()
    ratio = [0.295,0.351,0.398,0.445,0.5,0.545,0.600,0.648,0.695,0.75,0.796,0.850,0.895,0.945,1.0]
    for ix,scale in enumerate(s):
        # if ix>0 and ix < len(s)-1:
        #     scale_2 = s[ix-1]
        # else:
        scale_2 = scale
        arb = Arb_upsample(scale,scale_2).cuda()
        arb.eval()
        out = arb(fea,scale,scale_2)
        print(math.floor(ratio[ix]*256))
        out_img = imresize_np(test_np_data,ratio[ix],antialiasing=True)
        print('bicubic shape',out_img.shape,ratio[ix])
        print(out.shape)
   