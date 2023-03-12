import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
import math

def default_init_weights(module, scale=1):
    """Initialize network weights.
    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
class SA_adapt(nn.Module):
    def __init__(self, channels):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1)

    def forward(self, x, scale, scale2):
        mask = self.mask(x)
        adapted = self.adapt(x, scale, scale2)

        return x + adapted * mask
class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
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

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
class ResidualBlockNoBN_inv(nn.Module):
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        # self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = involution(mid_channels,3,1)
        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x
    
class ResidualBlocksWithInputConv_insert_sav(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30,K=1):
        super().__init__()

        main = []
        self.num_blocks  =num_blocks
        self.K  =K
        # a convolution used to match the channels of the residual blocks
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        # residual blocks
        for i in range(num_blocks):
            main.append(ResidualBlockNoBN())
        
        self.main = nn.Sequential(*main)
        sa_adapt = []
        for i in range(num_blocks // self.K):
            sa_adapt.append(SA_adapt(64))
        self.sa_adapt = nn.Sequential(*sa_adapt)

    def forward(self, feat,scale1,scale2):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, hw)
        """
        feat = self.relu1(self.conv1(feat))
        for i in range( self.num_blocks):
            feat = self.main[i](feat)
            if (i+1)%self.K==0:
                feat = self.sa_adapt[i//self.K](feat,scale1,scale2)
        return feat
    
    
class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, hw)
        """
        return self.main(feat)

class ResidualBlocksWithInputConv_inv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks-10, mid_channels=out_channels)
        )
        main.append(
            make_layer(
                 ResidualBlockNoBN_inv, 10, mid_channels=out_channels)
        )
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, hw)
        """
        return self.main(feat)

def flow_warp(x,
              flow,
              ret_mask = False,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    if ret_mask:
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, grid_flow, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        res = output * mask

      
        return res, mask
    
    return output
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )
def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ResBlock, self).__init__()
        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes,
                                   3, stride, 1, bias=False)
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv_woact(out_planes, out_planes, 3, 1, 1)
        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)
        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x
c = 16

class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = ResBlock(64, c,stride=1)
        self.conv2 = ResBlock(c, 2*c)
        self.conv3 = ResBlock(2*c, 4*c)


    def forward(self, x, flow):
        x = self.conv1(x)
        f1 = flow_warp(x, flow.permute(0, 2, 3, 1))
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f2 = flow_warp(x, flow.permute(0, 2, 3, 1))
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f3 = flow_warp(x, flow.permute(0, 2, 3, 1))

        return [f1, f2, f3]
class FusionNet(nn.Module):

    def __init__(self):
        super(FusionNet, self).__init__()
        self.down0 = ResBlock(12, 2*c,stride=1)
        self.down1 = ResBlock(4*c, 4*c)
        self.down2 = ResBlock(8*c, 8*c)
        # self.down3 = ResBlock(16*c, 16*c)

        self.up0 = deconv(16*c, 4*c)
        self.up1 = deconv(8*c, 2*c)
        self.mix = nn.Conv2d(4*c,c,3,1,1)
        # self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 4, 3, 1, 1)

    def forward(self, img0, img1, flow_forward,flow_backward, c0, c1, flow_gt):
        warped_img0 = flow_warp(img0, flow_backward.permute(0, 2, 3, 1))
        warped_img1 = flow_warp(img1, flow_forward.permute(0, 2, 3, 1))
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = flow_warp(img0, flow_gt[:, :2])
            warped_img1_gt = flow_warp(img1, flow_gt[:, 2:4])
        s0 = self.down0(torch.cat((img0,img1,warped_img0, warped_img1), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        # s2 = self.mid(s2)
        # s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        # s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up1(torch.cat((x, s1), 1))
        x = self.mix(torch.cat((x, s0), 1))
        # x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt

class Predict(nn.Module):

    def __init__(self):
        super(Predict, self).__init__()
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
    def forward(self,img0,img1,c0,c1,flow_forard,flow_backward):
        c0 = self.contextnet(c0, flow_backward)
        c1 = self.contextnet(c1, flow_forard)
        # flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
        #                      align_corners=False) * 2.0
        flow_gt = None
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1,flow_forard,flow_backward, c0, c1, flow_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        return pred
class RefineUNet(nn.Module):
    def __init__(self):
        super(RefineUNet, self).__init__()

        self.scale = 2
        self.nf = 64
        self.conv1 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
        self.conv2 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
        self.lrelu = nn.ReLU()
        self.NN = nn.UpsamplingNearest2d(scale_factor=2)        
        self.enc1 = nn.Conv2d((4*self.nf)//self.scale//self.scale, self.nf, [4, 4], 2, [1, 1])
        self.enc2 = nn.Conv2d(self.nf, 2*self.nf, [4, 4], 2, [1, 1])
        self.enc3 = nn.Conv2d(2*self.nf, 4*self.nf, [4, 4], 2, [1, 1])
        self.dec0 = nn.Conv2d(4*self.nf, 4*self.nf, [3, 3], 1, [1, 1])
        self.dec1 = nn.Conv2d(4*self.nf + 2*self.nf, 2*self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc2
        self.dec2 = nn.Conv2d(2*self.nf + self.nf, self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc1
        self.dec3 = nn.Conv2d(self.nf, 1+args.img_ch, [3, 3], 1, [1, 1]) ## input added with warped image

    def forward(self, concat):
        enc1 = self.lrelu(self.enc1(concat))
        enc2 = self.lrelu(self.enc2(enc1))
        out = self.lrelu(self.enc3(enc2))       
        out = self.lrelu(self.dec0(out))
        out = self.NN(out)      
        out = torch.cat((out,enc2),dim=1)
        out = self.lrelu(self.dec1(out))        
        out = self.NN(out)
        out = torch.cat((out,enc1),dim=1)
        out = self.lrelu(self.dec2(out))        
        out = self.NN(out)
        out = self.dec3(out)
        return out
class block(nn.Module):
    
    def __init__(self,nf = 64):
        super().__init__()

        self.conv3d_ks2 = nn.Conv3d(nf, nf, (2,3,3), stride=(1,1,1), padding=(0,1,1))
        self.conv3d_ks3 = nn.Conv3d(nf, nf, (3,3,3), stride=(1,1,1), padding=(0,1,1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, feat):
        identity = feat
        feat_ks2 = self.lrelu(self.conv3d_ks2(feat))
        feat_ks3 = self.lrelu(self.conv3d_ks3(feat))
        feat_out = torch.cat([feat_ks2[:,:,0:1,:,:],feat_ks3,feat_ks2[:,:,1:2,:,:]],dim=2)
        return feat_out+identity


# temporal fuse atten

    




class ref_attention(nn.Module):
    def __init__(self,nf = 64):
        super(ref_attention, self).__init__()
        self.ta1 = nn.Conv2d(nf, nf, 3, padding=1)
        self.ta2 = nn.Conv2d(
            nf, nf, 3, padding=1)
        self.feat_fusion = nn.Sequential( nn.Conv2d(3*nf, nf, 3, padding=1),
                                         nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                         )
        self.sa1 = nn.Sequential( nn.Conv2d(3*nf, nf, 3, padding=1),
                                         nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                         )
        self.sa2 = nn.Sequential( nn.Conv2d(2*nf, nf, 3, padding=1),
                                         nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                         )
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.spatial_attn_add2 = nn.Conv2d(nf, nf, 1)
        
    def forward(self, x):
        # bcthw -> btchw
        x = x.permute(0,2,1,3,4).contiguous()
        n, t, c, h, w = x.size()
        hs_ref = self.ta1(
            x[:, 1, :, :, :].clone())
        emb = self.ta2(x.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)
        att_l = []
        for i in range(t):
            neighbor = emb[:, i, :, :, :]
            corr = torch.sum(neighbor * hs_ref, 1)  # (n, h, w)
            att_l.append(corr.unsqueeze(1))  # (n, 1, h, w)
        atten_m = F.softmax(torch.cat(att_l, dim=1), dim=1)   # (n, t, h, w)
        atten_m = atten_m.unsqueeze(2).expand(n, t, c, h, w)
        atten_m = atten_m.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        feat = torch.mul(atten_m, x.view(n, -1, h, w))
        
        feat_time = self.feat_fusion(feat)
        x = x.view(n, -1, h, w)
        attn_sp = self.sa1(x)
        attn_max = self.max_pool(attn_sp)
        attn_avg = self.avg_pool(attn_sp)
        attn = self.sa2(torch.cat([attn_max, attn_avg], dim=1))
        attn = self.upsample(attn)
        attn_space = self.spatial_attn_add2((attn))
        attn_space = torch.sigmoid(attn_space)
        
        feat_space  =feat_time * attn_space
        return (feat_time+feat_space)/2.0

class TFA_with_onlyTSA(nn.Module):
    
    def __init__(self,nf = 64):
        super(TFA_with_onlyTSA, self).__init__()
        # ks = 1 跟 ks = 3 混着来？？？
    
        self.att =  ref_attention()
        self.short_cut_out =  nn.Conv2d(3*nf, nf, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        #x b,c,t,h,w
        b,c,t,h,w = x.size()
        out = self.lrelu(self.short_cut_out(x.contiguous().view(b,c*t,h,w)))+self.att(x)
        return out


if __name__=='__main__':
    # test_block = make_layer(ResidualBlockNoBN,5,mid_channels=64)
    # n, c, h, w = 7,64,64,64
    # in_data = torch.randn(n,c,h,w)
    # out_data = test_block(in_data)
    # print(out_data.shape)
    # (n, h, w) = 2,64,64
    #
    # test_flow = torch.randn(n, h, w, 2)
    # test_img = torch.randn(n,3,h,w)
    # warped = flow_warp(test_img,test_flow)
    # # print(warped.shape)
    # fn = Predict()
    # img0 = torch.randn(n,3,h,w)
    # img1 = torch.randn(n,3,h,w)
    # c0 = torch.randn(n,64,h,w)
    # c1 = torch.randn(n, 64, h, w)
    # flow_forward = torch.randn(n,2,h,w)
    # flow_backward = torch.randn(n,2,h,w)
    # flow_gt = None
    # res = fn(img0,img1,c0,c1,flow_forward,flow_backward)
    # print(res.shape)
    
    ##params 1.135744
    tfa = TFA()
    print("#params" , sum([p.numel() for p in tfa.parameters()])/1000000)
    in_data = torch.randn(1,64,3,64,64)
    out = tfa(in_data)
    print(out.shape)