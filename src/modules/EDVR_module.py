import torch.nn as nn
import torch
import torchvision
from mmcv.cnn import ConvModule

from modules.general_module import ResidualBlockNoBN,make_layer
# from modules.edvr_net import PCDAlignment, TSAFusion
from mmcv.runner import load_checkpoint
from utils.logger import get_root_logger


class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.
    We use EDVR-M in IconVSR. To adopt pretrained models, please
    specify "pretrained".
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
        pretrained (str): The pretrained model path. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 out_channel=3,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True,
                 pretrained=None):

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

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        """Forward function for EDVRFeatureExtractor.
        Args:
            x (Tensor): Input tensor with shape (n, t, 3, h, w).
        Returns:
            Tensor: Intermediate feature with shape (n, mid_channels, h, w).
        """

        n, t, c, h, w = x.size()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.contiguous().view(-1, c, h, w)))
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

        return feat

class EDVRalign(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.
    We use EDVR-M in IconVSR. To adopt pretrained models, please
    specify "pretrained".
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
        pretrained (str): The pretrained model path. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 out_channel=3,
                 mid_channels=64,
                 num_frames=4,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 
                 with_tsa=False,
                 pretrained=None):

        super().__init__()
       
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
                center_frame_idx=1)
        else:
            self.fusion = nn.Conv2d((num_frames-1) * mid_channels, mid_channels, 1,
                                    1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        """Forward function for EDVRFeatureExtractor.
        Args:
            x (Tensor): Input tensor with shape (n, t, 3, h, w).
        Returns:
            Tensor: Intermediate feature with shape (n, mid_channels, h, w).
        """

        n, t, c, h, w = x.size()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.contiguous().view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        # L2
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        # L3
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        # pcd alignment

        aligned_feat = []
        for i in range(t-1):
            feat1 = [
                l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(),
                l3_feat[:, i, :, :, :].clone()
            ]
            feat2 = [
                l1_feat[:, i+1, :, :, :].clone(), l2_feat[:, i+1, :, :, :].clone(),
                l3_feat[:, i+1, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_alignment(feat1, feat2))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)

        return feat
if __name__=='__main__':
    edvr = EDVRFeatureExtractor(pretrained='../../pretrained_weight/edvr.pth')
    edvr = edvr.cuda()
    (n, t, c, h, w) = 2,5,3,64,64
    test_in = torch.randn(n, t, c, h, w)
    test_in = test_in.cuda()
    out_data = edvr(test_in)
    print(out_data.shape)