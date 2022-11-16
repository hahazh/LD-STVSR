from  src.modules.flow_module import SPyNet
import torch
import os
import src.utils.flow_config as flow_config
import numpy as np
class Off_line_generator:
    def __init__(self):
        #init cuda env
        # os.environ["CUDA_VISIBLE_DEVICES"] = flow_config.Flow_config.device
        #load weight
        self.flownet = SPyNet(pretrained=None).cuda()

        pre_weight = torch.load('../../pretrained_weight/spynet.pth')
        pre_weight['mean'] = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        pre_weight['std'] = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.flownet.load_state_dict(pre_weight)

    def generate_flow(self,hrs):

        if isinstance(hrs,np.ndarray):
            hrs = torch.from_numpy(hrs).cuda()
        elif isinstance(hrs,torch.Tensor):
            hrs = hrs.cuda()
        n, t, c, h, w = hrs.size()
        hrs_1 = hrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        hrs_2 = hrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.flownet(hrs_1, hrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.flownet(hrs_2, hrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward