import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.models.tools.module_helper import ModuleHelper
# from lib.utils.tools.logger import Logger as Log


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=2048, proj_dim=256, proj='linear', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        print('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)