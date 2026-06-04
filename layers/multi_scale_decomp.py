'''
2026.03.09:模型组件设计
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDecomp(nn.Module):
    """
    Multi-scale moving average decomposition
    """

    def __init__(self, kernel_sizes=[3,5,7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def moving_avg(self, x, kernel):
        padding = kernel // 2
        x = F.pad(x, (0,0,padding,padding), mode='replicate')
        x = F.avg_pool1d(x.permute(0,2,1), kernel, stride=1)
        return x.permute(0,2,1)
 
    def forward(self, x):

        trends = []

        for k in self.kernel_sizes:
            trend = self.moving_avg(x, k)
            trends.append(trend)

        return trends