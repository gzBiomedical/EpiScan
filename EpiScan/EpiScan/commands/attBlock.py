import torch.nn as nn
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
       

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Rose_block(nn.Module):
    def __init__(self, channel,outchannel,ratio=16):
        super(Rose_block, self).__init__()
        self.outchannel = outchannel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, outchannel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        b, c, _,_ = x.size()
        y = self.avg_pool(x)
        # print('111111111111')
        # print(y.shape)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1)
        # return x[:,-self.outchannel:,:] * y
        x = x.squeeze(-1)
        return x[:,-self.outchannel:,:]* y.expand_as(x[:,-self.outchannel:,:])


def rotate_matTorch(x,axis,radian):
    eps = 1e-5
    axistemp = axis.squeeze(2).transpose(0, 1)
    axistemp = torch.div(axistemp, (torch.linalg.norm(axis,dim=(1,2))+eps))
    axistemp = torch.mul(axistemp,radian)
    axisend= axistemp.transpose(0, 1)
    # Aaxis = torch.empty(len(axis[:,:,:]), 3, 3)
    # for bi in range(len(axis[:,:,:])):
    #     Aaxis[bi,:,:] = torch.cross(torch.eye(3), axisend[bi,:])
    Aaxis = torch.cross(torch.eye(3).cuda(), axisend)
    # print(Aaxis)
    # print('-----------gggggg------------------------')
    rot_matrix = torch.linalg.matrix_exp(Aaxis)
    # print(rot_matrix)
    # print('-----------uuuuuu------------------------')
    x1 = torch.matmul(rot_matrix, x)
    return x1
