import numpy as np
import torch
import torch.nn as nn

from modules import *

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.conv1 = ConvBlockSequentail(in_channels = input_dim, out_channels = 64, kernel_size = 1, init_type = "xavier", activation = nn.ReLU(), use_batchnorm = True)
        self.conv2 = ConvBlockSequentail(in_channels = 64, out_channels = 1, kernel_size = 1, init_type = "xavier", activation = None, use_batchnorm = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        feature = x
        if h != w
            raise BaseException("input feature h not match w\n")
        mask = self.conv1(x)
        mask = self.conv2(mask)
        mask = self.sigmoid(mask)
        identity = torch.from_numpy(np.eye(h).astype(np.float32)).repeat(b)
        mask = mask + identity
        mask_map = torch.zeros(b, c, h, w)
        torch.addcmul(mask_map, 1, mask, feature)
        return mask_map

if __name__ == "__main__":
    from torch.autograd import Variable
    inputs = torch.rand(8,128,224,224).cuda()
    inputs = Variable(inputs)
    model = AttentionModule(128).cuda()
    out = model(inputs)
    print("end")
