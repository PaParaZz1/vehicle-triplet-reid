import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from modules import *

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.conv1 = ConvBlockSequential(in_channels = input_dim, out_channels = 64, kernel_size = 1, init_type = "xavier", activation = nn.ReLU(), use_batchnorm = True)
        self.conv2 = ConvBlockSequential(in_channels = 64, out_channels = 1, kernel_size = 1, init_type = "xavier", activation = None, use_batchnorm = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        feature = x
        mask = self.conv1(x)
        mask = self.conv2(mask)
        mask = self.sigmoid(mask)
        identity = Variable(torch.ones(h,w).repeat(b, 1, 1, 1)).cuda()
        mask_residual = mask + identity
        mask_residual = mask_residual.repeat(1,c,1,1)
        mask_map = mask_residual * feature
        return mask_map, mask

class MultiBranchAttention(nn.Module):
    def __init__(self, input_dim, output_dim, pool, branch_number):
        super(MultiBranchAttention, self).__init__()
        pool_options = ['concat', 'addition']
        if pool is None:
            raise BaseException("not set pool option\n")
        elif pool in pool_options:
            self.pool = pool
        else:
            raise BaseException("invalid pool option\n")

        self.branch_number = branch_number
        self.multi_attention = {}
        for i in range(self.branch_number):
            self.multi_attention['attention_branch{}'.format(i)] = AttentionModule(input_dim).cuda()    
        if self.pool == 'concat':
            self.fc = nn.Sequential(
                        FCBlockSequential(input_dim*branch_number, input_dim, init_type="xavier", activation=nn.ReLU(), use_batchnorm=True),
                        FCBlockSequential(input_dim, 1024, init_type="xavier", activation=nn.ReLU(), use_batchnorm=True))
        elif self.pool == 'addition':
            self.fc = nn.Sequential(FCBlockSequential(input_dim, 1024, init_type="xavier", activation=nn.ReLU(), use_batchnorm=True))
        self.output = FCBlockSequential(1024, output_dim, init_type="xavier", activation=None, use_batchnorm=False)

    def forward(self, x):
        multi_mask = []
        multi_mask_map = []
        for i in range(self.branch_number):
            item1, item2 = self.multi_attention['attention_branch{}'.format(i)](x) 
            multi_mask_map.append(item1)
            multi_mask.append(item2) 
        if self.pool == 'concat':
            mask_map_pool = torch.cat(multi_mask_map, dim=1)
        elif self.pool == 'addition':
            mask_map_pool = torch.stack(multi_mask_map, dim=0)
            mask_map_pool = mask_map_pool.sum(0)
        x = torch.mean(mask_map_pool, dim=3)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        x = self.output(x)
        return x, multi_mask
        
def Ihead(input_dim, output_dim, pool, branch_number, head_type=None):
    if head_type == None:
        raise BaseException("not set head type\n")
    elif head_type == "MultiBranchAttention":
        return MultiBranchAttention(input_dim, output_dim, pool, branch_number)
    else:
        raise NotImplementedError("not implemented head type:{}".format(head_type))

if __name__ == "__main__":
    # attention module test
    inputs = torch.rand(8,128,224,224).cuda()
    inputs = Variable(inputs)
    model = AttentionModule(128).cuda()
    out = model(inputs)
    print("attention module test pass")

    # multi attention test
    multi_model = MultiBranchAttention(128, 128, pool='addition', branch_number=5).cuda()
    out, multi_mask = multi_model(inputs)
    print("multi attention test pass")
