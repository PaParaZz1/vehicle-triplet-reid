from importlib import import_module
import numpy as np
import torch
import torch.nn as nn

BABKBONE = "backbone.resnet_interface"
HEAD = "head.multi_branch_attention"

class ReIDNetwork(nn.Module):
    def __init__(self, backbone, head, feature_dim, embedding_dim, pool, branch_number):
        super(ReIDNetwork, self).__init__()
        package_backbone = import_module(BABKBONE)
        self.backbone = package_backbone.Iresnet(pretrained=True, is_backbone=True, backbone_type=backbone)
        package_head = import_module(HEAD)
        self.head = package_head.Ihead(feature_dim, embedding_dim, pool, branch_number, head)
        
    def forward(self, x):
        x = self.backbone(x)
        embedding_feature, multi_mask = self.head(x)
        return embedding_feature, multi_mask

if __name__ == "__main__":
    from torch.autograd import Variable
    backbone = "resnet50"
    head = "MultiBranchAttention"
    feature_dim = 2048
    embedding_dim = 128
    pool = 'concat'
    branch_number = 5
    model = ReIDNetwork(backbone, head, feature_dim, embedding_dim, pool, branch_number).cuda()
    inputs = torch.rand(4,3,256,128).cuda()
    inputs = Variable(inputs)
    feature, mask = model(inputs)
    print(feature.shape)
    print(mask[0].shape)
    print("ReIDNetwork test pass")
