from importlib import import_module
import numpy as np
import torch
import torch.nn as nn

BABKBONE = "backbone.resnet_interface"
HEAD = "head.MBA_KL"

class ReIDNetwork(nn.Module):
    def __init__(self, backbone, head, feature_dim, embedding_dim, pool, branch_number):
        super(ReIDNetwork, self).__init__()
        package_backbone = import_module(BABKBONE)
        self.backbone = package_backbone.Iresnet(pretrained=True, is_backbone=True, backbone_type=backbone)
        package_head = import_module(HEAD)
        self.head = package_head.MultiBranchAttention(feature_dim, embedding_dim, pool, branch_number)
        
    def forward(self, x):
        x = self.backbone(x)
        embedding_feature, multi_mask = self.head(x)
        return embedding_feature, multi_mask
if __name__ == "__main__":
    backbone = "resnet50"
    head = "head.MBA_KL"
    feature_dim = 512
    embedding_dim = 128
    pool = 'addition'
    branch_number = 5
    model = ReIDNetwork(backbone, head, feature_dim, embedding_dim, pool, branch_number)
    print("ReIDNetwork test pass")
