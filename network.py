from importlib import import_module
import numpy as np
import torch
import torch.nn as nn

class ReIDNetwork(nn.Module):
    def __init__(self, backbone, head):
        super(ReIDNetwork, self).__init__()
        self.backbone = import_module(backbone)
        self.head = import_module(head)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
if __name__ == "__main__":
    print("ReIDNetwork test pass")
