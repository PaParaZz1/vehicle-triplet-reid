import torch
import torch.nn as nn
import torch.nn.functional as F



class KLLoss(nn.Module):
    def __init__(self, weight=0, size_average=True):
        super(KLLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def kl_divergence(inputs, target, size_average):
        log_inputs = torch.log(inputs) 
        return F.kl_div(log_inputs, target, size_average=size_average)
        
    def forward(self, multi_mask):
        branch_number = len(multi_mask)
        kl_loss = None
        for i in range(branch_number):
            for j in range(branch_number):
                if i == j:
                    continue
                else:
                    kl_loss += kl_divergence(multi_mask[i], multi_mask[j], self.size_average)
        kl_loss *= weight
