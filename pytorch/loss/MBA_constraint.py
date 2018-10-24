import torch
import torch.nn as nn
from torch.autograd import Variable



class KLLoss(nn.Module):
    def __init__(self, weight=0, size_average=True):
        super(KLLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def kl_divergence(self, inputs, target):
        inputs = inputs.detach()
        target = target.detach()
        log_inputs = torch.log(inputs) 
        criterion = nn.KLDivLoss(size_average=self.size_average)
        return criterion(log_inputs, target)
        
    def forward(self, multi_mask):
        branch_number = len(multi_mask)
        b, c, h, w = multi_mask[0].shape
        kl_loss = Variable(torch.zeros(b)).cuda()
        for i in range(branch_number):
            for j in range(branch_number):
                if i == j:
                    continue
                else:
                    kl_loss += self.kl_divergence(multi_mask[i], multi_mask[j])
        kl_loss *= self.weight
        return kl_loss.mean()
