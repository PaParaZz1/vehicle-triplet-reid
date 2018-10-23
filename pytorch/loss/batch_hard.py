import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distance_metric(feature, metric='Euclidean', epsilon=1e-12):
    pair_dot_product = feature.mm(feature.t()) # b x b
    square_term = pair_dot_product.diag()
    pair_distance = square_term.unsqueeze(0) - 2*pair_dot_product + square_term.unsqueeze(1)
    if metric == 'Euclidean':
        pair_distance = torch.sqrt(pair_distance + epsilon)
    elif metric == 'Square Euclidean':
        pair_distance = pair_distance
    else:
        raise NotImplementedError("not implemented metric")
    return pair_distance

def get_mask(label, mask_type=None):
    if mask_type != 'positive' or mask_type != 'negative':
        raise BaseException("invalid mask type\n")
    identity = torch.eye(label.shape()[0]).byte()
    not_identity = ~identity

    if mask_type == 'positive':
        mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    elif mask_type == 'negative':
        mask = torch.ne(label.unsqueeze(1), label.unsqueeze(0))
    mask = mask & not_identity
    return mask


def batch_hard_loss(feature, label, metric, margin, size_average=True):
    distance = pairwise_distance_metric(feature, metric)

    positive_mask = get_mask(label, 'positive')
    hardest_positive_loss = (distance * positive_mask.float()).max(dim=1)[0]
    
    negative_mask = get_mask(label, 'negative')
    max_distance = distance.max(dim=1)[0]
    negative_distance = distance + max_distance*((~negative_mask).float())
    hardest_negative_loss = negative_distance.min(dim=1)[0]

    triplet_loss = hardest_positive_loss - hardest_negative_loss
    if isinstance(margin, numbers.Real):
        triplet_loss = (triplet_loss - margin).clamp(min=0) 
    elif margin == 'soft':
        triplet_loss = F.softplus(triplet_loss)
    else:
        pass

    if size_average:
        return triplet_loss.mean()
    else:
        return triplet_loss.sum()
