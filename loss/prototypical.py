# coding=utf-8
import numpy as np
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support, mode, margin=0):
        super(PrototypicalLoss, self).__init__()
        assert mode in ['avg', 'max', 'min', 'maxmin', 'pchd']
        self.n_support = n_support
        self.mode = mode
        self.margin = margin


    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support, self.mode, self.margin)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def point_convex_hull_dist(point, hull, mode='min'):
    '''
    Compute the distance between a point and a convex hull(PCHD), including
    minimum and maximum distance 
    '''
    # point: 1 * D
    # hull: set_size * D
    assert mode in ['min', 'max']
    num_points, num_dims = hull.shape
    # coeff_ini = np.random.rand(num_points, 1)
    coeff_ini = np.array([0.33, 0.33, 0.33])
    cons = ({'type': 'ineq', 'fun': lambda coeff: coeff[0]},
            {'type': 'ineq', 'fun': lambda coeff: coeff[1]},
            {'type': 'ineq', 'fun': lambda coeff: coeff[2]},
            {'type': 'ineq', 'fun': lambda coeff: 1 - coeff[0]},
            {'type': 'ineq', 'fun': lambda coeff: 1 - coeff[1]},
            {'type': 'ineq', 'fun': lambda coeff: 1 - coeff[2]},
            {'type': 'eq', 'fun': lambda coeff: 1 - np.sum(coeff)}
            )
    if mode == 'min':
        def min_fun(p, h):
            def fun(coeff):
                res = coeff[0] * h[0]
                for i in range(h.shape[0]-1):
                    res += coeff[i+1] * h[i+1]
                return np.power(res-p, 2).sum()
            return fun

        res = minimize(min_fun(point, hull), coeff_ini, constraints=cons)
    else:  # max
        def max_fun(p, h):
            def fun(coeff):
                res = coeff[0] * h[0]
                for i in range(h.shape[0]-1):
                    res += coeff[i+1] * h[i+1]
                return -np.power(res-p, 2).sum()
            return fun

        res = minimize(max_fun(point, hull), coeff_ini, constraints=cons)

    # dist = res.fun
    # if res.success:
    #     return res.x  # return coefficients
    # else:
    #     raise Exception
    return res


def prototypical_loss(input, target, n_support, mode, margin):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    assert mode in ['avg', 'max', 'min', 'maxmin', 'pchd']

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    if mode == 'avg':
        prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
        dists = euclidean_dist(query_samples, prototypes)
    elif mode == 'max':
        dists = []
        for idx_list in support_idxs:
            dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
            dist, _ = torch.max(dist_all, dim=1)
            dists.append(dist)
        dists = torch.stack(dists).t()
    elif mode == 'min':
        # dists = []
        # for idx_list in support_idxs:
        #     dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
        #     dist, _ = torch.min(dist_all, dim=1)
        #     dists.append(dist)
        # dists = torch.stack(dists).t()

        n_selected_samples = 16
        n_criterion = 4
        dists = []
        for idx_list in support_idxs:
            dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
            dist, _ = torch.min(dist_all, dim=1)
            dists.append(dist)
        dists = torch.stack(dists).t()
        dists_sorted, _ = torch.sort(dists)

        # hard mining: choose 16 most difficult query samples for loss computation
        diff = -n_criterion * dists_sorted[:, 0] + torch.sum(dists_sorted[:, 1:n_criterion+1], dim=1)
        _, inds = torch.sort(diff)
        selected_inds = inds[:n_selected_samples]
        dists = dists[selected_inds]

        log_p_y = F.log_softmax(-dists, dim=1)
        target_inds = selected_inds.expand(n_classes, n_selected_samples).long().t()
        loss_val = -log_p_y.gather(1, target_inds).view(-1).mean()
        return loss_val

    elif mode == 'maxmin':
        hard = False
        if hard:
            n_selected_samples = 16
            n_criterion = 4
            min_eucli_dists = []
            max_self_eucli_dists = []
            for i, idx_list in enumerate(support_idxs):
                dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
                min_dist, _ = torch.min(dist_all, dim=1)
                max_dist = torch.max(dist_all[i])
                min_eucli_dists.append(min_dist)
                max_self_eucli_dists.append(max_dist)
            min_eucli_dists = torch.stack(min_eucli_dists).t()
            # max_self_eucli_dists = torch.stack(max_self_eucli_dists)
            dists_sorted, _ = torch.sort(min_eucli_dists)

            # hard mining: choose 16 most difficult query samples for loss computation
            diff = -n_criterion * dists_sorted[:, 0] + torch.sum(dists_sorted[:, 1:n_criterion+1], dim=1)
            _, inds = torch.sort(diff)
            selected_inds = inds[:n_selected_samples]
            dists = min_eucli_dists[selected_inds]
            # print(dists.shape)
            for i in range(n_selected_samples):
                dists[i][selected_inds[i]] = max_self_eucli_dists[selected_inds[i]]

            log_p_y = F.log_softmax(-dists, dim=1)
            target_inds = selected_inds.expand(n_classes, n_selected_samples).long().t()
            # print(target_inds)
            # print(-log_p_y.gather(1, target_inds))
            loss_val = -log_p_y.gather(1, target_inds).view(-1).mean()
            return loss_val
        else:
            dists = []
            max_self_eucli_dists = []
            for i, idx_list in enumerate(support_idxs):
                dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
                min_dist, _ = torch.min(dist_all, dim=1)
                max_dist = torch.max(dist_all[i])
                dists.append(min_dist)
                max_self_eucli_dists.append(max_dist)
            dists = torch.clamp(torch.stack(dists).t() - margin, min=0.0)
            # dists = torch.stack(dists).t()
            for i in range(n_classes):
                dists[i][i] = max_self_eucli_dists[i]

    else:  # mode == 'pchd'
        hard = False
        # import time
        # t1 = time.time()
        n_pchd = 4
        min_eucli_dists = []
        max_self_eucli_dists = []
        for i, idx_list in enumerate(support_idxs):
            dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
            min_dist, _ = torch.min(dist_all, dim=1)
            max_dist = torch.max(dist_all[i])
            min_eucli_dists.append(min_dist)
            max_self_eucli_dists.append(max_dist)
        min_eucli_dists = torch.stack(min_eucli_dists).t()
        # max_self_eucli_dists = torch.stack(max_self_eucli_dists)
        dists_sorted, min_inds = torch.sort(min_eucli_dists)

        if hard:
            # hard mining: choose 16 most difficult query samples for loss computation
            n_selected_samples = 16
            diff = -n_pchd * dists_sorted[:, 0] + torch.sum(dists_sorted[:, 1:n_pchd+1], dim=1)
            _, inds = torch.sort(diff)
            selected_inds = inds[:n_selected_samples]
            dists = min_eucli_dists[selected_inds]
            # to reduce computational cost, only compute accurate PCHD between the query sample
            # and 4 nearest convex hulls, and use minimum distance between the query sample and
            # support samples to approximate PCHD for other support samples
            for i in range(n_selected_samples):
                for j in range(n_pchd):
                    '''
                    If you directly convert tensors to ndarrays using numpy(), the distance cannot
                    be computed correctly. I don't know why.
                    '''
                    # point = query_samples[selected_inds[i]].detach().numpy()
                    # hull = input_cpu[support_idxs[min_inds[selected_inds[i], j+1]]].detach().numpy()
                    point = query_samples[selected_inds[i]].detach().tolist()
                    hull = input_cpu[support_idxs[min_inds[selected_inds[i], j+1]]].detach().tolist()
                    point_ = np.array(point)
                    hull_ = np.array(hull)
                    res = point_convex_hull_dist(point_, hull_)
                    if res.success:
                        coeff = torch.FloatTensor(res.x)
                        proto = coeff[0] * input_cpu[support_idxs[min_inds[selected_inds[i], j+1]]][0]
                        for k in range(2):
                            proto += coeff[k+1] * input_cpu[support_idxs[min_inds[selected_inds[i], j+1]]][k+1]
                        dist = torch.pow(query_samples[selected_inds[i]]-proto, 2).sum()
                        if dists[i, min_inds[selected_inds[i], j+1]] > dist:
                            dists[i, min_inds[selected_inds[i], j+1]] = dist
                dists[i][selected_inds[i]] = max_self_eucli_dists[selected_inds[i]]

            log_p_y = F.log_softmax(-dists, dim=1)
            target_inds = selected_inds.expand(n_classes, n_selected_samples).long().t()
            loss_val = -log_p_y.gather(1, target_inds).view(-1).mean()
            # t2 = time.time()
            # print('{}s'.format(t2-t1))
            return loss_val
        else:
            dists = min_eucli_dists
            for i in range(n_classes):
                for j in range(n_pchd):
                    point = query_samples[i].detach().tolist()
                    hull = input_cpu[support_idxs[min_inds[i, j+1]]].detach().tolist()
                    point_ = np.array(point)
                    hull_ = np.array(hull)
                    res = point_convex_hull_dist(point_, hull_)
                    if res.success:
                        coeff = torch.FloatTensor(res.x)
                        proto = coeff[0] * input_cpu[support_idxs[min_inds[i, j+1]]][0]
                        for k in range(2):
                            proto += coeff[k+1] * input_cpu[support_idxs[min_inds[i, j+1]]][k+1]
                        dist = torch.pow(query_samples[i]-proto, 2).sum()
                        if dists[i, min_inds[i, j+1]] > dist:
                            dists[i, min_inds[i, j+1]] = dist
                dists[i][i] = max_self_eucli_dists[i]

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    # _, y_hat = log_p_y.max(2)
    # acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    # return loss_val, acc_val
    return loss_val


class PrototypicalTripletLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support, mode, margin=None):
        super(PrototypicalTripletLoss, self).__init__()
        assert mode in ['avg', 'max', 'min']
        self.n_support = n_support
        self.mode = mode
        self.margin = margin
        if margin is not None:
            self.loss_fn = nn.MarginRankingLoss(margin=margin)
        else:
            self.loss_fn = nn.SoftMarginLoss()


    def forward(self, input, target):
        return prototypical_triplet_loss(self.loss_fn, input, target, self.n_support, self.mode, self.margin)


def prototypical_triplet_loss(loss_fn, input, target, n_support, mode, margin):

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
   
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]

    if mode == 'avg':
        prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
        dists = euclidean_dist(query_samples, prototypes)
    elif mode == 'max':
        dists = []
        for idx_list in support_idxs:
            dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
            dist, _ = torch.max(dist_all, dim=1)
            dists.append(dist)
        dists = torch.stack(dists)
    elif mode == 'min':
        dists = []
        for idx_list in support_idxs:
            dist_all = euclidean_dist(query_samples, input_cpu[idx_list])
            dist, _ = torch.min(dist_all, dim=1)
            dists.append(dist)
        dists = torch.stack(dists)

    # use query samples as anchors, corresponding prototypes as positive samples,
    # and choose negative samples from other prototypes
    dist_ap = torch.diag(dists)
    dist_an = []
    for i in range(n_classes):
        dist_i = dists[i]
        dist_an.append(torch.min(torch.cat((dist_i[:i], dist_i[i+1:]))))  # hard mining
    dist_an = torch.stack(dist_an)

    y = dist_an.data.new().resize_as_(dist_an.data).fill_(1)
    if margin is not None:
        loss_val = loss_fn(dist_an, dist_ap, y)
    else:
        loss_val = loss_fn(dist_an - dist_ap, y)
    
    return loss_val
