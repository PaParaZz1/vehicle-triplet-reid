import torch
import torch.nn as nn
import numpy as np


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders, net_type='resnet50', normalize_feature=True, gpu=0):
    assert net_type in ['resnet50', 'dense', 'pcb']
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        # print(img.shape)
        n, c, h, w = img.size()
        count += n

        if count % (10 * n) == 0:
            print(count)

        if net_type == 'dense':
            ff = torch.FloatTensor(n, 1024).zero_()
        elif net_type == 'pcb':
            ff = torch.FloatTensor(n, 256, 6).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()

        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = img.cuda(gpu)
            outputs = model(input_img) 
            f = outputs.data.cpu()
            # print(ff.shape)
            # print(f.shape)
            ff = ff + f
        # norm feature
        if net_type == 'pcb':
            if normalize_feature:
                # feature size (n, 2048, 6)
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            if normalize_feature:
                # print(ff.shape)
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    print('total: {:d}'.format(count))
    return features


def get_id(fids):
    camera_id = []
    labels = []
    for fid in fids:
        filename = fid.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0:3]))
    # print(camera_id)
    return camera_id, labels


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    # dist = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=False)
    # score = dist(gf, qf.expand_as(gf)).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i+1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
