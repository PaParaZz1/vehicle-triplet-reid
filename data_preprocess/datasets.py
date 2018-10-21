#-*- coding=utf-8-*-

import os.path
import torch.utils.data
import torch
import numpy as np
import random
import math

class PartDataset(torch.utils.data.Dataset):
    def __init__(self, root, catfile='cat.txt', npoints=4096, train=True, classification=False):
        self.npoints = npoints
        self.root = root
        self.category = {}
        self.classification = classification

        #生成所有类别及其分类编号
        with open(os.path.join(root, 'data/BDCI/training', catfile)) as f_catfile:
            lines = f_catfile.readlines()
            self.category = dict(zip(lines, range(len(lines))))
        #建立路径
        train_files = os.path.join(root, 'data/BDCI/new_train')
        dev_files = os.path.join(root, 'data/BDCI/new_dev')

        #构建数据集
        if train:
            #数据集为train所用的sample
            self.datapath = sorted(os.listdir(train_files))
            self.datapath = [os.path.join(train_files, i) for i in self.datapath]
        else:
            #数据集为dev所用的sample
            self.datapath = sorted(os.listdir(dev_files))
            self.datapath = [os.path.join(dev_files, i) for i in self.datapath]

    def __getitem__(self, index):
        fn = self.datapath[index]
        #载入sample文件内容
        file_set = np.load(fn).astype(np.float32)
        #转为torch的Tensor类型
        point_set = torch.from_numpy(file_set[:,1:])
        seg_set = torch.from_numpy(file_set[:,0])
        #返回数据对
        return point_set, seg_set

    def __len__(self):
        return len(self.datapath)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, catfile='cat.txt', npoints=4096, train=False, classification=False):
        self.npoints = npoints
        self.root = root
        self.category = {}
        self.classification = classification

        #生成所有类别及其分类编号
        with open(os.path.join(root, 'data/BDCI/training', catfile)) as f_catfile:
            lines = f_catfile.readlines()
            self.category = dict(zip(lines, range(len(lines))))
        #建立路径
        test_point = os.path.join(self.root, 'data/BDCI/test/pts')

        #构建数据集
        if train:
            #数据集为0.8
            point_files = sorted(os.listdir(test_point)[:int(0.8*len(test_point))])
            self.datapath = [os.path.join(test_point, point_files[i]) for i in range(len(point_files))]
        else:
            #数据集为all
            point_files = sorted(os.listdir(test_point))
            self.datapath = [os.path.join(test_point, point_files[i]) for i in range(len(point_files))]

    def __getitem__(self, index):
        fn = self.datapath[index]
        #载入sample文件内容
        point_set = list(np.loadtxt(fn, delimiter=',').astype(np.float32).tolist())
        #归一化
        means = list(np.mean(point_set, axis=0).tolist())
        fangcha = list(np.var(point_set, axis=0).tolist())
        fangcha = list(map(math.sqrt, fangcha))
        for i in range(len(point_set)):
            point_set[i][0] = (point_set[i][0] - means[0]) / fangcha[0]
            point_set[i][1] = (point_set[i][1] - means[1]) / fangcha[1]
            point_set[i][2] = (point_set[i][2] - means[2]) / fangcha[2]
        point_set = np.array(point_set)
        point_set = torch.from_numpy(point_set)
        #返回坐标，文件名
        return point_set, os.path.basename(fn)

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    print('test')
    d = PartDataset(root='/mnt/lustre/niuyazhe')
    print(len(d))
    ps, seg = d[0]
    print(ps[0], seg[0])
    print(ps.size(), ps.type(), seg.size(), seg.type())
    d = TestDataset(root='/mnt/lustre/niuyazhe')
    print(len(d))
    ps, name = d[0]
    print(name)
    print(ps.size(), ps.type())
