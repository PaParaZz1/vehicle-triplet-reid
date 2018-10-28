from argparse import ArgumentParser
import os
import sys
import common
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel

sys.path.append('./data_preprocess')
sys.path.append('./backbone')
from backbone import BACKBONE_CHOICES
from head import HEAD_CHOICES
from data_preprocess.dataloader import create_dataloader
from network import ReIDNetwork
from loss.batch_hard import batch_hard_loss
from loss.MBA_constraint import KLLoss 
from loss.prototypical import PrototypicalLoss
from loss.npair import NpairLoss


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """Decay exponentially in the later phase of training. All parameters in the 
    optimizer share the same learning rate.

    Args:
    optimizer: a pytorch `Optimizer` object
    base_lr: starting learning rate
    ep: current epoch, ep >= 1
    total_ep: total number of epochs to train
    start_decay_at_ep: start decaying at the BEGINNING of this epoch

    Example:
    base_lr = 2e-4
    total_ep = 300
    start_decay_at_ep = 201
    It means the learning rate starts at 2e-4 and begins decaying after 200 
    epochs. And training stops after 300 epochs.

    NOTE: 
    It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.005 ** (float(ep + 1 - start_decay_at_ep)
                                        / (total_ep + 1 - start_decay_at_ep))))
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def train(opt):
    if not opt.train_set:
        raise BaseException("not specify the 'train_set' argument")
        sys.exit(1)
    if not opt.image_root:
        raise BaseException("not specify the 'image_root' argument")
        sys.exit(1)
    if not torch.cuda.is_available():
        raise BaseException("no available GPU, only support for GPU")
        sys.exit(1)

    gids = ''.join(opt.gpu.split())
    device_ids = [int(gid) for gid in gids.split(',')]
    first_device = device_ids[0]

    train_dataloader = create_dataloader(opt, is_train=True)

    is_backbone = False if opt.loss == 'softmax' else True
    model = ReIDNetwork(1000, opt.backbone_name, opt.head_name, opt.feature_dim, opt.embedding_dim,
                        opt.pool, opt.branch_number, is_backbone=is_backbone).cuda(first_device)
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids)

    model.train()

    if opt.initial_checkpoint != None:
        model.load_state_dict(torch.load(opt.initial_checkpoint))
    
    if len(device_ids) == 1:
        if opt.loss == 'softmax':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,
                                        momentum=0.9, weight_decay=opt.weight_decay_factor)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate,
                                         weight_decay=opt.weight_decay_factor)
    else:
        if opt.loss == 'softmax':
            optimizer = torch.optim.SGD(model.module.parameters(), lr=opt.learning_rate,
                                        momentum=0.9, weight_decay=opt.weight_decay_factor)
        else:
            optimizer = torch.optim.Adam(model.module.parameters(), lr=opt.learning_rate,
                                         weight_decay=opt.weight_decay_factor)

    if opt.lr_policy == 'step':
        lr_decay_milestones = [x for x in range(
            opt.decay_start_epoch, opt.train_epochs, opt.lr_decay_steps)]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_decay_milestones, gamma=opt.lr_decay_factor)
    elif opt.lr_policy == 'exp':
        lr_scheduler = None

    # batch_size = opt.batch_p * opt.batch_k 

    if 'kl' in opt.loss:
        kl_loss = KLLoss(opt.mba_constraint_weight, size_average=True)
    if 'prototypical' in opt.loss:
        proto_loss = PrototypicalLoss(n_support=opt.num_support, mode=opt.distance_mode, margin=opt.margin)
    if opt.loss == 'softmax':
        softmax_loss = nn.CrossEntropyLoss()
    elif opt.loss == 'n_pair':
        npair_loss = NpairLoss(l2_reg=0.002, gpu=first_device)
    
    train_loss = []
    t0 = int(time.time())
    for epoch in range(opt.train_epochs):
        if opt.lr_policy == 'step':
            lr_scheduler.step()
        elif opt.lr_policy == 'exp':
            adjust_lr_exp(optimizer, opt.learning_rate, epoch+1, opt.train_epochs, opt.decay_start_epoch)
        for index, data in enumerate(train_dataloader):
            images, labels = data
            _, _, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            labels = labels.view(-1)
            images, labels = images.cuda(first_device), labels.cuda(first_device)

            if 'kl' in opt.loss:
                feature, mask = model(images)
            else:
                output = model(images)
                # print(output.shape)

            if opt.loss == 'softmax':
                loss = softmax_loss(output, labels)
            elif opt.loss == 'batch_hard':
                loss = batch_hard_loss(output, labels, metric=opt.metric, margin=opt.margin)
            elif opt.loss == 'batch_hard+kl':
                loss = batch_hard_loss(output, labels, metric=opt.metric, margin=opt.margin) + kl_loss(mask)
            elif opt.loss == 'n_pair':
                loss = npair_loss(output, labels)
            elif opt.loss == 'prototypical':
                loss = proto_loss(output, labels).cuda(first_device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('[%d: %d batch] train loss:%f'%(epoch, index, loss.item()))
            train_loss.append(loss.item())

        t = int(time.time())
        print('average train loss: {:.6f}'.format(np.mean(train_loss)))
        train_loss = []
        print('Time elapsed: {} h {} m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))
        if epoch >= opt.train_epochs // 2 and epoch % opt.checkpoint_frequency == 0:
            torch.save(model.state_dict(), "%s/checkpoint_%d.pth"%(opt.experiment_root, epoch))
            print('Model {} saved.'.format(epoch))
    torch.save(model.state_dict(), "{}/checkpoint_{}.pth".format(opt.experiment_root, 'last'))
    print('Last model saved.')
    

if __name__ == '__main__':
    parser = ArgumentParser(description='train a vehicle ReID nerwork')
    parser.add_argument(
        '--experiment_root', required=True, type=common.writeable_directory,
        help='Location used to store checkpoints and dumped data.')

    parser.add_argument(
        '--train_set',
        help='Path to the train_set csv file.')

    parser.add_argument(
        '--image_root', type=common.readable_directory,
        help='Path that will be pre-pended to the filenames in the train_set csv.')

    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='When this flag is provided, all other arguments apart from the '
             'experiment_root are ignored and a previously saved set of arguments '
             'is loaded.')

    parser.add_argument(
        '--backbone_name', default='resnet50', choices=BACKBONE_CHOICES,
        help='Name of the backbone to use.')

    parser.add_argument(
        '--head_name', default='MultiBranchAttention', choices=HEAD_CHOICES,
        help='Name of the head to use.')

    parser.add_argument(
        '--feature_dim', default=2048, type=common.positive_int,
        help='Dimensionality of the backbone output feature.This dim must match backbone')

    parser.add_argument(
        '--embedding_dim', default=128, type=common.positive_int,
        help='Dimensionality of the embedding space.')

    parser.add_argument(
        '--pool', default='concat', type=str,
        help='pool type for multi-branch-attention')

    parser.add_argument(
        '--branch_number', default=5, type=common.positive_int,
        help='multi attention branch numbers')

    parser.add_argument(
        '--initial_checkpoint', default=None,
        help='Path to the checkpoint file of the pretrained network.')

    parser.add_argument(
        '--batch_p', default=32, type=common.positive_int,
        help='The number P used in the PK-batches')

    parser.add_argument(
        '--batch_k', default=4, type=common.positive_int,
        help='The number K used in the PK-batches')

    parser.add_argument(
        '--data_augment', default=True, type=bool,
        help='whether use input data augmentation')

    parser.add_argument(
        '--resize_height', default=288, type=common.positive_int,
        help='Height used to resize a loaded image. This is ignored when no data '
             'augmentation is applied.')

    parser.add_argument(
        '--resize_width', default=144, type=common.positive_int,
        help='Width used to resize a loaded image. This is ignored when no data '
             'augmentation is applied.')

    parser.add_argument(
        '--num_workers', default=8, type=int,
        help='Number of dataloader workers.')

    parser.add_argument(
        '--margin', default='soft', type=common.float_or_string,
        help='What margin to use: a float value for hard-margin, "soft" for '
             'soft-margin, or no margin if "none".')

    parser.add_argument(
        '--metric', default='Euclidean', type=str,
        help='Which metric to use for the distance between embeddings.')

    parser.add_argument(
        '--loss', default='batch_hard', type=str,
        help='Enable the super-mega-advanced top-secret sampling stabilizer.')

    parser.add_argument(
        '--learning_rate', default=3e-4, type=common.positive_float,
        help='The initial value of the learning-rate, before it kicks in.')

    parser.add_argument(
        '--lr_policy', default='step', type=str,
        help='Learning rate decay policy during training')

    parser.add_argument(
        '--lr_decay_factor', default=0.96, type=common.positive_float,
        help='Learning rate decay factor')

    parser.add_argument(
        '--lr_decay_steps', default=4000, type=common.positive_int,
        help='Learning rate decay steps')

    parser.add_argument(
        '--train_epochs', default=600, type=common.positive_int,
        help='Number of training epochs.')

    parser.add_argument(
        '--decay_start_epoch', default=300, type=int,
        help='At which epoch the learning-rate decay should kick-in.'
             'Set to -1 to disable decay completely.')

    parser.add_argument(
        '--weight_decay_factor', default=0.001, type=common.positive_float,
        help='Weight decay factor')

    parser.add_argument(
        '--mba_constraint_weight', default=0.1, type=float,
        help='mba constraint weight')

    parser.add_argument(
        '--checkpoint_frequency', default=1000, type=common.nonnegative_int,
        help='After how many iterations a checkpoint is stored. Set this to 0 to '
             'disable intermediate storing. This will result in only one final '
             'checkpoint.')

    parser.add_argument(
        '--detailed_logs', action='store_true', default=False,
        help='Store very detailed logs of the training in addition to TensorBoard'
             ' summaries. These are mem-mapped numpy files containing the'
             ' embeddings, losses and FIDs seen in each batch during training.'
             ' Everything can be re-constructed and analyzed that way.')

    parser.add_argument(
        '--num_support', default=3, type=int,
        help='number of support samples in every episode')

    parser.add_argument(
        '--num_query', default=1, type=int,
        help='number of query samples in every episode')

    parser.add_argument(
        '--distance_mode', default='maxmin', type=str,
        help='distance measurement method of prototypical loss')

    parser.add_argument(
        '--gpu', default='0', type=str,
        help='gpu device ids used in training')

    opt = parser.parse_args()
    train(opt) 
