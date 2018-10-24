from argparse import ArgumentParser
import os
import sys
import common


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('./data_preprocess')
sys.path.append('./backbone')
from backbone import BACKBONE_CHOICES
from head import HEAD_CHOICES
from data_preprocess.dataloader import create_dataloader
from network import ReIDNetwork
from loss.batch_hard import batch_hard_loss
from loss.MBA_constraint import KLLoss 

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

    train_dataloader = create_dataloader(opt, is_train=True)
    model = ReIDNetwork(opt.backbone_name, opt.head_name, opt.feature_dim, opt.embedding_dim, opt.pool, opt.branch_number).cuda()
    #model = nn.DataParallel(model.cuda())
    model.train()

    if opt.initial_checkpoint != None:
        model.load_state_dict(torch.load(opt.initial_checkpoint))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay_factor)
    lr_decay_milestones = [x for x in range(opt.decay_start_iteration, opt.train_iterations, opt.lr_decay_steps)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_milestones, gamma=opt.lr_decay_factor)

    batch_size = opt.batch_p * opt.batch_k 
    
    for epoch in range(opt.train_iterations):
        lr_scheduler.step()
        for index, data in enumerate(train_dataloader):
            images, labels = data
            _, _, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            labels = labels.view(-1)
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            feature, mask = model(images)

            kl_loss = KLLoss(opt.mba_constraint_weight, size_average=False)
            loss = batch_hard_loss(feature, labels, metric=opt.metric, margin=opt.margin) + kl_loss(mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('[%d: %d batch] train loss:%f'%(epoch, index, loss.data[0]))

        if epoch % opt.checkpoint_frequency == 0:
            torch.save(model.state_dict(), "%s/checkpoint_%d.pth"%(opt.experiment_root, epoch))
    
    

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
        '--num_workers', default=8, type=common.positive_int,
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
        '--lr_decay_factor', default=0.96, type=common.positive_float,
        help='Learning rate decay factor')

    parser.add_argument(
        '--lr_decay_steps', default=4000, type=common.positive_int,
        help='Learning rate decay steps')

    parser.add_argument(
        '--train_iterations', default=25000, type=common.positive_int,
        help='Number of training iterations.')

    parser.add_argument(
        '--decay_start_iteration', default=15000, type=int,
        help='At which iteration the learning-rate decay should kick-in.'
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

    opt = parser.parse_args()
    train(opt) 
