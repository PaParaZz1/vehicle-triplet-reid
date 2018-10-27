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

def test(opt):
    if not opt.test_set:
        raise BaseException("not specify the 'test_set' argument")
        sys.exit(1)
    if not opt.image_root:
        raise BaseException("not specify the 'image_root' argument")
        sys.exit(1)
    if not torch.cuda.is_available():
        raise BaseException("no available GPU, only support for GPU")
        sys.exit(1)
    try:
        os.makedirs(opt.experiment_root)
    except OSError:
        pass

    test_dataloader, dataset_size = create_dataloader(opt, is_train=False, drop_last=False)
    model = ReIDNetwork(opt.backbone_name, opt.head_name, opt.feature_dim, opt.embedding_dim, opt.pool, opt.branch_number).cuda()
    model = nn.DataParallel(model)
    #model = model.cuda()

    if opt.initial_checkpoint != None:
        from collections import OrderedDict
        old_state_dict = torch.load(opt.initial_checkpoint)
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("load model")
    else:
        raise BaseException("not specify the 'initial_checkpoint' argument")
    model.eval()
    
    batch_size = opt.test_batch_size 
    
    feature_vectors = []
    
    for index, data in enumerate(test_dataloader):
        images = data
        b, n, c, h, w = images.shape
        print('b{}+n{}'.format(b,n))
        images = images.view(-1,c,h,w)
        images = Variable(images).cuda()
        feature, _ = model(images)
        feature = feature.data.cpu().numpy()
        feature_vectors.append(feature)
        print(index)

    feature_vectors = np.asarray(feature_vectors)
    np.save('feature.npy', feature_vectors)
        

    
    

if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate a vehicle ReID nerwork')
    parser.add_argument(
        '--experiment_root', required=True, type=common.writeable_directory,
        help='Location used to store checkpoints and dumped data.')

    parser.add_argument(
        '--test_set',
        help='Path to the test_set csv file.')

    parser.add_argument(
        '--image_root', type=common.readable_directory,
        help='Path that will be pre-pended to the filenames in the train_set csv.')

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
        '--initial_checkpoint', default='save_worker100/checkpoint_800.pth',
        help='Path to the checkpoint file of the pretrained network.')

    parser.add_argument(
        '--test_batch_size', default=64, type=common.positive_int,
        help='The batch size used in test')

    parser.add_argument(
        '--data_augment', default=False, type=bool,
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
        '--num_workers', default=1, type=common.positive_int,
        help='Number of dataloader workers.')

    parser.add_argument(
        '--metric', default='Euclidean', type=str,
        help='Which metric to use for the distance between embeddings.')

    parser.add_argument(
        '--detailed_logs', action='store_true', default=False,
        help='Store very detailed logs of the training in addition to TensorBoard'
             ' summaries. These are mem-mapped numpy files containing the'
             ' embeddings, losses and FIDs seen in each batch during training.'
             ' Everything can be re-constructed and analyzed that way.')

    opt = parser.parse_args()
    test(opt) 
