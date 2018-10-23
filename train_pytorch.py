from argparse import ArgumentParser
import common


import numpy as np
import torch
from torch.autograd import Variable

from backbone import BACKBONE_CHOICES
from head import HEAD_CHOICES

def train(opt):

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
        '--backbone_name', default='resnet_50', choices=BACKBONE_CHOICES,
        help='Name of the backbone to use.')

    parser.add_argument(
        '--head_name', default='MultiAttentionBranch', choices=HEAD_CHOICES,
        help='Name of the head to use.')

    parser.add_argument(
        '--feature_dim', default=512, type=common.positive_int,
        help='Dimensionality of the backbone output feature.')

    parser.add_argument(
        '--embedding_dim', default=128, type=common.positive_int,
        help='Dimensionality of the embedding space.')

    parser.add_argument(
        '--initial_checkpoint', default=None,
        help='Path to the checkpoint file of the pretrained network.')

    parser.add_argument(
        '--batch_p', default=32, type=common.positive_int,
        help='The number P used in the PK-batches')

    parser.add_argument(
        '--batch_k', default=4, type=common.positive_int,
        help='The numberK used in the PK-batches')

    parser.add_argument(
        '--net_input_height', default=256, type=common.positive_int,
        help='Height of the input directly fed into the network.')

    parser.add_argument(
        '--net_input_width', default=128, type=common.positive_int,
        help='Width of the input directly fed into the network.')

    parser.add_argument(
        '--pre_crop_height', default=288, type=common.positive_int,
        help='Height used to resize a loaded image. This is ignored when no crop '
             'augmentation is applied.')

    parser.add_argument(
        '--pre_crop_width', default=144, type=common.positive_int,
        help='Width used to resize a loaded image. This is ignored when no crop '
             'augmentation is applied.')

    parser.add_argument(
        '--num_workers', default=8, type=common.positive_int,
        help='Number of dataloader workers.')

    parser.add_argument(
        '--margin', default='soft', type=common.float_or_string,
        help='What margin to use: a float value for hard-margin, "soft" for '
             'soft-margin, or no margin if "none".')

    parser.add_argument(
        '--metric', default='Euclidean', choices=loss.cdist.supported_metrics,
        help='Which metric to use for the distance between embeddings.')

    parser.add_argument(
        '--loss', default='batch_hard', choices=loss.LOSS_CHOICES.keys(),
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
        '--checkpoint_frequency', default=1000, type=common.nonnegative_int,
        help='After how many iterations a checkpoint is stored. Set this to 0 to '
             'disable intermediate storing. This will result in only one final '
             'checkpoint.')

    parser.add_argument(
        '--flip_augment', action='store_true', default=False,
        help='When this flag is provided, flip augmentation is performed.')

    parser.add_argument(
        '--crop_augment', action='store_true', default=False,
        help='When this flag is provided, crop augmentation is performed. Based on'
             'The `crop_height` and `crop_width` parameters. Changing this flag '
             'thus likely changes the network input size!')

    parser.add_argument(
        '--detailed_logs', action='store_true', default=False,
        help='Store very detailed logs of the training in addition to TensorBoard'
             ' summaries. These are mem-mapped numpy files containing the'
             ' embeddings, losses and FIDs seen in each batch during training.'
             ' Everything can be re-constructed and analyzed that way.')

    opt = parser.parse_args()
    train(opt) 
