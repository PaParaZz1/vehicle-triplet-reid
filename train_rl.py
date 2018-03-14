#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import timedelta
from importlib import import_module
import logging.config
import os
from signal import SIGINT, SIGTERM
import sys
import time

import json
import numpy as np
# from scipy import stats
import cv2
import copy
import tensorflow as tf
from tensorflow.contrib import slim

import common
import lbtoolbox as lb
import loss
from nets import NET_CHOICES
from heads import HEAD_CHOICES
from RL_utils import TripletStorage, PolicyGradient
from utils import show_stats

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = ArgumentParser(description='Train a ReID network.')



# Required.
parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--train_set',
    help='Path to the train_set csv file.')

parser.add_argument(
    '--image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

# Optional with sane defaults.

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')

parser.add_argument(
    '--model_name', default='resnet_v1_50', choices=NET_CHOICES,
    help='Name of the model to use.')

parser.add_argument(
    '--head_name', default='fc1024', choices=HEAD_CHOICES,
    help='Name of the head to use.')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default=None,
    help='Path to the checkpoint file of the pretrained network.')

# TODO move these defaults to the .sh script?
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
# TODO end

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel loading.')

parser.add_argument(
    '--margin', default='soft', type=common.float_or_string,
    help='What margin to use: a float value for hard-margin, "soft" for '
         'soft-margin, or no margin if "none".')

parser.add_argument(
    '--metric', default='euclidean', choices=loss.cdist.supported_metrics,
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

parser.add_argument(
    '--rl_learning_rate', default=1e-3, type=common.positive_float,
    help='learning rate for traininig reinforcement learning agent')

parser.add_argument(
    '--rl_epsilon', default=1.15, type=common.positive_float,
    help='epsilon greedy for traininig reinforcement learning agent')

parser.add_argument(
    '--rl_reward_decay', default=0.995, type=common.positive_float,
    help='discount param for traininig reinforcement learning agent')

parser.add_argument(
    '--rl_epsilon_decay', default=0.1, type=common.positive_float,
    help='decay param for epsilon in epsilon-greedy for traininig reinforcement learning agent')

parser.add_argument(
    '--rl_activation', default='sigmoid', choices=['sigmoid', 'softmax', 'norm_sigmoid', 'tanh'],
    help='choose activation function in reinforcement learning model')

parser.add_argument(
    '--rl_sample_num', default=10, type=common.positive_int,
    help='number of action samples')

parser.add_argument(
    '--rl_hidden_units', default=256, type=common.positive_int,
    help='number of hidden units in policy networks')

parser.add_argument(
    '--rl_baseline', default='mean-std', choices=['mean', 'mean-std', 'none'],
    help='use different baseline')

parser.add_argument(
    '--rl_decay_start_iteration', default=1000, type=common.positive_int,
    help='learning rate for traininig reinforcement learning agent')

parser.add_argument(
    '--rl_lr_decay_steps', default=2000, type=common.positive_int,
    help='learning rate for traininig reinforcement learning agent')

parser.add_argument(
    '--rl_lr_decay_factor', default=0.9, type=common.positive_float,
    help='learning rate for traininig reinforcement learning agent')


def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original FID count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.ceil(batch_k / count), tf.int32) * count
    full_range = tf.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random_shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

    return selected_fids, tf.fill([batch_k], pid)


def dist(a, b): return np.sqrt(np.sum(np.square(a-b), axis=1) + 1e-12)

def main():
    args = parser.parse_args()

    # We store all arguments in a json file. This has two advantages:
    # 1. We can always get back and see what exactly that experiment was
    # 2. We can resume an experiment as-is without needing to remember all flags.
    args_file = os.path.join(args.experiment_root, 'args.json')
    if args.resume:
        if not os.path.isfile(args_file):
            raise IOError('`args.json` not found in {}'.format(args_file))

        print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)
        args_resumed['resume'] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        # values from the file, but we also want to check for some possible
        # conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value:
                    print('Warning: For the argument `{}` we are using the'
                          ' loaded value `{}`. The provided value was `{}`'
                          '.'.format(key, resumed_value, value))
                    args.__dict__[key] = resumed_value
            else:
                print('Warning: A new argument was added since the last run:'
                      ' `{}`. Using the new value: `{}`.'.format(key, value))

    else:
        # If the experiment directory exists already, we bail in fear.
        if os.path.exists(args.experiment_root):
            if os.listdir(args.experiment_root):
                print('The directory {} already exists and is not empty.'
                      ' If you want to resume training, append --resume to'
                      ' your call.'.format(args.experiment_root))
                exit(1)
        else:
            os.makedirs(args.experiment_root)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)



    log_file = os.path.join(args.experiment_root, "train")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('train')

    # Also show all parameter values at the start, for ease of reading logs.
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))

    # Check them here, so they are not required when --resume-ing.
    if not args.train_set:
        parser.print_help()
        log.error("You did not specify the `train_set` argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        log.error("You did not specify the required `image_root` argument!")
        sys.exit(1)

    # Load the data from the CSV file.
    pids, fids = common.load_dataset(args.train_set, args.image_root)
    max_fid_len = max(map(len, fids))  # We'll need this later for logfiles.

    # Setup a tf.Dataset where one "epoch" loops over all PIDS.
    # PIDS are shuffled after every epoch and continue indefinitely.
    unique_pids = np.unique(pids)
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset = dataset.take((len(unique_pids) // args.batch_p) * args.batch_p)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

    # For every PID, get K images.
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=args.batch_k))

    # Ungroup/flatten the batches for easy loading of the files.
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)
    dataset = dataset.map(
        lambda fid, pid: common.fid_to_image(
            fid, pid, image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)

    # Augment the data if specified by the arguments.
    if args.flip_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.random_crop(im, net_input_size + (3,)), fid, pid))

    # Group it back into PK batches.
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # output_types: (tf.float32, tf.string, tf.string)
    # output_shapes: ((None, 224, 224, 3), (None), (None))

    '''
    # use handle so that we can feed data from difference sources into the model
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
                handle, dataset.output_types, dataset.output_shapes)
    '''

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)
    sup_graph = tf.Graph()
    with sup_graph.as_default():
        # Since we repeat the data infinitely, we only need a one-shot iterator.
        train_iter = dataset.make_one_shot_iterator()

        image_holder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        fid_holder = tf.placeholder(tf.string, shape=[None])
        pid_holder = tf.placeholder(tf.string, shape=[None])
        rl_dataset = tf.data.Dataset.from_tensors((image_holder, fid_holder, pid_holder))
        rl_iter = rl_dataset.make_initializable_iterator()

        # use handle so that we can feed data from difference sources into the model
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
                    handle, dataset.output_types, dataset.output_shapes)
        images, fids, pids = iterator.get_next()
        
        # Feed the image through the model. The returned `body_prefix` will be used
        # further down to load the pre-trained weights for all variables with this
        # prefix.
        endpoints, body_prefix = model.endpoints(images, is_training=True)
        with tf.name_scope('head'):
            endpoints = head.head(endpoints, args.embedding_dim, is_training=True)
        
        # Create the loss in two steps:
        # 1. Compute all pairwise distances according to the specified metric.
        # 2. For each anchor along the first dimension, compute its loss.
        dists = loss.cdist(endpoints['emb'], endpoints['emb'], metric=args.metric)
        losses, train_top1, prec_at_k, _, neg_dists, pos_dists, pos_indices, neg_indices = loss.LOSS_CHOICES[args.loss](
            dists, pids, args.margin, batch_precision_at_k=args.batch_k-1)
        
        # Count the number of active entries, and compute the total batch loss.
        num_active = tf.reduce_sum(tf.cast(tf.greater(losses, 1e-5), tf.float32))
        loss_mean = tf.reduce_mean(losses)

        # These are collected here before we add the optimizer, because depending
        # on the optimizer, it might add extra slots, which are also global
        # variables, with the exact same prefix.
        model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)
        
        # Define a saver for the complete model.
        sup_saver = tf.train.Saver(max_to_keep=0)
        sup_init = tf.global_variables_initializer()
        

    # Some logging for tensorboard.
    tf.summary.histogram('loss_distribution', losses)
    tf.summary.scalar('loss', loss_mean)
    tf.summary.scalar('batch_top1', train_top1)
    tf.summary.scalar('batch_prec_at_{}'.format(args.batch_k-1), prec_at_k)
    tf.summary.scalar('active_count', num_active)
    tf.summary.histogram('embedding_dists', dists)
    tf.summary.histogram('embedding_pos_dists', pos_dists)
    tf.summary.histogram('embedding_neg_dists', neg_dists)
    tf.summary.histogram('embedding_lengths',
                         tf.norm(endpoints['emb_raw'], axis=1))

    # Create the mem-mapped arrays in which we'll log all training detail in
    # addition to tensorboard, because tensorboard is annoying for detailed
    # inspection and actually discards data in histogram summaries.
    if args.detailed_logs:
        log_embs = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'embeddings'),
            dtype=np.float32, shape=(args.train_iterations, batch_size, args.embedding_dim))
        log_loss = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'losses'),
            dtype=np.float32, shape=(args.train_iterations, batch_size))
        log_fids = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'fids'),
            dtype='S' + str(max_fid_len), shape=(args.train_iterations, batch_size))

    # set params for reinforcement learning
    ACTION_NUMS = 1536
    EPSILON = args.rl_epsilon
    # create agent
    Agent = PolicyGradient(
            n_actions=ACTION_NUMS,
            n_features=1536,
            learning_rate=args.rl_learning_rate,
            reward_decay=args.rl_reward_decay,
            is_train=True,
            rl_activation=args.rl_activation,
            rl_hidden_units=args.rl_hidden_units,
            rl_decay_start_iteration=args.rl_decay_start_iteration,
            rl_lr_decay_steps=args.rl_lr_decay_steps,
            rl_lr_decay_factor=args.rl_lr_decay_factor,
            )
    rl_graph, rl_init, rl_saver = Agent.train_handle()

    with tf.Session(config=config, graph=sup_graph) as sess_sup, tf.Session(config=config, graph=rl_graph) as sess_rl:
        Agent.get_sess(sess_rl)
        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.
            # load params for rl model
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root, latest_filename='checkpoint-rl')
            log.info('Restoring RL Model from checkpoint: {}'.format(last_checkpoint))
            rl_saver.restore(sess_rl, last_checkpoint)

            # load params for supervised model
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root, latest_filename='checkpoint-sup')
            log.info('Restoring Supervised Model from checkpoint: {}'.format(last_checkpoint))
            sup_saver.restore(sess_sup, last_checkpoint)
        else:
            # But if we're starting from scratch, we may need to load some
            # variables from the pre-trained weights, and random init others.
            sess_sup.run(sup_init)
            sess_rl.run(rl_init)
            if args.initial_checkpoint is not None:
                saver = tf.train.Saver()
                # saver = tf.train.Saver(model_variables)
                saver.restore(sess_sup, args.initial_checkpoint)

            # In any case, we also store this initialization as a checkpoint,
            # such that we could run exactly reproduceable experiments.
            sup_saver.save(sess_sup, os.path.join(args.experiment_root, 'checkpoint-sup'), latest_filename='checkpoint-sup', global_step=0)
            rl_saver.save(sess_rl, os.path.join(args.experiment_root, 'checkpoint-rl'), latest_filename='checkpoint-rl', global_step=0)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess_sup.graph)
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess_rl.graph)

        start_step = sess_rl.run(Agent.global_step)
        log.info('Starting training from iteration {}.'.format(start_step))

        # initialize train data handle
        train_handle = sess_sup.run(train_iter.string_handle())
        rl_handle = sess_sup.run(rl_iter.string_handle())

        # initialize storage for triplet embeddings and previous triplet loss
        triplet_storage = TripletStorage()

        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            # for i in range(start_step, args.train_iterations):
            for step in range(start_step, args.train_iterations):
                # Just forward to get triplets and images, no updates
                start_time = time.time()
                b_ftrs, b_embs, b_loss, b_fids, b_pids, \
                _pos_dist, _neg_dist, _pos_indices, _neg_indices = \
                    sess_sup.run([endpoints['model_output'], endpoints['emb'], losses, fids, pids, \
                    pos_dists, neg_dists, pos_indices, neg_indices], feed_dict={handle:train_handle})
                pos_ftrs, neg_ftrs = copy.deepcopy(b_ftrs[_pos_indices]), copy.deepcopy(b_ftrs[_neg_indices])

                rl_actions = []
                for sample_idx in range(args.rl_sample_num):
                    # start to do reinforcement learning
                    rl_actions.append(Agent.choose_action(b_ftrs))

                rl_rewards = []
                for sample_idx in range(args.rl_sample_num):    
                    cur_embs = sess_sup.run(endpoints['emb'], feed_dict={endpoints['model_output']:b_ftrs * rl_actions[sample_idx]})
                    pos_embs = sess_sup.run(endpoints['emb'], feed_dict={endpoints['model_output']:pos_ftrs * rl_actions[sample_idx]})
                    neg_embs = sess_sup.run(endpoints['emb'], feed_dict={endpoints['model_output']:neg_ftrs * rl_actions[sample_idx]})
                    cur_loss = np.log(1 + np.exp(dist(cur_embs, pos_embs) - dist(cur_embs, neg_embs)))
                    rl_rewards.append(b_loss - cur_loss)
                
                # normalize reward
                if args.rl_baseline == 'mean':
                    rl_rewards -= np.mean(rl_rewards, axis=0)
                elif args.rl_baseline == 'mean-std':
                    rl_rewards -= np.mean(rl_rewards, axis=0)
                    rl_rewards /= np.std(rl_rewards, axis=0)

                rl_losses = []
                for sample_idx in range(args.rl_sample_num):
                    Agent.store_transition(b_ftrs, rl_actions[sample_idx], rl_rewards[sample_idx])
                    rl_loss, rl_lr = Agent.learn()
                    rl_losses.append(rl_loss)

                # step = sess_rl.run(Agent.global_step)
                elapsed_time = time.time() - start_time
                log.info('RL | Step {} | lr {:.5e} | Action {:.2f} | Reward {: .4e} | Loss {: .4e} | Speed {:.2f}s/iter'.format(step, rl_lr, np.mean(np.count_nonzero(rl_actions, axis=2)), np.mean(rl_rewards), np.mean(rl_losses), elapsed_time))

                # Save a checkpoint of training every so often.
                if (args.checkpoint_frequency > 0 and step % args.checkpoint_frequency == 0):
                    sup_saver.save(sess_sup, os.path.join(args.experiment_root, 'checkpoint-sup'), latest_filename='checkpoint-sup', global_step=step)
                    rl_saver.save(sess_rl, os.path.join(args.experiment_root, 'checkpoint-rl'), latest_filename='checkpoint-rl', global_step=step)

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        sup_saver.save(sess_sup, os.path.join(args.experiment_root, 'checkpoint-sup'), latest_filename='checkpoint-sup', global_step=step)
        rl_saver.save(sess_rl, os.path.join(args.experiment_root, 'checkpoint-rl'), latest_filename='checkpoint-rl', global_step=step)


if __name__ == '__main__':
    main()
