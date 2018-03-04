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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = ArgumentParser(description='Train a ReID network.')

# set params for reinforcement learning
MAX_PLAY_STEP = 25
PATCH_W, PATCH_H = 16, 16
# PATCH_W, PATCH_H = 32, 32
PATCH_STRIDE = 8 
# total number of actions is [(224-8)/8]^2 + 1 = 730
# in spite of termial action, each action indicates the index of top-left corner of patch
ACTION_NUMS = 730
EPSILON = 0.1


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


def dist(a, b): return np.sqrt(np.sum(np.square(a-b)) + 1e-12)


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

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    train_iter = dataset.make_one_shot_iterator()
    # images, fids, pids = train_iter.get_next()

    # output_types: (tf.float32, tf.string, tf.string)
    # output_shapes: ((None, 224, 224, 3), (None), (None))

    # use handle so that we can feed data from difference sources into the model
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
                handle, dataset.output_types, dataset.output_shapes)
    images, fids, pids = iterator.get_next()

    image_holder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    fid_holder = tf.placeholder(tf.string, shape=[None])
    pid_holder = tf.placeholder(tf.string, shape=[None])
    # rl_dataset = tf.data.Dataset.from_tensor_slices((image_holder, fid_holder, pid_holder)).batch(1).prefetch(1)
    # rl_dataset = tf.data.Dataset.from_tensor_slices((image_holder, fid_holder, pid_holder))
    rl_dataset = tf.data.Dataset.from_tensors((image_holder, fid_holder, pid_holder))
    rl_iter = rl_dataset.make_initializable_iterator()

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

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

    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

    # Define the optimizer and the learning-rate schedule.
    # Unfortunately, we get NaNs if we don't handle no-decay separately.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if 0 <= args.decay_start_iteration < args.train_iterations:
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            tf.maximum(0, global_step - args.decay_start_iteration),
            # args.train_iterations - args.decay_start_iteration, args.weight_decay_factor)
            args.lr_decay_steps, args.lr_decay_factor, staircase=True)
    else:
        learning_rate = args.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Feel free to try others!
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    # Define a saver for the complete model.
    checkpoint_saver = tf.train.Saver(max_to_keep=0)

    with tf.Session(config=config) as sess:
        # create agent
        Agent = PolicyGradient(
                n_actions=ACTION_NUMS,
                n_features=args.embedding_dim,
                sess=sess,
                learning_rate=0.01,
                reward_decay=0.995,
                )

        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root)
            log.info('Restoring from checkpoint: {}'.format(last_checkpoint))
            checkpoint_saver.restore(sess, last_checkpoint)
        else:
            # But if we're starting from scratch, we may need to load some
            # variables from the pre-trained weights, and random init others.
            sess.run(tf.global_variables_initializer())
            if args.initial_checkpoint is not None:
                saver = tf.train.Saver(model_variables)
                saver.restore(sess, args.initial_checkpoint)

            # In any case, we also store this initialization as a checkpoint,
            # such that we could run exactly reproduceable experiments.
            checkpoint_saver.save(sess, os.path.join(
                args.experiment_root, 'checkpoint'), global_step=0)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess.graph)

        start_step = sess.run(global_step)
        log.info('Starting training from iteration {}.'.format(start_step))

        # initialize train data handle
        train_handle = sess.run(train_iter.string_handle())
        rl_handle = sess.run(rl_iter.string_handle())

        # initialize storage for triplet embeddings and previous triplet loss
        triplet_storage = TripletStorage()

        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(start_step, args.train_iterations):
                # Just forward to get triplets and images, no updates
                start_time = time.time()
                b_imgs, b_embs, b_loss, b_fids, b_pids, \
                _pos_dist, _neg_dist, _pos_indices, _neg_indices = \
                    sess.run([images, endpoints['emb'], losses, fids, pids, \
                    pos_dists, neg_dists, pos_indices, neg_indices], feed_dict={handle:train_handle})
                triplet_storage.add_storage(copy.deepcopy(b_embs), copy.deepcopy(b_embs[_pos_indices]), copy.deepcopy(b_embs[_neg_indices]), copy.deepcopy(b_loss))
                print('anchor {} | pos {}'.format(b_embs.shape, b_embs[_pos_indices].shape))

                # start to do reinforcement learning
                sorted_agent = np.random.choice(range(args.batch_p * args.batch_k), 5)
                for agent_idx in sorted_agent:
                    cur_embeddings = triplet_storage.get_anchor_embs(agent_idx)
                    cur_img = b_imgs[agent_idx]
                    for rlstep in range(MAX_PLAY_STEP):
                        # print('embeddings | max {} | min {} | mean {}'.format(np.max(b_embs[agent_idx]), np.min(b_embs[agent_idx]), np.mean(b_embs[agent_idx])))
                        # print('embeddings | max {} | min {} | mean {}'.format(np.max(cur_embeddings), np.min(cur_embeddings), np.mean(cur_embeddings)))
                        # print('images | max {} | min {} | mean {}'.format(np.max(b_imgs[agent_idx]), np.min(b_imgs[agent_idx]), np.mean(b_imgs[agent_idx])))
                        # cv2.imwrite(os.path.join('att_img', '{}_agent-{}_step-{}.jpg'.format(i, agent_idx, rlstep)), b_imgs[agent_idx].astype(np.uint8))
                        # action = Agent.choose_action(b_embs[agent_idx])
                        action = Agent.choose_action(cur_embeddings)
                        # if np.random.random() < EPSILON:
                        #     action = np.random.randint(0, ACTION_NUMS)
                        if action == ACTION_NUMS - 1:
                            break
                        x1, y1 = np.array(np.unravel_index(action, (27, 27))) * PATCH_STRIDE
                        # x2, y2 = np.maximum((223, 223), np.array([x1, y1], dtype=np.int) + 16)
                        x2, y2 = np.minimum((223, 223), (x1 + PATCH_W, y1 + PATCH_H))
                        # b_imgs[agent_idx][x1:x2, y1:y2] = 1e-5
                        cur_img[x1:x2, y1:y2] = 1e-5

                        print('images | max {} | min {} | mean {} | shape {}'.format(
                                                np.max(cur_img), np.min(cur_img), 
                                                np.mean(cur_img), np.array(cur_img).shape))
 
                        sess.run(rl_iter.initializer, feed_dict={
                                                image_holder:np.expand_dims(cur_img, 0),
                                                # image_holder:np.expand_dims(b_imgs[agent_idx], 0),
                                                fid_holder:np.expand_dims(b_fids[agent_idx], 0),
                                                pid_holder:np.expand_dims(b_pids[agent_idx], 0)})
                    
                        # b_imgs[agent_idx], b_embs[agent_idx] = sess.run([images, endpoints['emb']], 
                        # b_imgs[agent_idx], cur_embeddings = sess.run([images, endpoints['emb']], 
                        # cur_img, cur_embeddings = sess.run([images, endpoints['emb']], 
                        cur_embeddings = sess.run(endpoints['emb'], feed_dict={handle:rl_handle})
                        cur_embeddings = cur_embeddings[0]
                        # cur_img, cur_embeddings = cur_img[0], cur_embeddings[0]
                        prev_loss = triplet_storage.get_loss(agent_idx)
                        # cur_loss = dist(b_embs[agent_idx], triplet_storage.get_neg_embs(agent_idx)) - dist(b_embs[agent_idx], triplet_storage.get_pos_embs(agent_idx))
                        cur_loss = dist(cur_embeddings, triplet_storage.get_neg_embs(agent_idx)) \
                                - dist(cur_embeddings, triplet_storage.get_pos_embs(agent_idx))
                        reward = cur_loss - prev_loss
                        triplet_storage.update_loss(cur_loss, agent_idx)
                        # Agent.store_transition(b_embs[agent_idx], action, reward)
                        Agent.store_transition(cur_embeddings, action, reward)
                        triplet_storage.update_anchor_embs(cur_embeddings, agent_idx)
                        log.info('RL | Agent {} | Step {} | Action {} | Reward {}'.format(agent_idx, rlstep, action, reward))
                        # if abs(reward) < 1e-6:
                        #     break
                    _dis_reward, _rl_loss = Agent.learn()
                    # log.info('RL | Discounted Reward {} | Loss {}'.format(_dis_reward, _rl_loss))
                triplet_storage.clear_storage()

                # Supervised Learning
                supervise_iter = tf.data.Dataset.from_tensor_slices((b_imgs, b_fids, b_pids)).batch(args.batch_p * args.batch_k).prefetch(args.batch_p * args.batch_k).make_one_shot_iterator()
                supervise_handle = sess.run(supervise_iter.string_handle())
                _, summary, step, b_prec_at_k, b_embs, b_loss, b_fids, \
                _pos_dist, _neg_dist, _pos_indices, _neg_indices = \
                    sess.run([train_op, merged_summary, global_step, \
                    prec_at_k, endpoints['emb'], losses, fids, \
                    pos_dists, neg_dists, pos_indices, neg_indices], feed_dict={handle:supervise_handle})
                elapsed_time = time.time() - start_time

                # Compute the iteration speed and add it to the summary.
                # We did observe some weird spikes that we couldn't track down.
                summary2 = tf.Summary()
                summary2.value.add(tag='secs_per_iter', simple_value=elapsed_time)
                summary_writer.add_summary(summary2, step)
                summary_writer.add_summary(summary, step)

                if args.detailed_logs:
                    log_embs[i], log_loss[i], log_fids[i] = b_embs, b_loss, b_fids

                # Do a huge print out of the current progress.
                seconds_todo = (args.train_iterations - step) * elapsed_time
                log.info('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
                         'batch-p@{}: {:.2%}, ETA: {} ({:.2f}s/it)'.format(
                             step,
                             float(np.min(b_loss)),
                             float(np.mean(b_loss)),
                             float(np.max(b_loss)),
                             args.batch_k-1, float(b_prec_at_k),
                             timedelta(seconds=int(seconds_todo)),
                             elapsed_time))
                sys.stdout.flush()
                sys.stderr.flush()

                # Save a checkpoint of training every so often.
                if (args.checkpoint_frequency > 0 and
                        step % args.checkpoint_frequency == 0):
                    checkpoint_saver.save(sess, os.path.join(
                        args.experiment_root, 'checkpoint'), global_step=step)

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        checkpoint_saver.save(sess, os.path.join(
            args.experiment_root, 'checkpoint'), global_step=step)


if __name__ == '__main__':
    main()
