#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf

import common
import loss
from RL_utils import PolicyGradient
from utils import show_stats


parser = ArgumentParser(description='Evaluate a ReID embedding.')

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--excluder', required=True, choices=('market1501', 'diagonal'),
    help='Excluder function to mask certain matches. Especially for multi-'
         'camera datasets, one often excludes pictures of the query person from'
         ' the gallery if it is taken from the same camera. The `diagonal`'
         ' excluder should be used if this is *not* required.')

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
        'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--query_dataset', required=True,
    help='Path to the query dataset csv file.')

parser.add_argument(
    '--query_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--gallery_dataset', required=True,
    help='Path to the gallery dataset csv file.')

parser.add_argument(
    '--gallery_embeddings', required=True,
    help='Path to the h5 file containing the query embeddings.')

parser.add_argument(
    '--metric', required=True, choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--filename', type=FileType('w'),
    help='Optional name of the json file to store the results in.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on your memory usage.')

parser.add_argument(
    '--rl_hidden_units', nargs='+', type=int)

parser.add_argument(
    '--rl_activation', default='sigmoid', choices=['sigmoid', 'softmax', 'norm_sigmoid', 'tanh'],
    help='choose activation function for reinforcement learning')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    # Load the query and gallery data from the CSV files.
    query_pids, query_fids = common.load_dataset(args.query_dataset, None)
    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, None)

    # Load the two datasets fully into memory.
    with h5py.File(args.query_embeddings, 'r') as f_query:
        print('query data embs {}'.format(f_query))
        query_ftrs = np.array(f_query['ftr'])
        query_embs = []
        for b_idx in range(5):
            query_embs.append(np.array(f_query['emb_b{}'.format(b_idx)]))
        query_full_embs = np.array(f_query['emb_full'])
        query_raw_embs = np.array(f_query['emb_raw'])

    with h5py.File(args.gallery_embeddings, 'r') as f_gallery:
        gallery_embs = []
        for b_idx in range(5):
            gallery_embs.append(np.array(f_gallery['emb_b{}'.format(b_idx)]))
        gallery_full_embs = np.array(f_gallery['emb_full'])
        gallery_raw_embs = np.array(f_gallery['emb_raw'])

    # Just a quick sanity check that both have the same embedding dimension!
    query_dim = query_embs[0].shape[1]
    gallery_dim = gallery_embs[0].shape[1]
    if query_dim != gallery_dim:
        raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                         'dimension'.format(query_dim, gallery_dim))

    # Setup the dataset specific matching function
    excluder = import_module('excluders.' + args.excluder).Excluder(gallery_fids)

    batch_embs = []
    for b_idx in range(5):
        batch_embs.append(None)

    # We go through the queries in batches, but we always need the whole gallery
    batch_pids, batch_fids, batch_ftrs, batch_full_embs, batch_raw_embs, *batch_embs = tf.data.Dataset.from_tensor_slices(
        (query_pids, query_fids, query_ftrs, query_full_embs, query_raw_embs, *query_embs)
    ).batch(args.batch_size).make_one_shot_iterator().get_next()

    sup_graph = tf.Graph()
    with sup_graph.as_default():
        query_embs_ph = tf.placeholder(dtype=tf.float32, shape=[None, 128], name='query_embs')
        gallery_embs_ph = tf.placeholder(dtype=tf.float32, shape=[None, len(gallery_fids), 128], name='gallery_embs')
        batch_distances = loss.cdist(query_embs_ph, gallery_embs_ph, metric=args.metric)
        instance_distances = loss.inst_dist(query_embs_ph, gallery_embs_ph, metric=args.metric)

    Agent = PolicyGradient(n_actions=5,
                        n_features=1536,
                        is_train=False,
                        rl_activation=args.rl_activation,
                        rl_hidden_units=args.rl_hidden_units)

    rl_graph, rl_init, rl_saver = Agent.train_handle()

    # Loop over the query embeddings and compute their APs and the CMC curve.
    aps = []
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    with tf.Session(config=config) as sess, \
            tf.Session(config=config, graph=rl_graph) as sess_rl, \
            tf.Session(config=config, graph=sup_graph) as sess_sup:
        Agent.get_sess(sess_rl)
        if args.checkpoint is None:
            checkpoint_rl = tf.train.latest_checkpoint(args.experiment_root, latest_filename='checkpoint-rl')
        else:
            checkpoint_rl = os.path.join(args.experiment_root, '{}-rl-{}'.format(*(args.checkpoint.split('-'))))
        rl_saver.restore(sess_rl, checkpoint_rl)

        for start_idx in count(step=args.batch_size):
            try:
                # Compute distance to all gallery embeddings
                _embs = []
                for b_idx in range(5):
                    _embs.append(None)
                pids, fids, ftrs, full_embs, raw_embs, *_embs = sess.run([
                    batch_pids, batch_fids, batch_ftrs, batch_full_embs, batch_raw_embs, *batch_embs])

                full_distances = sess_sup.run(instance_distances, 
                            feed_dict={query_embs_ph:full_embs, 
                                    gallery_embs_ph:np.expand_dims(gallery_full_embs, 0)})

                raw_distances = sess_sup.run(instance_distances, 
                            feed_dict={query_embs_ph:raw_embs, 
                                    gallery_embs_ph:np.expand_dims(gallery_raw_embs, 0)})
                
                norm_ftrs = (ftrs - np.mean(ftrs)) / np.std(ftrs)
                actions = Agent.choose_action(ftrs)
                actions = np.expand_dims(np.transpose(actions, (1, 0)), 2)
                slt_query_embs = np.sum(_embs * actions, axis=0)
                slt_gallery_embs = np.sum(np.expand_dims(gallery_embs, 1) * np.expand_dims(actions, 3), axis=0)
                inst_distances = sess_sup.run(instance_distances, 
                                        feed_dict={query_embs_ph:slt_query_embs, 
                                                gallery_embs_ph:slt_gallery_embs})

                print('\rEvaluating batch {}-{}/{}'.format(
                        start_idx, start_idx + len(fids), len(query_fids)),
                      flush=True, end='')
            except tf.errors.OutOfRangeError:
                print()  # Done!
                break

            # Convert the array of objects back to array of strings
            pids, fids = np.array(pids, '|U'), np.array(fids, '|U')

            # Compute the pid matches
            pid_matches = gallery_pids[None] == pids[:,None]

            # Get a mask indicating True for those gallery entries that should
            # be ignored for whatever reason (same camera, junk, ...) and
            # exclude those in a way that doesn't affect CMC and mAP.

            # distances = full_distances
            distances = raw_distances
            '''
            topk_threshold = 50
            for b_idx in range(len(distances)):
                topk = distances[b_idx].argsort()[:topk_threshold]
                others = distances[b_idx].argsort()[topk_threshold:]
                # distances[b_idx][topk] = distances[b_idx][topk] + inst_distances[b_idx][topk]
                distances[b_idx] = distances[b_idx] + inst_distances[b_idx]
                # distances[b_idx][topk] = inst_distances[b_idx][topk]
                distances[b_idx][others] = np.inf
            '''

            mask = excluder(fids)
            distances[mask] = np.inf
            pid_matches[mask] = False

            # Keep track of statistics. Invert distances to scores using any
            # arbitrary inversion, as long as it's monotonic and well-behaved,
            # it won't change anything.
            scores = 1 / (1 + distances)
            for i in range(len(distances)):
                ap = average_precision_score(pid_matches[i], scores[i])

                if np.isnan(ap):
                    print()
                    print("WARNING: encountered an AP of NaN!")
                    print("This usually means a person only appears once.")
                    print("In this case, it's because of {}.".format(fids[i]))
                    print("I'm excluding this person from eval and carrying on.")
                    print()
                    continue

                aps.append(ap)
                # Find the first true match and increment the cmc data from there on.
                k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
                cmc[k:] += 1

    # Compute the actual cmc and mAP values
    cmc = cmc / len(query_pids)
    mean_ap = np.mean(aps)

    # Save important data
    if args.filename is not None:
        json.dump({'mAP': mean_ap, 'CMC': list(cmc), 'aps': list(aps)}, args.filename)

    # Print out a short summary.
    print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
        mean_ap, cmc[0], cmc[1], cmc[4], cmc[9]))

if __name__ == '__main__':
    main()
