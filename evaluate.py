#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from importlib import import_module
from itertools import count
import os, shutil

import h5py
import json
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf

import common
import loss


parser = ArgumentParser(description='Evaluate a ReID embedding.')

parser.add_argument(
    '--excluder', required=True, choices=('market1501', 'diagonal'),
    help='Excluder function to mask certain matches. Especially for multi-'
         'camera datasets, one often excludes pictures of the query person from'
         ' the gallery if it is taken from the same camera. The `diagonal`'
         ' excluder should be used if this is *not* required.')

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
    '--display', action='store_true', default=False)

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
        query_embs = np.array(f_query['emb'])
    with h5py.File(args.gallery_embeddings, 'r') as f_gallery:
        gallery_embs = np.array(f_gallery['emb'])

    print('gallery shape: {}'.format(gallery_embs.shape))
    # Just a quick sanity check that both have the same embedding dimension!
    query_dim = query_embs.shape[1]
    gallery_dim = gallery_embs.shape[1]
    gallery_amount = gallery_embs.shape[0]
    if query_dim != gallery_dim:
        raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                         'dimension'.format(query_dim, gallery_dim))

    # Setup the dataset specific matching function
    excluder = import_module('excluders.' + args.excluder).Excluder(gallery_fids)

    # We go through the queries in batches, but we always need the whole gallery
    batch_pids, batch_fids, batch_embs = tf.data.Dataset.from_tensor_slices(
        (query_pids, query_fids, query_embs)
    ).batch(args.batch_size).make_one_shot_iterator().get_next()

    gallery_iter = tf.data.Dataset.from_tensor_slices(gallery_embs).batch(gallery_amount).make_initializable_iterator()
    gallery_embs = gallery_iter.get_next()

    batch_distances = loss.cdist(batch_embs, gallery_embs, metric=args.metric)
    print('batch distance: {}'.format(batch_distances))

    # Loop over the query embeddings and compute their APs and the CMC curve.
    aps = []
    cmc = np.zeros(len(gallery_pids), dtype=np.int32)
    with tf.Session(config=config) as sess:
        for start_idx in count(step=args.batch_size):
            try:
                sess.run(gallery_iter.initializer)
                # Compute distance to all gallery embeddings
                distances, pids, fids = sess.run([
                    batch_distances, batch_pids, batch_fids])
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
            # print('pid match matrix shape: {}'.format(pid_matches.shape))
            # print('pid_matchs[0]: {}'.format(pid_matches[0]))
            # print('fids: {}'.format(fids))

            # Get a mask indicating True for those gallery entries that should
            # be ignored for whatever reason (same camera, junk, ...) and
            # exclude those in a way that doesn't affect CMC and mAP.
            mask = excluder(fids)
            distances[mask] = np.inf
            pid_matches[mask] = False
            # print('distance matrix: {}'.format(distances.shape))

            # Keep track of statistics. Invert distances to scores using any
            # arbitrary inversion, as long as it's monotonic and well-behaved,
            # it won't change anything.
            scores = 1 / (1 + distances)
            for i in range(len(distances)):
                # for each query instance
                ap = average_precision_score(pid_matches[i], scores[i])

                if np.isnan(ap):
                    print("\nWARNING: encountered an AP of NaN!")
                    print("This usually means a person only appears once.")
                    print("In this case, it's because of {}.".format(fids[i]))
                    print("I'm excluding this person from eval and carrying on.\n")
                    continue

                aps.append(ap)
                # Find the first true match and increment the cmc data from there on.
                # print('match shape: {}'.format(pid_matches[i, np.argsort(distances[i])].shape))
                k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
                if args.display:
                    mismatched_idx = np.argsort(distances[i])[:k]
                    if k > 5:
                        print('Mismatch | fid: {}'.format(fids[i]))
                        os.system('cp {} {}'.format(os.path.join('/data2/wangq/VD1/', fids[i]), os.path.join('./mismatched-5', 'batch-{}_query-{}_{}'.format(start_idx / args.batch_size, i, fids[i].split('/')[-1]))))
                        for l in range(k):
                            os.system('cp {} {}'.format(os.path.join('/data2/wangq/VD1/', gallery_fids[mismatched_idx[l]]), os.path.join('./mismatched-5', 'batch-{}_query-{}_gallary-{}_{}'.format(start_idx / args.batch_size, i, l, gallery_fids[mismatched_idx[l]].split('/')[-1]))))
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
