import numpy as np
import os

query_fn = './excluders/VeRi-776-query.txt'
jk_fn = './excluders/VeRi-776-jk.txt'

class Excluder(object):
    def __init__(self, gallery_fids):
        # Store the gallery data
        self.gallery_fids = gallery_fids
        self.query_list = []
        self.jk_list = []
        with open(query_fn, 'r') as f:
            for line in f:
                self.query_list.append(os.path.join('./image_query', line.split('\n')[0]))

        with open(jk_fn, 'r') as f:
            for line in f:
                self.jk_list.append(map(int, line.split(' ')[:-1]))

    def __call__(self, query_fids):
        # Only make sure we don't match the exact same image.
        # print(self.query_list)
        mask = np.zeros((len(query_fids), len(self.gallery_fids)), dtype=bool)
        for i in range(len(query_fids)):
            inst = self.query_list.index(query_fids[i])
            for jk_idx in self.jk_list[inst]:
                mask[i, jk_idx-1] = True
        return mask
