#!/bin/env python
import csv
import os

dataset = file('VD1_large_query.csv', 'w')
dataset_writer = csv.writer(dataset)
i = 0
with open('./query_ref/large_set.txt', 'r') as f:
    for line in f:
        image_path = os.path.join('./image/', '{}.jpg'.format(line.split(' ')[0]))
        if os.path.exists(image_path):
            dataset_writer.writerow([line.split(' ')[1], image_path])
        else:
            print('Image {} does not exist'.format(image_path))
        i += 1
        print('Processing count: {}'.format(i))

print('Finished create training dataset')
dataset.close()

