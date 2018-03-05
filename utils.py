import numpy as np

def show_stats(name, x):
    print('Stats for {} | max {} | min {} | mean {}'.format(name, np.max(x), np.min(x), np.mean(x)))
