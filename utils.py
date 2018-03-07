import numpy as np

def show_stats(name, x):
    print('Stats for {} | max {:.5e} | min {:.5e} | mean {:.5e}'.format(name, np.max(x), np.min(x), np.mean(x)))
