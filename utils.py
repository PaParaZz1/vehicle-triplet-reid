import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

def show_stats(name, x):
    print('Stats for {} | max {:.5e} | min {:.5e} | mean {:.5e}'.format(name, np.max(x), np.min(x), np.mean(x)))

def available_gpu_num():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)
    # print('in_splits {}'.format(in_splits))

    out_splits = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_splits.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    # print('out_splits {}'.format(out_splits))

    return tf.concat(out_splits, axis=0)
