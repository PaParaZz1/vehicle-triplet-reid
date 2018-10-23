import tensorflow as tf

# from nets.inception_v4 import inception_v4, inception_arg_scope
from nets.vgg import vgg_16
from nets.vgg import vgg_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

    with tf.contrib.slim.arg_scope(vgg_arg_scope(weight_decay=0.0002)):
        _, endpoints = vgg_16(image, num_classes=None, is_training=is_training, global_pool=True)
    print('endpoints of vgg16 {}'.format(endpoints))

    # endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
    #     endpoints['vgg_16/pool5'], [1, 2], name='pool5', keep_dims=False)
    endpoints['model_output'] = endpoints['global_pool']

    return endpoints, 'vgg_16'
