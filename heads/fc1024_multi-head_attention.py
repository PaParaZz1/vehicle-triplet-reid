import tensorflow as tf
from tensorflow.contrib import slim

def head(endpoints, embedding_dim, is_training):
    batch_norm_params = {
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': None,
            }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            attention_projection = slim.conv2d(endpoints['resnet_v2_50/block4'], 512, [1, 1], scope='attention_projection')
            masks = []
            masked_maps = []
            for i in range(4):
                attention_branch_mask = attention_branch(attention_projection, i)
                masks.append(attention_branch_mask)
                endpoints['attention_mask{}'.format(i)] = attention_branch_mask
                masked_map = attention_branch_mask * attention_projection
                masked_maps.append(masked_map)

    _masked = tf.concat(masked_maps, axis=3, name='concated_mask')

    endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
            _masked, [1, 2], name='_pool5', keep_dims=False)

    endpoints['head_output'] = slim.fully_connected(
        endpoints['model_output'], 1024, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        })

    endpoints['emb'] = endpoints['emb_raw'] = slim.fully_connected(
        endpoints['head_output'], embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')

    return endpoints

def attention_branch(_input, name):
    attention_branch_conv1 = slim.conv2d(endpoints['resnet_v2_50/block4'], 64, [1, 1], scope='attention_branch{}_conv1'.format(name))
    attention_branch_conv2 = slim.conv2d(attention_branch_conv1, 1, [1, 1], scope='attention_branch{}_conv2'.format(name))
    attention_branch_mask = tf.sigmoid(attention_branch_conv2, name='attention_branch{}_mask'.format(name))
    return attention_branch_mask
