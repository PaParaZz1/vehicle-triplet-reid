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
            [slim.conv2d, slim.max_pool2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(0.0),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            projection_conv = slim.conv2d(endpoints['resnet_v2_50/block4'], 256, [1, 1], scope='projection_conv')
            preprocess_residual_block = residual_block(projection_conv)
            trunk_branch_block1 = residual_block(preprocess_residual_block)
            trunk_branch_block2 = residual_block(trunk_branch_block1)
            mask_branch_block1 = residual_block(preprocess_residual_block)
            mask_branch_block2 = residual_block(mask_branch_block1)
            mask_branch_conv = slim.conv2d(mask_branch_block2, 256, [1, 1], scope='mask_branch_conv')
            mask_branch_prob = tf.sigmoid(mask_branch_conv)
            _masked = (1 + mask_branch_prob) * trunk_branch_block2

    endpoints['attention_mask'] = mask_branch_prob

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

def residual_block(input_features):
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
            residual_conv1 = slim.conv2d(input_features, 64, [1, 1])
            residual_conv2 = slim.conv2d(residual_conv1, 64, [3, 3])
            residual_conv3 = slim.conv2d(residual_conv2, 256, [1, 1])
            output = residual_conv3 + input_features
    return output
