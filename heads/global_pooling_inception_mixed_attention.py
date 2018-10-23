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
            attention_branch_projection = slim.conv2d(endpoints['Mixed_7d'], 256, [1, 1], scope='attention_branch_projection')
            attention_branch_conv1 = slim.conv2d(attention_branch_projection, 64, [1, 1], scope='attention_branch_conv1')
            attention_branch_conv2 = slim.conv2d(attention_branch_conv1, 64, [3, 3], scope='attention_branch_conv2')
            attention_branch_conv3 = slim.conv2d(attention_branch_conv2, 256, [1, 1], scope='attention_branch_conv3')
            attention_branch_residual = slim.conv2d(attention_branch_conv3 + attention_branch_projection, 1536, [1, 1], scope='attention_branch_residual')
            attention_branch_mask = tf.sigmoid(attention_branch_residual)
            endpoints['attention_mask'] = attention_branch_mask
            _masked = (1 + attention_branch_mask) * endpoints['Mixed_7d']
            embs = slim.conv2d(_masked, embedding_dim, [1, 1], scope='embs')

    endpoints['emb'] = endpoints['emb_raw'] = endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
            _masked, [1, 2], name='_pool5', keep_dims=False)

    return endpoints
