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
            masked_maps = []
            projection_conv = slim.conv2d(endpoints['Mixed_7d'], 512, [1, 1], scope='projection_conv')
            attention_block4_conv1 = slim.conv2d(projection_conv, 64, [1, 1], scope='attention_block4_conv1')
            attention_block4_conv2 = slim.conv2d(attention_block4_conv1, 1, [1, 1], scope='attention_block4_conv2')
            attention_block4_mask = tf.sigmoid(attention_block4_conv2)
            masked_maps.append(projection_conv * attention_block4_mask)

            attention_block3_conv1 = slim.conv2d(endpoints['Mixed_6h'], 64, [1, 1], scope='attention_block3_conv1')
            print('attention_block3_conv1: {}'.format(attention_block3_conv1))
            attention_block3_conv2 = slim.conv2d(attention_block3_conv1, 1, [3, 3], padding='VALID', scope='attention_block3_conv2')
            attention_block3_mask = tf.sigmoid(attention_block3_conv2)
            attention_block3_pool = slim.max_pool2d(attention_block3_mask, [2, 2], stride=[2, 2], scope='attention_block3_pool')
            masked_maps.append(projection_conv * attention_block3_pool)

            attention_block2_conv1 = slim.conv2d(endpoints['Mixed_5e'], 64, [1, 1], scope='attention_block2_conv1')
            attention_block2_conv2 = slim.conv2d(attention_block2_conv1, 1, [3, 3], padding='VALID', scope='attention_block2_conv2')
            attention_block2_mask = tf.sigmoid(attention_block2_conv2)
            attention_block2_pool = slim.max_pool2d(attention_block2_mask, [4, 4], stride=[4, 4], scope='attention_block2_pool')
            masked_maps.append(projection_conv * attention_block2_pool)

            '''
            attention_block1_conv1 = slim.conv2d(endpoints['Mixed_4a'], 64, [1, 1], scope='attention_block1_conv1')
            # attention_block1_pool1 = slim.max_pool2d(attention_block1_conv1, [2, 2], scope='attention_block1_pool1')
            attention_block1_conv2 = slim.conv2d(attention_block1_conv1, 1, [1, 1], padding='VALID', scope='attention_block1_conv2')
            # attention_block1_pool2 = slim.max_pool2d(attention_block1_conv2, [2, 2], scope='attention_block1_pool2')
            attention_block1_mask = tf.sigmoid(attention_block1_conv2)
            attention_block1_pool = slim.max_pool2d(attention_block1_mask, [8, 8], scope='attention_block1_pool')
            masked_maps.append(projection_conv * attention_block1_pool)
            '''

    # endpoints['attention_mask_block1'] = attention_block1_mask
    endpoints['attention_mask_block2'] = attention_block2_mask
    endpoints['attention_mask_block3'] = attention_block3_mask
    endpoints['attention_mask_block4'] = attention_block4_mask
    _masked = tf.concat(masked_maps, 3)

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
