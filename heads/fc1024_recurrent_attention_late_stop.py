import tensorflow as tf
from tensorflow.contrib import slim

def head(endpoints, embedding_dim, is_training):
    
    M = 7
    L = M*M
    D = 64
    dim = [L, D]
    attention_steps = 16

    # endpoints['resnet_v2_50/block4'] is in shape of (?, 7, 7, 2048)
    batch_norm_params = {
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': None,
            }
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(0.0),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            attention_branch_conv = slim.conv2d(endpoints['resnet_v2_50/block4'], dim[1], [1, 1], scope='attention_branch_conv')
            # create a BasicRNNCell
            features = tf.reshape(attention_branch_conv, [-1, dim[0], dim[1]], name='attention_branch_features')
            a_i = tf.reshape(features, [-1, dim[1]])
            a_i = slim.fully_connected(inputs=a_i, num_outputs=dim[1], biases_initializer=None, scope='a_i')
            a_i = tf.reshape(a_i, [-1, dim[0], dim[1]])
            gru_cell = tf.contrib.rnn.GRUCell(num_units=dim[1])
            
            # defining initial state
            # state = gru_cell.zero_state(tf.shape(endpoints['resnet_v2_50/block4'])[0], dtype=tf.float32)
            _input = tf.reduce_mean(features, 1)
            state = slim.fully_connected(inputs=tf.reduce_mean(features, 1), num_outputs=D, biases_initializer=None, scope='init_state')
            
            attention_maps = []
            _masked = []
            signals = []
            _masked.append(features)

            with tf.variable_scope("GRU_Attention"):
                for i in range(attention_steps):
                    if i > 0: tf.get_variable_scope().reuse_variables()
                    # state is in shape (?, 64)
                    output, state = gru_cell(_input, state)
                    h = tf.expand_dims(slim.fully_connected(inputs=state, num_outputs=dim[1], biases_initializer=None, scope='hidden2h'), 1)
                    stop_check = tf.round(tf.sigmoid(slim.fully_connected(inputs=h, num_outputs=1, biases_initializer=None, scope='stop_check')))
                    signals.append(stop_check)
                    e = tf.reshape(tf.add(a_i, h), [-1, dim[1]])
                    _att = slim.fully_connected(inputs=e, num_outputs=1, scope='e2attention')
                    _alpha = tf.nn.softmax(tf.reshape(_att, [-1, dim[0]]))
                    attention_maps.append(_alpha)
                    _mask = tf.multiply(features, tf.expand_dims(_alpha, 2))
                    _masked.append(_mask * tf.reduce_min(tf.stack(signals, 1), 1))
                    _input = tf.reduce_sum(_mask, 1)

    _mask_concat = tf.concat(_masked[:-1], 2)
    _masked = tf.reshape(_mask_concat, [-1, M, M, attention_steps * dim[1]])
    
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
