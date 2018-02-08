import tensorflow as tf
from tensorflow.contrib import slim

def head(endpoints, embedding_dim, is_training):
    endpoints['emb'] = endpoints['emb_raw'] = slim.fully_connected(
        endpoints['model_output'], embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')

    endpoints['model_logits'] = endpoints['model_raw'] = slim.fully_connected(
        endpoints['model_output'], 1232, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='model_cls')

    endpoints['color_logits'] = endpoints['color_raw'] = slim.fully_connected(
        endpoints['model_output'], 11, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='color_cls')

    return endpoints
