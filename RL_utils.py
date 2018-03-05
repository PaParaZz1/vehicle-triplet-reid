"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from utils import show_stats

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
            is_train=True
            ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.is_train = is_train

        self.ep_obs, self.ep_as, self.ep_rs = None, None, None

        self.g_rl = tf.Graph()
        self._build_net()

        # self.sess = tf.Session(graph=self.g_rl)

    def train_handle(self):
        return self.g_rl, self.init_rl, self.saver_rl

    def get_sess(self, sess):
        self.sess = sess

    def _build_net(self):
        batch_norm_params = {
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'updates_collections': tf.GraphKeys.UPDATE_OPS,
                'fused': None,
                }
        with self.g_rl.as_default():
            with slim.arg_scope(
                    [slim.fully_connected],
                    # is_training=is_train,
                    weights_regularizer=slim.l2_regularizer(0.0),
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    with tf.name_scope('inputs'):
                        self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="rl_observations")
                        self.tf_acts = tf.placeholder(tf.float32, [None, self.n_features], name="rl_actions_num")
                        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="rl_actions_value")
                        dense1 = slim.fully_connected(self.tf_obs, 256, scope='rl_dense1')
                        dense2 = slim.fully_connected(dense1, self.n_actions, scope='rl_dense2')

                        # self.all_act_prob = tf.nn.softmax(dense2, name='rl_act_prob')  # use softmax to convert to probability
                        self.all_act_prob = tf.sigmoid(dense2, name='rl_act_prob')
                        # self.all_act_prob = tf.tanh(dense2, name='rl_act_prob')

            with tf.name_scope('loss'):
                # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
                # neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=dense2, labels=self.tf_acts)   # this is negative log of chosen action
                # or in this way:
                # neg_log_prob = -tf.reduce_sum(tf.log(self.all_act_prob) * self.tf_acts \
                #         + tf.log(1 - self.all_act_prob) * (1 - self.tf_acts), axis=1)
                neg_log_prob = -tf.reduce_sum(tf.log(self.all_act_prob) * self.tf_acts, axis=1)
                self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

            with tf.name_scope('train'):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
            self.init_rl = tf.global_variables_initializer()
            self.saver_rl = tf.train.Saver(max_to_keep=0)

    def choose_action(self, observation):
        # prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: np.expand_dims(observation, 0)})
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        # action = [np.random.choice(range(prob_weights.shape[1]), p=prob_weights[i]) for i in range(len(prob_weights))]
        # prob_weights = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in prob_weights]
        if self.is_train:
            show_stats('prob', prob_weights)
        action = np.round(prob_weights)
        # action = np.argmax(prob_weights)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs = s
        self.ep_as = a
        self.ep_rs = r

    def learn(self):
        if len(self.ep_as) == 0:
            return None, None
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        _ , _loss, _step = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = None, None, None    # empty episode data
        # self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm, _loss, _step

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        '''
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0.
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        '''
        discounted_ep_rs = self.ep_rs

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


class TripletStorage(object):
    
    def __init__(self):
        self._triplet_loss = None
        self._pos_embs = None
        self._neg_embs = None
        self._anchor_embs = None

    def add_storage(self, anchor_embs, pos_embs, neg_embs, losses):
        self._triplet_loss = losses
        self._anchor_embs = anchor_embs
        self._pos_embs = pos_embs
        self._neg_embs = neg_embs

    def update_loss(self, loss, idx):
        self._triplet_loss[idx] = loss

    def update_anchor_embs(self, embs, idx):
        self._anchor_embs[idx] = embs
        # print('embeddings | max {} | min {} | mean {}'.format(np.max(embs), np.min(embs), np.mean(embs)))

    def get_loss(self, idx): return self._triplet_loss[idx]

    def get_anchor_embs(self, idx): return self._anchor_embs[idx]

    def get_pos_embs(self, idx): return self._pos_embs[idx]

    def get_neg_embs(self, idx): return self._neg_embs[idx]

    def clear_storage(self):
        self._triplet_loss = None
        self._anchor_embs = None
        self._pos_embs = None
        self._neg_embs = None


'''
class Triplet(object):

    def __init__(self, index, loss):
        self.idx = index
        self.triplet_loss = loss
'''
