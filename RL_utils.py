"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

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
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        batch_norm_params = {
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'updates_collections': tf.GraphKeys.UPDATE_OPS,
                'fused': None,
                }
        with slim.arg_scope(
                [slim.fully_connected, slim.conv2d],
                # is_training=True,
                weights_regularizer=slim.l2_regularizer(0.0),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.name_scope('inputs'):
                    self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
                    self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
                    self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        
                    '''
                    dense1 = slim.fully_connected(self.tf_obs, 1024, scope='dense1')
                    dense2 = slim.fully_connected(dense1, 1024, scope='dense2')
                    dense3 = slim.fully_connected(dense2, 1024, scope='dense3')
                    dense4 = slim.fully_connected(dense3, self.n_actions, scope='dense4')
                    '''
                    dense1 = slim.fully_connected(self.tf_obs, 256, scope='dense1')
                    dense2 = slim.fully_connected(dense1, self.n_actions, scope='dense2')

                    self.all_act_prob = tf.nn.softmax(dense2, name='act_prob')  # use softmax to convert to probability

                with tf.name_scope('loss'):
                    # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
                    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense2, labels=self.tf_acts)   # this is negative log of chosen action
                    # or in this way:
                    # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
                    loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

                with tf.name_scope('train'):
                    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        # action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        action = [np.random.choice(range(prob_weights.shape[1]), p=prob_weights[i]) for i in range(len(prob_weights))]
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


class TripletStorage(object):
    
    def __init__(self):
        self.triplet_loss = None
        self.pos_embs = None
        self.neg_embs = None

    def add_storage(self, pos_embs, neg_embs, losses):
        self.triplet_loss = losses
        self.pos_embs = pos_embs
        self.neg_embs = neg_embs

    def update_loss(self, losses):
        self.triplet_loss = losses

    def get_loss(self):
        return self.triplet_loss

    def get_pos_embs(self): return self.pos_embs

    def get_neg_embs(self): return self.neg_embs

    def clear_storage(self):
        self.triplet_loss = None
        self.pos_embs = None
        self.neg_embs = None


'''
class Triplet(object):

    def __init__(self, index, loss):
        self.idx = index
        self.triplet_loss = loss
'''
