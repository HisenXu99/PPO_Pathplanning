# -*- coding: UTF-8 -*-
#没写oldpi网络的情况
from cv2 import merge
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow import keras
import time
import cv2
import os
import math
import sys


# A_LR = 0.001
# A_DIM =11
# UPDATE_STEPS = 10

class PPO:
    def __init__(self,loadpath,LR):
        self.GAMMA=0.9

#########################################Network#################################
        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512
        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4
        # Parameters for CNN
        self.img_size = 80  # input image size
        self.first_conv = [8, 8, self.Num_stackFrame, 32]
        self.second_conv = [4, 4, 32, 64]
        self.third_conv = [3, 3, 64, 64]
        self.first_dense = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], 11]

        self.n_actions=11
        self.n_features=4

#########################################Train#################################
        self.ent_coef=0.01
        self.vf_coef=0.5

        self.lr=LR



        self.load_path = loadpath

        self.epsilon=0.2

        self.Num_action = 11


#######################################Initialize Network###############################
        self.network()
        self.sess,self.saver=self.init_sess()
        pass

    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):  # 初始化偏置项
        return tf.Variable(self.xavier_initializer(shape))

    # Xavier Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(2.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    def conv2d(self, x, w, stride):  # 定义一个函数，用于构建卷积层
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    # Xavier Weights initializer
    def normc_initializer(self,std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)
        return _initializer



    def network(self):
        tf.reset_default_graph()

        # Input------image
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.Num_stackFrame],name="image")
        self.x_normalize = (self.x_image - (255.0/2)) / (255.0/2)  # 归一化处理

        self.s = tf.placeholder(tf.float32, [None, 4], "state")
        self.a = tf.placeholder(tf.float32, None, "action")
        self.td = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Network'):
            with tf.variable_scope('CNN'):
                w_conv1 = self.weight_variable(self.first_conv)  # w_conv1 = ([8,8,4,32])
                b_conv1 = self.bias_variable([self.first_conv[3]])  # b_conv1 = ([32])

                # second_conv=[4,4,32,64]
                w_conv2 = self.weight_variable(self.second_conv)  # w_conv2 = ([4,4,32,64])
                b_conv2 = self.bias_variable([self.second_conv[3]])  # b_conv2 = ([64])

                # third_conv=[3,3,64,64]
                w_conv3 = self.weight_variable(self.third_conv)  # w_conv3 = ([3,3,64,64])
                b_conv3 = self.bias_variable([self.third_conv[3]])  # b_conv3 = ([64])

                h_conv1 = tf.nn.relu(self.conv2d(self.x_normalize, w_conv1, 4) + b_conv1)
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
                h_pool3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64])  # 将tensor打平到vector中

                cnn_out1 = tf.layers.dense(
                    inputs=h_pool3_flat,
                    units=512,    # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='cnn_out1'
                )
                cnn_out2 = tf.layers.dense(
                    inputs=cnn_out1,
                    units=self.n_features,    # output units
                    activation=tf.nn.softmax,   # get action probabilities
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='cnn_out2'
                )

            h_concat = tf.concat([cnn_out2, self.s], axis=1)



            with tf.variable_scope('AC'):
                # h_concat = tf.concat([h_pool3_flat, rnn_out], axis=1)
                with tf.variable_scope('Actor'):
                    l1 = tf.layers.dense(
                        inputs=h_concat,
                        units=512,    # number of hidden units
                        activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        name='l1'
                    )
                    self.acts_prob = tf.layers.dense(
                        inputs=l1,
                        units=self.n_actions,    # output units
                        activation=tf.nn.softmax,   # get action probabilities
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        name='acts_prob'
                    )

                with tf.variable_scope('Critic'):
                    l1 = tf.layers.dense(
                        inputs=h_concat,
                        units=512,  # number of hidden units
                        activation=tf.nn.relu,  # None
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        name='l1'
                    )

                    self.v = tf.layers.dense(
                        inputs=l1,
                        units=1,  # output units
                        activation=None,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        name='V'
                    )

        with tf.variable_scope('Train'):
            self.a = tf.placeholder(tf.float32, [None,self.n_actions], "action")
            self.td = tf.placeholder(tf.float32, [None,1], "td_error")  # TD_error
            self.discount_r = tf.placeholder(tf.float32, [None, 1], "v_next")
            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.discount_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.td_error))   # TD_error = (r+gamma*V_next) - V_eval
            with tf.variable_scope('exp_v'):
                log_prob = tf.log(tf.reduce_sum(self.acts_prob*self.a,axis = 1 ))
                self.exp_v = tf.reduce_mean(log_prob * self.td)  # advantage (TD_error) guided loss
            loss=-self.exp_v+0.5*self.closs
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        with tf.variable_scope('Record'):
            aloss=tf.summary.scalar("Actor_loss",self.exp_v )
            closs=tf.summary.scalar("Critic_loss", self.closs)
            total_loss=tf.summary.scalar("Total_loss",loss )
            self.merged=tf.summary.merge_all()  
        pass

    
    def init_sess(self):
        # Initialize variables
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存  
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, self.load_path + "/model.ckpt")
        print("Model restored.")

        return sess,saver

    def choose_action(self,s,image):
        probs = self.sess.run(self.acts_prob, {self.s:s,self.x_image:image})   # get probabilities for all actions
        # with open('Empty/saved_networks/'+self.date_time+'/value.txt','a') as f:             
        #     # f.write('prob')
        #     f.write('\r')
        #     np.savetxt(f, probs, delimiter=',', fmt = '%s')
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
        pass


    def train(self,state_stack,image_stack,rewards_stack,actions_stack,train_step,next_state,next_image):
        v_s_ = self.sess.run(self.v, {self.s:next_state,self.x_image:next_image})[0,0]
        #用来看看prob的变化
        probs = self.sess.run(self.acts_prob, {self.s:next_state,self.x_image:next_image})   # get probabilities for all actions
        discounted_r = []
        for r in rewards_stack[::-1]:   #buffer_r[::-1]返回倒序的原list
            v_s_ = r + self.GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r=np.array(discounted_r)[:, np.newaxis]
        td_error,v= self.sess.run([self.td_error,self.v],
                                {self.s:state_stack, self.x_image:image_stack,self.discount_r: discounted_r})

        feed_dict = {self.s:state_stack, self.x_image:image_stack, self.a: actions_stack, self.td: td_error, self.discount_r: discounted_r}
        _, _ = self.sess.run([self.train_op, self.merged], feed_dict)
        return v_s_,probs

    def save_model(self):
        # ------------------------------
        save_path = self.saver.save(
            self.sess, 'Obstacle/saved_networks/'+self.date_time + "/model.ckpt")
        # ------------------------------
        pass