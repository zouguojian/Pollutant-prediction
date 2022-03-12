# -- coding: utf-8 --

import tensorflow as tf

class CnnClass(object):
    def __init__(self, hp=None, placeholders=None):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.hp = hp
        self.batch_size = self.hp.batch_size
        self.h = self.hp.h
        self.w = self.hp.w
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.hidden_size = self.hp.hidden_size
        self.placeholders = placeholders

    def cnn(self,x):
        '''
        :param x: shape is [batch size, site num, features, channel]
        :return: shape is [batch size, height, channel]
        '''

        with tf.variable_scope('cnn_layer',reuse=tf.AUTO_REUSE):
            layer1 = tf.layers.conv1d(inputs=x,filters=64, kernel_size=3,strides=1,padding='same')
            bias1 = tf.get_variable("bias1", [64], initializer=tf.constant_initializer(0.1))
            relu1=tf.nn.relu(tf.nn.bias_add(layer1,bias1))

            layer2 = tf.layers.conv1d(inputs=relu1, filters=128, kernel_size=3,strides=1,padding='same')
            bias2 = tf.get_variable("bias2", [128],initializer=tf.constant_initializer(0.1))
            relu2=tf.nn.relu(tf.nn.bias_add(layer2,bias2))
            relu2 = tf.transpose(relu2, [0, 2, 1])
            print('relu2 output shape is : ', relu2.shape)

            ave_pool=tf.layers.average_pooling1d(inputs=relu2,pool_size=128,strides=1,padding='valid')
            ave_pool=tf.squeeze(ave_pool,axis=1)
            print('max_pool3 output shape is : ', ave_pool.shape)

            ful = tf.layers.dense(inputs=ave_pool, units=self.hidden_size, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='full_1')
            pres = tf.layers.dense(inputs=ful, units=self.output_length, reuse=tf.AUTO_REUSE, name='full_2')
        return pres