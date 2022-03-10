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
        self.batch_size=hp.batch_size
        self.h=hp.h
        self.w=hp.w
        self.placeholders=placeholders

    def cnn(self,x):
        '''
        :param x: shape is [batch size, site num, features, channel]
        :return: shape is [batch size, height, channel]
        '''

        with tf.variable_scope('cnn_layer',reuse=tf.AUTO_REUSE):
            filter1= tf.get_variable("weight1",[self.h,self.w,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1=tf.get_variable("bias1",[32], initializer=tf.constant_initializer(0.1))
            # filter1=tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w,1,32]),name='filter1')
            layer1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME')
            # bn1=tf.layers.batch_normalization(layer1,training=self.placeholders['is_training'])
            relu1=tf.nn.relu(tf.nn.bias_add(layer1,bias1))

            max_pool1=tf.nn.avg_pool(relu1, ksize=[1, 41, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # print('max_pool1 output shape is : ',max_pool1.shape)
            filter2 = tf.get_variable("weight2",[self.h,self.w, 32, 32],initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias2 = tf.get_variable("bias2", [32],initializer=tf.constant_initializer(0.1))
            # filter2 = tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w, 32, 32]), name='filter2')
            layer2 = tf.nn.conv2d(input=max_pool1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
            # bn2=tf.layers.batch_normalization(layer2,training=self.placeholders['is_training'])
            relu2=tf.nn.relu(tf.nn.bias_add(layer2,bias2))

            max_pool2=tf.nn.avg_pool(relu2, ksize=[1, 41, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # print('max_pool2 output shape is : ',max_pool2.shape)
            #
            # filter3 = tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w, 32, 64]), name='filter3')
            # layer3 = tf.nn.conv2d(input=max_pool2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
            # bn3=tf.layers.batch_normalization(layer3,training=self.placeholders['is_training'])
            # relu3=tf.nn.relu(bn3)
            #
            # max_pool3=tf.nn.max_pool(relu3, ksize=[1,self.h,self.w, 1], strides=[1, 2, 2, 1], padding='SAME')

            # print('max_pool3 output shape is : ', max_pool3.shape)

            cnn_shape = max_pool2.get_shape().as_list()
            nodes = cnn_shape[1] * cnn_shape[2] * cnn_shape[3]
            cnn_out = tf.reshape(max_pool2, [-1, nodes])
            '''shape is  : [batch size, site num, features, channel]'''
            # relu3=tf.reduce_mean(relu2,axis=3)

            # print('cnn_out shape is : ',cnn_out.shape)

            # x=tf.reduce_sum(tf.squeeze(x,axis=-1),axis=1)
            # x=tf.layers.dense(inputs=x,units=self.nodes,name='residual',reuse=tf.AUTO_REUSE)

            s=tf.layers.dense(inputs=cnn_out, units=self.nodes, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)

            # s=tf.add(x,s)

            # print('cnn output shape is : ',s.shape)
        return s