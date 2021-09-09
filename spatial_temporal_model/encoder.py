# -- coding: utf-8 --

import tensorflow as tf

class cnn_lstm(object):
    def __init__(self, batch_size, layer_num=1, nodes=128, highth=3, width=3, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.placeholders=placeholders
        self.h=highth
        self.w=width
        self.encoder()

    def cnn(self,x):
        '''
        :param x: shape is [batch size, site num, features, channel]
        :return: shape is [batch size, height, channel]
        '''

        with tf.variable_scope('cnn_layer',reuse=tf.AUTO_REUSE):
            filter1=tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w,1,32]),name='filter1')
            layer1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME')
            bn1=tf.layers.batch_normalization(layer1,training=self.placeholders['is_training'])
            relu1=tf.nn.relu(bn1)
            # max_pool1=tf.nn.max_pool(relu1, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

            # print('max_pool1 output shape is : ',max_pool1.shape)

            filter2 = tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w, 32, 32]), name='filter2')
            layer2 = tf.nn.conv2d(input=relu1, filter=filter2, strides=[1, 2, 2, 1], padding='SAME')
            bn2=tf.layers.batch_normalization(layer2,training=self.placeholders['is_training'])
            relu2=tf.nn.relu(bn2)
            # max_pool2=tf.nn.max_pool(relu2, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

            # print('max_pool2 output shape is : ',max_pool2.shape)

            filter3 = tf.Variable(initial_value=tf.random_normal(shape=[self.h,self.w, 32, 64]), name='filter3')
            layer3 = tf.nn.conv2d(input=relu2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
            bn3=tf.layers.batch_normalization(layer3,training=self.placeholders['is_training'])
            relu3=tf.nn.relu(bn3)

            max_pool3=tf.nn.max_pool(relu3, ksize=[1,self.h,self.w, 1], strides=[1, 2, 2, 1], padding='SAME')

            # print('max_pool3 output shape is : ', max_pool3.shape)

            cnn_shape = max_pool3.get_shape().as_list()
            nodes = cnn_shape[1] * cnn_shape[2] * cnn_shape[3]
            cnn_out = tf.reshape(max_pool3, [-1, nodes])
            '''shape is  : [batch size, site num, features, channel]'''
            # relu3=tf.reduce_mean(relu2,axis=3)

            # print('cnn_out shape is : ',cnn_out.shape)

            s=tf.layers.dense(inputs=cnn_out, units=128, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)

            # print('cnn output shape is : ',s.shape)
        return s

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1.0)
            return lstm_cell_
        self.e_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.initial_state = self.e_mlstm.zero_state(self.batch_size, tf.float32)

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''

        # for cnn to extract spatial features
        cnn_out=list()
        for time in range(inputs.shape[1]):
            cnn_out.append(self.cnn(tf.expand_dims(inputs[:,time,:],axis=-1)))

        inputs=tf.transpose(cnn_out,[1,0,2])

        print('the cnn output series shape is ; ', inputs.shape)

        # out put the store data
        with tf.variable_scope('encoder_cnn_lstm'):
            self.h_states, self.c_states = tf.nn.dynamic_rnn(cell=self.e_mlstm, inputs=inputs, initial_state=self.initial_state,dtype=tf.float32)

        return self.h_states, self.c_states

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,24,10])
    x=tf.placeholder(tf.float32, shape=[32, 24, 9, 10])
    r=cnn_lstm(batch_size=32,layer_num=1,nodes=128,highth=3,width=3)
    hs=r.encoding(x)


    # print(hs.shape)
    #
    # pre=r.decoding(hs)
    # print(pre.shape)