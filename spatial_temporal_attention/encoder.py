# -- coding: utf-8 --

import tensorflow as tf
from spatial_temporal_attention.spatial_attention import AttentionLayer

class s_lstm(object):
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

    def s_attention(self,x):
        '''
        :param x: shape is [batch size, input length, features]
        :return: shape is [batch size, features]
        '''

        with tf.variable_scope('s_attention',reuse=tf.AUTO_REUSE):
            x=tf.layers.dense(inputs=x, units=self.nodes, name='s_att',reuse=tf.AUTO_REUSE)

            x=AttentionLayer(inputs=x,hidden_size=self.nodes)
        return x

    def encoding(self, inputs, emb):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''

        # for s_att to extract spatial features
        stt_out=list()
        for time in range(inputs.shape[1]):
            stt_out.append(self.s_attention(inputs[:,time,:,:]))

        inputs=tf.transpose(stt_out,[1,0,2])
        inputs =tf.add(inputs,emb)

        print('the stt output series shape is ; ', inputs.shape)

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