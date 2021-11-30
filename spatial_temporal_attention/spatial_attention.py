# -- coding: utf-8 --

import tensorflow as tf

def AttentionLayer(inputs, hidden_size):
    '''
    :param inputs:  inputs shape:[batch_size, max_time, hidden_size]
    :param name:
    :param hidden_size:
    :return:
    '''

    with tf.variable_scope(name='spatial_attention'):
        # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
        # 因为使用双向GRU，所以其长度为2×hidden_szie
        u_context = tf.Variable(tf.truncated_normal([hidden_size]), name='u_context')
        #使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
        h = tf.layers.dense(inputs, hidden_size, activation_fn=tf.nn.tanh)
        #shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output