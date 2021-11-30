# -- coding: utf-8 --

import tensorflow as tf

def AttentionLayer(inputs, hidden_size):
    '''
    :param inputs:  inputs shape:[batch_size, input length, hidden_size]
    :param name:
    :param hidden_size:
    :return: [batch, hidden size]
    '''

    with tf.variable_scope('spatial_attention', reuse=tf.AUTO_REUSE):
        # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
        # 因为使用双向GRU，所以其长度为2×hidden_szie
        # u_context = tf.Variable(tf.truncated_normal([hidden_size]), name='u_context')
        u_context = tf.get_variable("u_context", [hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
        h = tf.layers.dense(inputs, hidden_size, activation=tf.nn.tanh)
        #shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output