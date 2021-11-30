# -- coding: utf-8 --

import tensorflow as tf

class Dcoderlstm(object):
    def __init__(self,batch_size,predict_time,layer_num=1,nodes=128,placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.predict_time=predict_time
        self.placeholders=placeholders

        self.decoder()


    def lstm_cell(self):
        '''
        :return: lstm
        '''
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.nodes)
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.placeholders['dropout'])

    def decoder(self):
        self.mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        self.initial_state=self.mlstm_cell.zero_state(self.batch_size,tf.float32)

    def attention(self, h_t, encoder_hs):
        '''
        h_t for decoder, the shape is [batch, h]
        encoder_hs for encoder, the shape is [batch, time ,h]
        :param h_t:
        :param encoder_hs:
        :return:
        '''
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t,multiples=[1,encoder_hs.shape[1],1])), 2)
        a_t = tf.nn.softmax(scores)  # [batch, time]
        a_t = tf.expand_dims(a_t, 2) # [batch, time, 1]
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[0,2,1]), a_t) #[batch ,h , 1]
        c_t = tf.squeeze(c_t, axis=2) #[batch, h]]
        h_t=tf.squeeze(h_t,axis=1)
        h_tld  = tf.layers.dense(tf.concat([h_t, c_t], axis=1),units=c_t.shape[-1],activation=tf.nn.relu) #[batch, h]
        return h_tld

    def decoding(self,encoder_hs, emb):
        '''
        :param h_state:
        :return:
        '''
        h=[]
        initial_state=self.initial_state
        h_state=encoder_hs[:,-1,:]
        for i in range(self.predict_time):
            h_state = tf.expand_dims(input=h_state,axis=1)
            h_state = tf.add(h_state, emb[:,i:i+1,:])
            with tf.variable_scope('decoder_lstm', reuse=tf.AUTO_REUSE):
                h_state, state = tf.nn.dynamic_rnn(cell=self.mlstm_cell, inputs=h_state,initial_state=initial_state,dtype=tf.float32)
                h_state=self.attention(h_t=h_state,encoder_hs=encoder_hs) # attention
                initial_state=state
                results=tf.layers.dense(inputs=h_state,units=1,name='layer',reuse=tf.AUTO_REUSE)
            h.append(results)
        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h),[1,2,0]),axis=1)