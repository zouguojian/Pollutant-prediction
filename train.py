# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function
from spatial_temporal_model.hyparameter import parameter
from spatial_temporal_model.encoder import cnn_lstm
from model.decoder import Dcoderlstm
from model.utils import construct_feed_dict
from model.encoder import Encoderlstm

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model.normalization as normalization

import spatial_temporal_model.process as data_load
import os
import argparse

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logs_path="board"

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=False,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0, stddev=1, seed=0))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs

class Model(object):
    def __init__(self,para):
        self.para=para
        self.pollutant_id={'AQI':0, 'PM2.5':1,'PM10':3, 'SO2':5, 'NO2':7, 'O3':9, 'CO':13}

        # define placeholders
        self.placeholders = {
            # None : batch _size * time _size
            'month': tf.placeholder(tf.int32, shape=(None, self.para.input_length+self.para.output_length), name='input_month'),
            'day': tf.placeholder(tf.int32, shape=(None, self.para.input_length+self.para.output_length), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.para.input_length+self.para.output_length), name='input_hour'),
            'features1': tf.placeholder(tf.float32, shape=[None, self.para.input_length, self.para.site_num, self.para.features1],name='input_1'),
            'features2': tf.placeholder(tf.float32, shape=[None, self.para.input_length, self.para.features2], name='input_2'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.para.output_length]),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'is_training': tf.placeholder(tf.bool, shape=(),name='input_is_training'),
        }
        self.model()

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''

        with tf.variable_scope('month'):
            self.m_emb = embedding(self.placeholders['month'], vocab_size=13, num_units=self.para.hidden_size,
                                    scale=False, scope="month_embed")
            print('d_emd shape is : ', self.m_emb.shape)

        with tf.variable_scope('day'):
            self.d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.para.hidden_size,
                                    scale=False, scope="day_embed")
            print('d_emd shape is : ', self.d_emb.shape)

        with tf.variable_scope('hour'):
            self.h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.hidden_size,
                                    scale=False, scope="hour_embed")
            print('h_emd shape is : ', self.h_emb.shape)

        # create model

        # this step use to encoding the input series data
        '''
        rlstm, return --- for example ,output shape is :(32, 3,  128)
        axis=0: bath size
        axis=1: input data time size
        axis=2: output feature size
        '''

        # shape is [batch, input length, embedding size]
        emb=tf.add_n([self.m_emb,self.d_emb,self.h_emb])

        # cnn时空特征提取
        l = cnn_lstm(batch_size=self.para.batch_size,
                     layer_num=self.para.hidden_layer,
                     nodes=self.para.hidden_size,
                     highth=self.para.h,
                     width=self.para.w,
                     placeholders=self.placeholders)

        # [batch, time ,hidden size]
        (h_states1, c_states1) = l.encoding(self.placeholders['features1'], emb[:,:self.para.input_length,:])
        print('h_states1 shape is : ', h_states1.shape)

        # lstm 时序特征提取
        encoder_init =Encoderlstm(self.para.batch_size,
                                    self.para.hidden_layer,
                                    self.para.hidden_size,
                                    placeholders=self.placeholders)

        ## [batch, time , hidden size]
        (h_states2, c_states2) = encoder_init.encoding(self.placeholders['features2'], emb[:,:self.para.input_length,:])

        print('h_states2 shape is : ', h_states2.shape)

        h_states=tf.layers.dense(tf.concat([h_states1,h_states2],axis=-1),units=self.para.hidden_size, activation=tf.nn.relu, name='layers')

        # this step to predict the pollutant concentration
        '''
        decoder, return --- for example ,output shape is :(32, 162, 1)
        axis=0: bath size
        axis=1: numbers of the nodes
        axis=2: label size
        '''
        decoder_init = Dcoderlstm(self.para.batch_size,
                                    self.para.output_length,
                                    self.para.hidden_layer,
                                    self.para.hidden_size,
                                    placeholders=self.placeholders)

        self.pres = decoder_init.decoding(h_states, emb[:,self.para.input_length: ,:])
        print('pres shape is : ', self.pres.shape)

        self.cross_entropy = tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels']), axis=0)))

        print(self.cross_entropy)
        print('cross shape is : ',self.cross_entropy.shape)

        tf.summary.scalar('cross_entropy',self.cross_entropy)
        # backprocess and update the parameters
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)

        print('#...............................in the training step.....................................#')

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def accuracy(self,label,predict):
        '''
        :param label: represents the observed value
        :param predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        error = label - predict
        average_error = np.mean(np.fabs(error.astype(float)))
        print("mae is : %.6f" % (average_error))

        rmse_error = np.sqrt(np.mean(np.square(label - predict)))
        print("rmse is : %.6f" % (rmse_error))

        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
        print('correlation coefficient is: %.6f' % (cor))

        # mask = label != 0
        # mape =np.mean(np.fabs((label[mask] - predict[mask]) / label[mask]))*100.0
        # mape=np.mean(np.fabs((label - predict) / label)) * 100.0
        # print('mape is: %.6f %' % (mape))
        sse = np.sum((label - predict) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        R2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        print('r^2 is: %.6f' % (R2))

        return average_error,rmse_error,cor,R2

    def describe(self,label,predict,prediction_size):
        '''
        :param label:
        :param predict:
        :param prediction_size:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label, 'b*:', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict, 'r*:', label=u'predicted value')
        # use the legend
        # plt.legend()
        plt.xlabel("time(hours)", fontsize=17)
        plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
        plt.title("the prediction of pm$_{2.5}", fontsize=17)
        plt.show()

    def initialize_session(self):
        self.sess=tf.Session()
        self.saver=tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return [num*(max-min)+min for num in a]

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''

        max_mae = 100
        self.sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

        self.iterate = data_load.DataIterator(site_id=self.para.target_site_id,
                                              site_num=self.para.site_num,
                                              pollutant_id=self.pollutant_id[self.para.pollutant_id],
                                              is_training=self.para.is_training,
                                              time_size=self.para.input_length,
                                              prediction_size=self.para.output_length,
                                              data_divide=self.para.data_divide,
                                              window_step=self.para.step,
                                              normalize=self.para.normalize)

        next_elements=self.iterate.next_batch(batch_size=self.para.batch_size, epochs=self.para.epochs,is_training=True)

        for i in range(int((self.iterate.train_p.shape[0] - (self.para.input_length + self.para.output_length))//self.para.step)
                    // self.para.batch_size * self.para.epochs):

            x1, x2, m, d, h, label =self.sess.run(next_elements)
            features1 = np.reshape(np.array(x1), [-1, self.para.input_length, self.para.site_num, self.para.features1])
            features2 = np.reshape(np.array(x2), [-1, self.para.input_length, self.para.features2])
            feed_dict = construct_feed_dict(features1, features2, m, d, h, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})
            feed_dict.update({self.placeholders['is_training']: self.para.is_training})

            summary, loss, _ = self.sess.run((merged,self.cross_entropy,self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))
            # writer.add_summary(summary, loss)

            # validate processing
            if i % 50 == 0:
                mae_error=self.evaluate()

                if max_mae>mae_error:
                    print("the validate average mae loss value is : %.6f" % (mae_error))
                    max_mae=mae_error
                    self.saver.save(self.sess,save_path=self.para.save_path+'model.ckpt')

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_list = list()
        predict_list = list()

        #with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

        self.iterate_test = data_load.DataIterator(site_id=self.para.target_site_id,
                                                   site_num=self.para.site_num,
                                                    pollutant_id=self.pollutant_id[self.para.pollutant_id],
                                                    is_training=self.para.is_training,
                                                    time_size=self.para.input_length,
                                                    prediction_size=self.para.output_length,
                                                    data_divide=self.para.data_divide,
                                                    normalize=self.para.normalize)

        next_ = self.iterate_test.next_batch(batch_size=self.para.batch_size, epochs=1,is_training=False)
        max,min=self.iterate_test.max_p[self.pollutant_id[self.para.pollutant_id]],self.iterate_test.min_p[self.pollutant_id[self.para.pollutant_id]]

        for i in range(int((self.iterate_test.test_p.shape[0] -(self.para.input_length+ self.para.output_length))//self.para.output_length)
                       // self.para.batch_size):
            x1, x2, m, d, h, label =self.sess.run(next_)
            features1 = np.reshape(np.array(x1), [-1, self.para.input_length, self.para.site_num, self.para.features1])
            features2 = np.reshape(np.array(x2), [-1, self.para.input_length, self.para.features2])

            feed_dict = construct_feed_dict(features1, features2,  m, d, h, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            feed_dict.update({self.placeholders['is_training']: self.para.is_training})

            pre = self.sess.run((self.pres), feed_dict=feed_dict)
            label_list.append(label)
            predict_list.append(pre)

        label_list=np.reshape(np.array(label_list,dtype=np.float32),[-1, self.para.output_length])
        predict_list=np.reshape(np.array(predict_list,dtype=np.float32),[-1, self.para.output_length])

        if self.para.normalize:
            label_list = np.array([self.re_current(row,max,min) for row in label_list])
            predict_list = np.array([self.re_current(row,max,min) for row in predict_list])
        else:
            label_list = np.array([row for row in label_list])
            predict_list = np.array([row for row in predict_list])

        # np.savetxt('results/results_label.txt',label_list,'%.3f')
        # np.savetxt('results/results_predict.txt', predict_list, '%.3f')

        label_list=np.reshape(label_list,[-1])
        predict_list=np.reshape(predict_list,[-1])

        print(label_list)
        print(predict_list)

        average_error, rmse_error, cor, R2= self.accuracy(label_list, predict_list)  #产生预测指标
        # self.describe(label_list, predict_list, self.para.output_length)   #预测值可视化
        return average_error

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:para.is_training = True
    else:
        para.batch_size=1
        para.is_training = False

    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#...................................finished............................................#')

if __name__ == '__main__':
    main()