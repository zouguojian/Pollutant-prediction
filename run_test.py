# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function
from model.hyparameter import parameter
from model.encoder import Encoderlstm
from model.decoder import Dcoderlstm
from model.utils import construct_feed_dict

import pandas as pd
import tensorflow as tf
import numpy as np
import model.decoder as decoder
import matplotlib.pyplot as plt
import model.normalization as normalization

import model.process as data_load
import os
import argparse

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logs_path="board"

class Model(object):
    def __init__(self,para):
        self.para=para
        self.pollutant_id={'AQI':3, 'PM2.5':4,'PM10':5, 'SO2':6, 'NO2':7, 'O3':8, 'CO':9}

        # define placeholders
        self.placeholders = {
            # None : batch _size * time _size
            'features': tf.placeholder(tf.float32, shape=[None, self.para.input_length, self.para.features]),
            'labels': tf.placeholder(tf.float32, shape=[None, self.para.output_length]),
            'dropout': tf.placeholder_with_default(0., shape=())
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

        '''
        feedforward and BN layer
        output shape:[batch, time_size,field_size,new_features]
        '''
        normal=normalization.Normalization(inputs=self.placeholders['features'],out_size=self.para.features,is_training=self.para.is_training)
        normal.normal()

        # create model

        # this step use to encoding the input series data
        '''
        rlstm, return --- for example ,output shape is :(32, 3,  128)
        axis=0: bath size
        axis=1: input data time size
        axis=2: output feature size
        '''
        encoder_init =Encoderlstm(self.para.batch_size,
                                    self.para.hidden_layer,
                                    self.para.hidden_size,
                                    placeholders=self.placeholders)

        ## [batch, time ,h]
        (h_states, c_states) = encoder_init.encoding(self.placeholders['features'])

        print('h_states shape is : ', h_states.shape)

        # encoder_init=encoder_lstm.lstm(self.para.batch_size,
        #                                self.para.hidden_layer,
        #                                self.para.hidden_size,
        #                                self.para.is_training)
        # encoder_init=encodet_gru.gru(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # encoder_init=encoder_rnn.rnn(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # (h_states,c_states)=encoder_init.encoding()

        # this step to presict the polutant concentration
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

        self.pres = decoder_init.decoding(h_states)
        print('pres shape is : ', self.pres.shape)

        self.cross_entropy = tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels']), axis=0)))

        print(self.cross_entropy)
        print('cross shape is : ',self.cross_entropy.shape)

        tf.summary.scalar('cross_entropy',self.cross_entropy)
        # backprocess and update the parameters
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)

        print('#...............................in the training step.....................................#')

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

    def test(self):
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
                                                    pollutant_id=self.pollutant_id[self.para.pollutant_id],
                                                    is_training=self.para.is_training,
                                                    time_size=self.para.input_length,
                                                    prediction_size=self.para.output_length,
                                                    data_divide=self.para.data_divide,
                                                    normalize=self.para.normalize)

        next_ = self.iterate_test.next_batch(batch_size=self.para.batch_size, epochs=1,is_training=False)
        max,min=self.iterate_test.max_list[self.pollutant_id[self.para.pollutant_id]],self.iterate_test.min_list[self.pollutant_id[self.para.pollutant_id]]

        for i in range(int((self.iterate_test.test_data.shape[0] -(self.para.input_length+ self.para.output_length))//self.para.output_length)
                       // self.para.batch_size):
            x, label =self.sess.run(next_)
            feed_dict = construct_feed_dict(x, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})

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

        np.savetxt('results/results_label.txt',label_list,'%.3f')
        np.savetxt('results/results_predict.txt', predict_list, '%.3f')

        label_list=np.reshape(label_list,[-1])
        predict_list=np.reshape(predict_list,[-1])
        average_error, rmse_error, cor, R2= self.accuracy(label_list, predict_list)  #产生预测指标
        self.describe(label_list, predict_list, self.para.output_length)   #预测值可视化
        return rmse_error

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    para.batch_size=1
    para.is_training = False

    pre_model = Model(para)
    pre_model.initialize_session()

    pre_model.test()

    print('#...................................finished............................................#')

if __name__ == '__main__':
    main()