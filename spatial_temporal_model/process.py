# -- coding: utf-8 --

# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import argparse
from spatial_temporal_model.hyparameter import parameter
import pandas as pd

class DataIterator():
    def __init__(self,
                 site_id=0,
                 site_num=9,
                 pollutant_id=4,
                 is_training=True,
                 time_size=3,
                 prediction_size=1,
                 data_divide=0.9,
                 window_step=1,
                 normalize=False):
        '''
        :param is_training: while is_training is True,the model is training state
        :param field_len:
        :param time_size:
        :param prediction_size:
        :param target_site:
        '''

        self.min_value=0.000000000001
        self.site_id=site_id                   # ozone ID
        self.site_num=site_num
        self.pollutant_id=pollutant_id
        self.time_size=time_size               # time series length of input
        self.prediction_size=prediction_size   # the length of prediction
        self.is_training=is_training           # true or false
        self.data_divide=data_divide           # the divide between in training set and test set ratio
        self.window_step=window_step           # windows step
        self.train_data=self.get_source_data('/Users/guojianzou/pollutant-prediction/data/all_train.csv').values
        self.test_data=self.get_source_data('/Users/guojianzou/pollutant-prediction/data/all_test.csv').values

        # self.data=self.source_data.loc[self.source_data['ZoneID']==self.site_id]
        print(self.train_data.shape, self.test_data.shape)
        self.total_data=np.concatenate([self.train_data,self.test_data],axis=0)

        # 初始化每时刻的输入
        self.max,self.min=self.get_max_min(self.total_data)   # max and min are list type, used for the later normalization
        self.normalize=normalize
        if self.normalize:
            self.normalization(self.train_data) #normalization
            self.normalization(self.test_data)  # normalization

    def get_source_data(self,file_path):
        '''
        :return:
        '''

        data=None
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
        except IOError:
            print("Error: do not to find or failed to read the file")
        else:
            print("successful to read the data")
        return data

    def get_max_min(self, data):
        '''
        :return: the max and min value of input features
        '''
        self.min_list=[]
        self.max_list=[]
        # print('the shape of features is :',data.shape[1])
        for i in range(data.shape[1]):
            self.min_list.append(min(data[:,i]))
            self.max_list.append(max(data[:,i]))
        print('the max feature list is :',self.max_list)
        print('the min feature list is :', self.min_list)
        return self.max_list,self.min_list

    def normalization(self,data):
        for i in range(data.shape[1]):
            data[:,i]=(data[:,i] - np.array(self.min[i])) / (np.array(self.max[i]) - np.array(self.min[i]+self.min_value))

    def generator(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''

        if self.is_training: data=self.train_data
        else: data=self.test_data

        shape=data.shape

        low,high=0,shape[0]

        while (low+(self.time_size+self.prediction_size) * self.site_num) <= high:
            label=data[low + self.time_size * self.site_num : low + (self.time_size + self.prediction_size) * self.site_num,self.pollutant_id: self.pollutant_id+1]
            label=np.concatenate([label[i * self.site_num : (i + 1) * self.site_num, :] for i in range(self.prediction_size)], axis=1)

            yield (np.array(data[low:low+self.time_size * self.site_num]), label[self.site_id])
            if self.is_training: low += self.window_step * self.site_num
            else: low += self.prediction_size * self.site_num

    def next_batch(self, batch_size=32, epochs=1, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training

        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32,tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.train_data.shape[0]-self.time_size-self.prediction_size)//self.window_step)
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
#
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataIterator(site_id=0,normalize=True,time_size=48,prediction_size=24,window_step=para.step)

    # print(iter.data.loc[iter.data['ZoneID']==0])
    next=iter.next_batch(32,10,is_training=True)
    with tf.Session() as sess:
        for i in range(4000):
            print(i)
            x,y=sess.run(next)
            print(x.shape)
            print(y.shape)