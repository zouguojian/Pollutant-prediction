# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import argparse
from spatial_temporal_model.hyparameter import parameter
import pandas as pd

class DataIterator():
    def __init__(self,
                 site_id=0,
                 site_num=41,
                 pollutant_id=2,
                 is_training=True,
                 time_size=48,
                 prediction_size=24,
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

        # 读取电厂的数据和某个站点的污染物浓度
        self.train_x_y=self.get_source_data('/Users/guojianzou/Documents/program/shanghai_weather/train_weather_day.csv').values[:,2:]
        self.test_x_y=self.get_source_data('/Users/guojianzou/Documents/program/shanghai_weather/test_weather_day.csv').values[:,2:]

        print(self.train_x_y.shape)
        print(self.test_x_y.shape)

        # print('the resource dada shape are : ',self.data_s.shape, self.data_p.shape)

        # 寻找数据集中的最大值和最小值
        self.max_train,self.min_train=self.get_max_min(self.train_x_y)
        self.max_test, self.min_test = self.get_max_min(self.test_x_y)
        self.max,self.min=[max(self.max_train[i],self.max_test[i]) for i in range(len(self.max_train))],[min(self.min_train[i],self.min_test[i]) for i in range(len(self.min_train))]

        print('the max feature list is :', self.max)
        print('the min feature list is :', self.min)

        self.normalize=normalize
        if self.normalize:
            self.normalization(self.train_x_y, self.min, self.max) # normalization
            self.normalization(self.test_x_y, self.min, self.max) # normalization

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
        min_list=[]
        max_list=[]

        for i in range(data.shape[1]):
            min_list.append(min(data[:,i]))
            max_list.append(max(data[:,i]))
        return max_list, min_list

    def normalization(self,data, min, max):
        for i in range(data.shape[1]):
            data[:,i]=(data[:,i] - np.array(min[i])) / (np.array(max[i]) - np.array(min[i]+self.min_value))

    def generator(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''

        if self.is_training:
            data_s=self.train_x_y
        else:
            data_s=self.test_x_y

        low, high =0, data_s.shape[0]

        while (low+self.time_size+self.prediction_size) <= high:
            label=data_s[low + self.time_size : low + self.time_size + self.prediction_size, 1]
            # label=np.concatenate([label[i : (i + 1), :] for i in range(self.prediction_size)], axis=1)

            yield (data_s[low:low+self.time_size],
                   label)
            if self.is_training:
                low += self.window_step
            else:
                low += self.prediction_size

    def next_batch(self, batch_size=32, epochs=1, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training

        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32, tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.train_x_y.shape[0]-self.time_size-self.prediction_size)//self.window_step)
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
#
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataIterator(site_id=0,site_num=41, data_divide=0.8, pollutant_id=2, normalize=True, time_size=48,prediction_size=24,window_step=para.step)

    # print(iter.data.loc[iter.data['ZoneID']==0])
    next=iter.next_batch(32,1,is_training=False)
    with tf.Session() as sess:
        for i in range(2):
            x,x1,y=sess.run(next)
            print(x.shape)
            print(x1.shape)
            print(y.shape)