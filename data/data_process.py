# -- coding: utf-8 --

import pandas as pd
import csv
import os
from model.hyparameter import parameter
import argparse


train_file='train_resource.csv'
test_file='test_resource.csv'

def train(file_address=None, target_site=None):
    '''
    :param file_address:
    :param target_site:
    :return:
    '''
    if os.path.exists('train.csv'):
        pass
        print('the training or validation data set has been exited!')
    elif os.path.exists(file_address):
        data=pd.read_csv(filepath_or_buffer=file_address,encoding='utf-8')

        train_data=data.loc[data['station']==target_site]

        print(train_data.values.shape)

        file = open('train.csv', 'w', encoding='utf-8')
        writer = csv.writer(file)
        print(train_data.keys())
        writer.writerow(['month','day','hour','AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'])

        for line in train_data.values:  # save
            st = line[0].split(' ')
            a = st[0].split('/')
            b = st[1].split(':')
            time=[float(a[1]), float(a[2]), float(b[0])]

            writer.writerow(time+list(line[2:]))
        file.close()


        print('create training or validation data set successful!')
    else:
        print('can not to find the original data set, maybe you give the wrong file address, please to check carefully!')

def test(file_address=None, target_site=None):
    '''
    :param file_address:
    :param target_site:
    :return:
    '''
    if os.path.exists('test.csv'):
        pass
        print('the test data set has been exited!')
    elif os.path.exists(file_address):
        data=pd.read_csv(filepath_or_buffer=file_address,encoding='utf-8')

        test_data=data.loc[data['station']==target_site]

        print(test_data.values.shape)

        file = open('test.csv', 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(['month', 'day', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'])

        for line in test_data.values:  # save
            st = line[0].split(' ')
            a = st[0].split('/')
            b = st[1].split(':')
            time = [float(a[1]), float(a[2]), float(b[0])]

            writer.writerow(time + list(line[2:]))
        file.close()


        print('create test data set successful!')
    else:
        print('can not to find the original data set, maybe you give the wrong file address, please to check carefully!')
    return



if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    # to generator training and validation data set
    train(file_address=train_file, target_site='Qingpu')

    # to generator test data set
    test(file_address=test_file, target_site='Qingpu')