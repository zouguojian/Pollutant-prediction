# -- coding: utf-8 --

import pandas as pd
import os
import csv


low_name=['beilun_1', 'beilun_2', 'beilun_3', 'beilun_4', 'beilun_5', 'beilun_6', 'beilun_7',
          'cangnan_1', 'cangnan_2',
          'caoejiang_1', 'caoejiang_2',
          'caojing_1', 'caojing_2',
          'changer_1', 'changer_2', 'changer_3', 'changer_4',
          'changshu_1', 'changshu_2', 'changshu_3',
          'changxing_1', 'changxing_2',
          'fengtai_1', 'fengtai_2', 'fengtai_3', 'fengtai_4',
          'huasu_1', 'huasu_2', 'huasu_3', 'huasu_4',
          'jiaxing_1', 'jiaxing_2',
          'jiaxinger_4', 'jiaxinger_5', 'jiaxinger_6', 'jiaxinger_7', 'jiaxinger_8',
          'lanxi_1', 'lanxi_2', 'lanxi_3', 'lanxi_4',
          'leqing_1', 'leqing_2', 'leqing_3', 'leqing_4',
          'liuhe_7', 'liuhe_8',
          'liuheng_1', 'liuheng_2',
          'longzihu_1', 'longzihu_2',
          'luohe_6',
          'niushan_1', 'niushan_2',
          'pingyu_3', 'pingyu_4', 'pingyu_5', 'pingyu_6',
          'qiangjiao_1', 'qiangjiao_2', 'qiangjiao_3', 'qiangjiao_4',
          'shidongkou_1', 'shidongkou_2',
          'shuzhou_1', 'shuzhou_2',
          'tongling_5',
          'tuanzhou_1', 'tuanzhou_2',
          'waigaoqiaoer_5', 'waigaoqiaoer_6',
          'waigaoqiaosan_7', 'waigaoqiaosan_8',
          'wangting_11', 'wangting_3', 'wangting_4',
          'wushashan_1', 'wushashan_2', 'wushashan_3', 'wushashan_4',
          'xinqiao_1', 'xinqiao_2',
          'yuanzhuang_1', 'yuanzhuang_2', 'yuanzhuang_4',
          'yuhuan_1', 'yuhuan_2', 'yuhuan_3', 'yuhuan_4',
          'zhaoji_1', 'zhaoji_2']


def train(file_path_1='', file_path_2='', save_path='data/factory.csv'):
    data=pd.read_csv(file_path_1)

    if os.path.exists('all_train.csv'):
        pass
        print('the training or validation data set has been exited!')

    elif os.path.exists(file_path_1) and os.path.exists(file_path_2):
        data=pd.read_csv(filepath_or_buffer=file_path_1,encoding='utf-8')

        train_data=data

        print(train_data.values.shape)

        file = open('all_train.csv', 'w', encoding='utf-8')
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
    pass