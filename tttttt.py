# -- coding: utf-8 --

import pandas as pd
import math
import csv

# data=pd.read_excel('/Users/guojianzou/Documents/博士相关论文/Housing prices/data/2019年归一化_all.xlsx')
#
# print(list(data.keys())[0:4])
# print(data.values.shape)
#
#
# data2=pd.read_excel('/Users/guojianzou/Documents/博士相关论文/Housing prices/data/上海门店202108.xlsx')
#
# print(list(data2.keys()[:5]))
# print(data2.values.shape)
#
# file = open('/Users/guojianzou/Documents/博士相关论文/Housing prices/data/couple.csv', 'w', encoding="utf_8_sig")
# writer = csv.writer(file)
# writer.writerow(['名称', '区域','小区'])
#
# for i in range(data2.values.shape[0]):
#     name2=data2.values[i][0]
#     max_len=100000.0
#
#     name_index=0
#     for j in range(data.values.shape[0]):
#         if math.sqrt((data2.values[i][3]-data.values[j][2])**2+(data2.values[i][4]-data.values[j][3])**2)<max_len:
#             max_len=math.sqrt((data2.values[i][3]-data.values[j][2])**2+(data2.values[i][4]-data.values[j][3])**2)
#             name_index=j
#     writer.writerow([name2, data2.values[i][1], data.values[name_index][1]])
#
# file.close()


import json
import pickle

# x = pickle.load(open("/Users/guojianzou/Documents/博士相关论文/Housing prices/data/text.txt","rb"), encoding='bytes')
# print(x)



import seaborn as sns
import matplotlib.pyplot as plt

# fmt(字符串格式代码，矩阵上标识数字的数据格式，比如保留小数点后几位数字)

import numpy as np

# plt.figure()

x = np.random.randn(4, 4)

f, (ax1, ax2) = plt.subplots( nrows=2)

sns.heatmap(x, annot=False, ax=ax1)

sns.heatmap(np.random.randn(3, 3), annot=True, fmt='.1f', ax=ax2)

plt.show()


