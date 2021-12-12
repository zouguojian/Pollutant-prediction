# -- coding: utf-8 --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# col=['factorygroup', 'datehour', 'YGGL', 'YQLL', 'YCRD', 'EYHL', 'DYHW']
#
# data=pd.read_csv('data/factorymergedata.csv',usecols=col)
#
#
#
# print(data.keys())
# print(data.values[-2])
#
#
# statics=dict()
# for value in data['factorygroup'].values:
#     if value not in statics:
#         statics[value]=1
#     else:statics[value]+=1
# print(statics)
#
#
#
# statics=list()
# for value in data['factorygroup'].values:
#     if value not in statics:
#         statics.append(value)
#     else:pass
# print(sorted(statics),len(statics))




def train(file='', save_path='data/factory.csv', begin='2021-04-00', end='2021-07-17'):
    pass





# plt.figure()
# # Label is observed value,Blue
# plt.plot(data.loc[data['station']=='PDzhangjiang'].values[:,3], 'b*:', label=u'PDzhangjiang')
# plt.plot(data.loc[data['station']=='Qingpu'].values[:,3], 'r*:', label=u'Qingpu')
# # Predict is predicted value，Red
# # use the legend
# plt.legend()
# plt.xlabel("time(hours)", fontsize=17)
# plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
# plt.title("the actual value of pm$_{2.5}$", fontsize=17)
# plt.show()


def describe(label, predict, prediction_size):
    '''
    :param label:
    :param predict:
    :param prediction_size:
    :return:
    '''
    plt.figure()
    # Label is observed value,Blue
    plt.plot(label[0:prediction_size], 'b*:', label=u'actual value')
    # Predict is predicted value，Red
    plt.plot(predict[0:prediction_size], 'r*:', label=u'predicted value')
    # use the legend
    # plt.legend()
    plt.xlabel("time(hours)", fontsize=17)
    plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
    plt.title("the prediction of pm$_{2.5}", fontsize=17)
    plt.show()

import numpy as np
import scipy.sparse as sp

a = np.array([[1,0,0,1,0],[0,1,1,0,0],[0,0,0,0,1],[1,0,0,0,1],[0,0,1,0,0]])
a = sp.coo_matrix(a)
print(a)

rowsum=np.array(a.sum(axis=1))
print(rowsum)

rowpow=np.power(rowsum, -0.5)
print(rowpow)

rowflatten=rowpow.flatten()
print(rowflatten)

def normalize_adj(adj):
    '''
    :param adj: Symmetrically normalize adjacency matrix
    :return:
    '''
    adj = sp.coo_matrix(adj) # 转化为稀疏矩阵表示的形式
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()