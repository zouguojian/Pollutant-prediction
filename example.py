# -- coding: utf-8 --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data/all_test.csv').values[:,3:]

plt.figure()
# Label is observed value,Blue
plt.plot(data[:,6], 'b*:', label=u'actual value')
# Predict is predicted value，Red
# use the legend
# plt.legend()
plt.xlabel("time(hours)", fontsize=17)
plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
plt.title("the actual value of pm$_{2.5}", fontsize=17)
plt.show()


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

