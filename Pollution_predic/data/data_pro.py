# -- coding: utf-8 --
import pandas as pd
import numpy as np

def generateTrainingSet(df=None):
    if df is None: return None
    else:
        df.rename(columns={'year': 'Year',
                           'month': 'Month',
                           'day': "Day",
                           'hour': 'Hour'}, inplace=True)
        df['PM2.5'].interpolate(inplace=True, )
        # df['PM10'].interpolate(inplace=True)
        # df['SO2'].interpolate(inplace=True)
        # df['NO2'].interpolate(inplace=True, method='polynomial', order=2)
        # df['CO'].interpolate(inplace=True)
        # df['O3'].interpolate(inplace=True)
        # df['TEMP'].interpolate(inplace=True)
        # df['PRES'].interpolate(inplace=True)
        # df['DEWP'].interpolate(inplace=True)
        # df['RAIN'].interpolate(inplace=True)
        # df['WSPM'].interpolate(inplace=True)
        # print(df.isnull().sum())
    return df

import os
datasetdf, datasets = [], []
for dirname, _, filenames in os.walk('archive'):
    for filename in filenames:
        fileAddress = os.path.join(dirname, filename)
        df = generateTrainingSet(df= pd.read_csv(fileAddress, encoding='utf-8')[['year','month','day','hour','PM2.5']])
        datasetdf.append(df.values)
        del df

for i in range(datasetdf[-1].shape[0]):
    sample = list()
    for j in range(12):
        sample.append([j] + list(datasetdf[j][i]))
    datasets.append(sample)
    del sample

datasets = np.array(datasets, dtype=np.float64)
print(datasets.shape)
np.savez('train.npz', data=datasets)

df = np.load('train.npz', allow_pickle=True)
print(df['data'][0])

# datasets = np.reshape(np.array(datasets),[-1,6])
# datasets = pd.DataFrame(datasets,columns=['Station','Year','Month','Day','Hour','PM2.5'], dtype=np.float64)
# datasets.to_csv('train.csv',index=False)
# print(datasets.shape)
# print(datasets.isnull().sum())
# datasets.describe().to_csv('describe_output.csv')