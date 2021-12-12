# -- coding: utf-8 --

import pandas as pd
import numpy as np
import csv
data=pd.read_csv('fgmergedata.csv',encoding='utf-8')


values=data.values
print(values.shape)

keys=list(data.keys())
print(keys)

print(data.values[928670])

site=pd.read_csv('site.csv')
sites=list(site.values[:,0][:25])
print(sites)



# 线性插值
def liner_insert(d1, key_0, key, x, y):
    d = pd.DataFrame()
    d['date'] = x
    d['val'] = y
    d['date'] = pd.to_datetime(d['date'])
    helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max(), freq='H')})

    d = pd.merge(d, helper, on='date', how='outer').sort_values('date')
    d['val'] = d['val'].interpolate(method='linear')

    if key_0 not in d1:
        d1[key_0]=d.values[:,0]
    d1[key]=d.values[:,1]
    # print(d['val'].values.shape)
    # helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max(), freq='H')})

def read_taget_sites():
    # 生成 n 个数据集
    d_dict={site:pd.DataFrame() for site in sites}
    new_sites=[]
    for site in sites:
        print(site)
        data1=data.loc[data['factory']==site]
        for key in keys[3:]:
            x,y=[],[]
            for i in range(data1.values.shape[0]):
                if np.isnan(data1[key].values[i]) or data1[key].values[i] is None:
                    print('continue account!!!!')
                    continue
                else:
                    x.append(data1.values[:,2][i])
                    y.append(data1[key].values[i])
            liner_insert(d_dict[site],'time',key,x,y)

            print(d_dict[site].values.shape)

        if d_dict[site].values.shape[0] != 10248:
            print('delete')
            del d_dict[site]
        else:new_sites.append(site)


    file = open('train.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys[1:])
    print(new_sites)
    for i in range(d_dict[new_sites[0]].values.shape[0]):
        for site in new_sites:
            writer.writerow([site] + list(d_dict[site].values[i]))
    file.close()
            # d_dict[site].to_csv(path_or_buf=site+'.csv',encoding='utf-8')

print('!!!___________________________begain_______________________________!!!')
read_taget_sites()
print('!!!___________________________finish_______________________________!!!')
