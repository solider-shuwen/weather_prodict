# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:26:40 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

path_sk_1 = r"E:\python3.6\test\sk\54511.csv"
df=pd.read_csv(path_sk_1, encoding = "gbk")

df['date'] = df['YEAR'].map(str)+'/'+df['MONTH'].map(str)+'/'+df['DAY'].map(str)+' '+df['HOUR'].map(str)+':'+df['MINUTE'].map(str)+':00'
data = df.loc[:, ['date', 'AT', 'W1', 'WW', 'TD', 'RAIN06']]
data.rename(columns = {'date':'实况日期','AT':'气温',
                       'W1':'过去天气1',
                       'WW':'天气现象',
                       'TD':'露点温度',
                       'RAIN06':'6小时降水量'}, inplace=True)
data['相对湿度'] = np.exp(17.67*data['露点温度']/(data['露点温度']+243.5))/np.exp(17.67*data['气温']/(data['气温']+243.5))
data['实况日期'] = pd.to_datetime(data['实况日期'])
# =============================================================================
# for i in range(len(data)):
#     if data.loc[i, '过去天气1'] == 9 or (data.loc[i, '天气现象'] in [13, 17, 29, 91, 92, 93, 94, 95, 96 ,97, 98, 99]):
#         data.loc[i, '雷暴'] = 1
#     else:
#         data.loc[i, '雷暴'] = 0
# =============================================================================
data['6小时降水量'] = data['6小时降水量'].fillna(0)
data57028 = data.sort_values(["实况日期"],ascending=True).reset_index(drop=True)

path_ec = 'ec_T_R54511.csv'
ec = pd.read_csv(path_ec)
ec['实况日期'] = pd.to_datetime(ec['实况日期'])
sk_ec = ec.merge(data57028)
temp = sk_ec['6小时降水量'].drop([0]).reset_index(drop=True)
sk_ec['未来6小时降水量'] = temp

# =============================================================================
# final = pd.DataFrame()
# for i in range(len(sk_ec)):
#     if sk_ec.loc[i, "实况日期"].month in [4, 5, 6, 7, 8, 9, 10] and sk_ec.loc[i, "实况日期"].year not in [2014, 2015]:
#         final = final.append(sk_ec.loc[i])
# sk_ec = final
# 
# =============================================================================
#——————————————————将暴雨与平常天气数量保持一至(训练数据)——————————————————————
# =============================================================================
# strong_index = np.array(sk_ec[sk_ec['未来雷暴'] == 1].index)
# norml_index = sk_ec[sk_ec['未来雷暴'] == 0].index
# random_normal_indices = np.random.choice(norml_index, len(sk_ec[sk_ec['未来雷暴'] == 1]), replace = False)#以暴雨数量为基本，随机取相同数量的正常天气
# random_normal_indices = np.array(random_normal_indices)
# under_sample_indices = np.concatenate([strong_index,random_normal_indices])#合并随机的正常天气和强天气
# under_sample_indices_all = under_sample_indices
# 
# time = pd.read_csv(r"E:\python3.6\test\np_bj\readme.csv", encoding = "gbk", header=None)
# time = time.reset_index(drop = False)
# time['实况日期'] = pd.to_datetime(time.iloc[:,1])
# 
# under_sample_data = sk_ec.loc[under_sample_indices_all,['实况日期','气温','相对湿度','未来雷暴']].sort_index(ascending=True).reset_index()
# del under_sample_data['index']
# under_sample_data = pd.merge(under_sample_data, time, how='left', on = '实况日期')
# 
# under_sample_indices = np.array([x for x in under_sample_data.loc[under_sample_data.index % 2 == 1,'index']])
# under_sample_data = under_sample_data.iloc[:83*30]
# 
# =============================================================================
time = pd.read_csv(r"E:\python3.6\test\np_bj\readme.csv", encoding = "gbk", header=None)
time = time.reset_index(drop = False)
time['实况日期'] = pd.to_datetime(time.iloc[:,1])
under_sample_data = pd.merge(sk_ec, time, how='left', on = '实况日期')

under_sample_data.to_csv(r"E:\python3.6\test\np_bj\precipitation_data.csv", index=None)
#——————————————————测试数据——————————————————————
# =============================================================================
# test = sk_ec.loc[39655:,['实况日期','气温','相对湿度','6小时降水量','未来雷暴']]
# 
# time = pd.read_csv(r"E:\python3.6\test\np_bj\readme.csv", encoding = "gbk", header=None)
# time = time.reset_index(drop = False)
# time['实况日期'] = pd.to_datetime(time.iloc[:,1])
# 
# test = pd.merge(test, time, how='left', on = '实况日期')
# test.to_csv(r"E:\python3.6\test\np_bj\test.csv", index=None)
# 
# =============================================================================
