# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:41:53 2017

@author: kong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import tensorflow as tf

#定义常量
layer_num=2 #隐层数量
rnn_unit=36      #hidden layer units
input_size=9
output_size=1
lr=0.00001    #学习率
time_step_ini=8
train_num=1240

#——————————————————导入训练数据——————————————————————
sk_kj850T=pd.read_csv("sk_kj.csv")     #读入数据

sk_kj850T['2m温度'] = sk_kj850T['2m温度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['2m温度'] = sk_kj850T['2m温度'].interpolate()

sk_kj850T['850hpa温度'] = sk_kj850T['850hpa温度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['850hpa温度'] = sk_kj850T['850hpa温度'].interpolate()

sk_kj850T['850hpa湿度'] = sk_kj850T['850hpa湿度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['850hpa湿度'] = sk_kj850T['850hpa湿度'].interpolate()

sk_kj850T['700hpa湿度'] = sk_kj850T['700hpa湿度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['700hpa湿度'] = sk_kj850T['700hpa湿度'].interpolate()

sk_kj850T['850W'] = sk_kj850T['850W'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['850W'] = sk_kj850T['850W'].interpolate()

sk_kj850T['700W'] = sk_kj850T['700W'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['700W'] = sk_kj850T['700W'].interpolate()

sk_kj850T['500hpa涡度'] = sk_kj850T['500hpa涡度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['500hpa涡度'] = sk_kj850T['500hpa涡度'].interpolate()

sk_kj850T['700hpa涡度'] = sk_kj850T['700hpa涡度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['700hpa涡度'] = sk_kj850T['700hpa涡度'].interpolate()

sk_kj850T['850hpa涡度'] = sk_kj850T['850hpa涡度'].replace(9.9990002605500006e+20, np.nan)
sk_kj850T['850hpa涡度'] = sk_kj850T['850hpa涡度'].interpolate()

sk_24_T = sk_kj850T['温度'][:-8]
sk_kj850T = sk_kj850T.iloc[8:].reset_index(drop=True)
sk_kj850T['24温度'] = sk_24_T
sk_kj850T = sk_kj850T.iloc[:1480]
sk_kj850T['az.alt'] = sk_kj850T['az']*sk_kj850T['alt']

predictor = ['24温度','850hpa温度','850hpa湿度','700hpa湿度','850hpa涡度','700hpa涡度','2m温度','alt','az.alt','温度']
# =============================================================================
# predictor = ['上一时次温度','850hpa温度','2m温度','温度']
# =============================================================================
data=sk_kj850T[predictor].values  
time=sk_kj850T['实况日期'].values
time_test=time[train_num:]

#获取训练集
def get_train_data(batch_size=20,time_step=time_step_ini,train_begin=0,train_end=train_num):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:input_size]
       y=normalized_train_data[i:i+time_step,input_size,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(time_step=time_step_ini,test_begin=train_num):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]
       y=normalized_test_data[i*time_step:(i+1)*time_step,input_size]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
    return mean,std,test_x,test_y

mean,std,test_x,test_y = get_test_data()

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X, keep_prob):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) #activation为使用的激活函数
    cell=tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    m_cell=tf.nn.rnn_cell.MultiRNNCell([cell] * layer_num, state_is_tuple=True)
    init_state=m_cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(m_cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm(batch_size=20,time_step=time_step_ini,train_begin=0,train_end=train_num):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm_"):
        pred,_=lstm(X, 0.5)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint('moduleT_paper')   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        loss_saver = []
        loss_set = 0.4
        #重复训练10000次
        for i in range(200):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            loss_saver.append(loss_)
# =============================================================================
#             if loss_set > loss_:
#                 loss_set = loss_
#                 print("保存模型：",saver.save(sess,'moduleT_paper/stock2.model'))
# =============================================================================
            if (i+1) % 100 == 0:
                loss_set = loss_
                #print("保存模型：",saver.save(sess,'moduleT_paper/stock2.model'))
        plt.figure()
        plt.plot(list(range(len(loss_saver))), loss_saver, color='b')
        #plt.show()
        return loss_set


loss_set = train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=time_step_ini):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope("sec_lstm_",reuse=True):
        pred,_=lstm(X, 1)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('moduleT_paper')
        saver.restore(sess, module_file) 
        test_predict=[]
        #每24小时预测8次
        predict=[]
        for step in range(len(test_x)):
# =============================================================================
#           if step%8 != 0 :
#               test_x[step][0][0] = predict[0]
# =============================================================================
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
          
        test_y=np.array(test_y)*std[input_size]+mean[input_size]
        test_predict=np.array(test_predict)*std[input_size]+mean[input_size]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/np.abs(test_y[:len(test_predict)]))  #偏差
        a = np.abs(test_predict - test_y[:len(test_predict)])
        l =float(sum(a<=2))/len(a)
        print("acc:", acc)
        print("l:",l)
# =============================================================================
#         print(test_predict)
#         print(test_y)
# =============================================================================
        #以折线图表示结果
        plt.figure(figsize=(10,5))
        #xs = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date() for d in time_test]
        #plt.xticks(rotation=-45)
# =============================================================================
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
#         plt.gca().xaxis.set_major_locator(mdates.HourLocator())
# =============================================================================
        hour = [(d+1)*3 for d in range(len(time_test))]
        #plt.ylable('Temprot')
        plt.xlabel('Hour')
        plt.ylabel('Temperature')
        l1, = plt.plot(hour, test_predict, 'k-')
        #l1.set_linewidth(2.0)
        l2, = plt.plot(hour, test_y, 'k:')
        l2.set_linewidth(3.0)
        l2.set_alpha(0.6)
        le = plt.legend([l1, l2], ['Pre', 'Real'], loc='upper left')
        plt.gca().add_artist(le)
        #plt.gcf().autofmt_xdate()
        plt.savefig('d:/p.svg',format='svg')
        plt.show()
        return test_predict,test_y
        

p , y = prediction() 