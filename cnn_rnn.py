# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:05:15 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

#载入index数据与实况数据
#under_sample_indix = pd.read_csv(r'E:\python3.6\test\np_bj\under_sample_indix.csv', encoding='gbk')
under_sample_data = pd.read_csv(r'E:\python3.6\test\np_bj\under_sample_data.csv', encoding='gbk')
# =============================================================================
# under_sample_data['气温'] = (under_sample_data['气温']-np.min(under_sample_data['气温']))/(np.max(under_sample_data['气温']) - np.min(under_sample_data['气温']))
# under_sample_data['相对湿度'] = (under_sample_data['相对湿度']-np.min(under_sample_data['相对湿度']))/(np.max(under_sample_data['相对湿度']) - np.min(under_sample_data['相对湿度']))
# under_sample_data = under_sample_data.fillna(under_sample_data.mean())
# 
# =============================================================================
# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 17
num_channels = 28
batch_size = 16
num_classes = 2

layer_num=1 #隐层数量
rnn_unit=1024      #hidden layer units
output_size=2
lr=0.0001    #学习率
time_step_ini=2
train_num=2500

pic_now = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='pic_now')
pic_last = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='pic_last')
# =============================================================================
# T_now = tf.placeholder(tf.float32, shape=[None], name='T_now')
# T_last = tf.placeholder(tf.float32, shape=[None], name='T_last')
# R_now = tf.placeholder(tf.float32, shape=[None], name='R_now')
# R_last = tf.placeholder(tf.float32, shape=[None], name='R_last')
# 
# =============================================================================
def load_train(now_paths, last_paths, labels):
    images_now = []
    images_last = []
    labels = labels

    #print('Going to read images')
    for path in now_paths:   
        image = np.load(path).reshape(-1)
        images_now.append(list(image))
    images_now = np.array(images_now).reshape((-1, img_size, img_size, num_channels))
    
    for path in last_paths:   
        image = np.load(path).reshape(-1)
        images_last.append(list(image))
    images_last = np.array(images_last).reshape((-1, img_size, img_size, num_channels))
    
    labels = np.array([[x,1-x] for x in labels])
    
    return images_now, images_last, labels

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
#train or test
training = tf.placeholder(tf.bool, name='training')
#定义cnn网络
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters,
               training):  
    
    ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases
    #layer = tf.layers.batch_normalization(layer, axis=3, training=training)
    layer = tf.nn.relu(layer)
    
    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    #layer = tf.nn.relu(layer)

    return layer

    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, 1, num_features])
    return layer

def cnn_model(pic_now, pic_last):
    ##Network graph params
    filter_size_conv1 = 3 
    num_filters_conv1 = 32
    
    filter_size_conv2 = 3
    num_filters_conv2 = 64
    
    filter_size_conv3 = 3
    num_filters_conv3 = 128
    
    #现在
    layer_conv1_now = create_convolutional_layer(input=pic_now,
                   num_input_channels=num_channels,
                   conv_filter_size=filter_size_conv1,
                   num_filters=num_filters_conv1,
                   training = training)
    layer_conv2_now = create_convolutional_layer(input=layer_conv1_now,
                   num_input_channels=num_filters_conv1,
                   conv_filter_size=filter_size_conv2,
                   num_filters=num_filters_conv2,
                   training = training)
    
    layer_conv3_now = create_convolutional_layer(input=layer_conv2_now,
                   num_input_channels=num_filters_conv2,
                   conv_filter_size=filter_size_conv3,
                   num_filters=num_filters_conv3,
                   training = training)
              
    layer_flat_now = create_flatten_layer(layer_conv3_now)
    #过去
    layer_conv1_last = create_convolutional_layer(input=pic_last,
                   num_input_channels=num_channels,
                   conv_filter_size=filter_size_conv1,
                   num_filters=num_filters_conv1,
                   training = training)
    layer_conv2_last = create_convolutional_layer(input=layer_conv1_last,
                   num_input_channels=num_filters_conv1,
                   conv_filter_size=filter_size_conv2,
                   num_filters=num_filters_conv2,
                   training = training)
    
    layer_conv3_last = create_convolutional_layer(input=layer_conv2_last,
                   num_input_channels=num_filters_conv2,
                   conv_filter_size=filter_size_conv3,
                   num_filters=num_filters_conv3,
                   training = training)
              
    layer_flat_last = create_flatten_layer(layer_conv3_last)
    
    layer = tf.concat([layer_flat_last, layer_flat_now], 1)
    print(layer.shape)
    
    return layer

#——————————————————定义rnn——————————————————
def lstm_model(layer):    
    time_step=layer.get_shape().as_list()[1]
    input_size=layer.get_shape().as_list()[2]#5186
    w_in=tf.Variable(tf.random_normal([input_size,rnn_unit]))
    b_in=tf.Variable(tf.constant(0.1,shape=[rnn_unit,]))
    input=tf.reshape(layer,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) #activation为使用的激活函数
    cell=tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    m_cell=tf.nn.rnn_cell.MultiRNNCell([cell] * layer_num, state_is_tuple=True)
    init_state=m_cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(m_cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn = tf.transpose(output_rnn, [1, 0, 2])
    #取最终的结果值
    print(output_rnn.get_shape())
    output = tf.gather(output_rnn, int(output_rnn.get_shape()[0]) - 1)
    #output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=tf.Variable(tf.random_normal([rnn_unit,num_classes]))
    b_out=tf.Variable(tf.constant(0.1,shape=[num_classes,]))
    pred=tf.matmul(output,w_out)+b_out
    #pred_sig=tf.clip_by_value(tf.nn.sigmoid(pred), 1e-10, 0.999999)
    pred_sig=tf.nn.softmax(pred, name= 'pred_sig')
    return pred,pred_sig#,final_states

def cost(y_, labels):
    with tf.name_scope('loss'):
        # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=labels)
    cross_entropy_cost = tf.reduce_mean(cross_entropy)
    return cross_entropy_cost

#——————————————————训练模型——————————————————
def train(now_paths_all, last_paths_all, labels_all, batch_size):
    with tf.Session() as sess:
        with tf.variable_scope("sec_lstm_"):
            layer = cnn_model(pic_now, pic_last)
            pred,_=lstm_model(layer)
        #损失函数
        #loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
        loss=cost(pred, y_true)
        train_op=tf.train.AdamOptimizer(lr).minimize(loss)
        saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
        #module_file = tf.train.latest_checkpoint('strong_clstm')   
        
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'strong_clstm/model.ckpt-449')
        tenboard_dir = './tensorboard'
    
        writer = tf.summary.FileWriter(tenboard_dir)
        writer.add_graph(sess.graph)
        
        loss_saver = []
        loss_set = 0.4
        #重复训练10000次
        for i in range(7000):
            for step in range(len(labels_all)//batch_size):
                now_paths = now_paths_all[step*batch_size : (step+1)*batch_size]
                last_paths = last_paths_all[step*batch_size : (step+1)*batch_size]
                labels = labels_all[step*batch_size : (step+1)*batch_size]
                
                images_now, images_last, labels = load_train(now_paths, last_paths, labels)

                sess.run(train_op,feed_dict={pic_now:images_now
                                                            ,pic_last:images_last
                                                            ,keep_prob: 0.5
                                                            ,y_true:labels
                                                            ,training : True})
                loss_=sess.run(loss, feed_dict={pic_now:images_now
                                                            ,pic_last:images_last
                                                            ,keep_prob: 1
                                                            ,y_true:labels
                                                            ,training : False})
                print(i,loss_)
               # print(pred_)
            loss_saver.append(loss_)
# =============================================================================
#             if loss_set > loss_:
#                 loss_set = loss_
#                 print("保存模型：",saver.save(sess,'strong_con/stock2.model'))
# =============================================================================
            if (i+1) % 30 == 0:
                loss_set = loss_
                print("保存模型：",saver.save(sess,'strong_clstm/model.ckpt',global_step=i))
        plt.figure()
        plt.plot(list(range(len(loss_saver))), loss_saver, color='b')
        #plt.show()
        return loss_set

#------------------训练-----------------------------------------------------------
path = "E:/python3.6/test/np_bj/"
index_now = [path + str(x) + '.npy' for x in under_sample_data['index']]
index_last = [path + str(x) + '.npy' for x in (under_sample_data['index'])-1]
lables = list(under_sample_data.loc[:, '未来雷暴'])
print(len(lables), len(index_last))
index_now_train = index_now
index_last_train = index_last
lables_train = lables

loss_set = train(now_paths_all = index_now_train, 
                 last_paths_all = index_last_train,
                 labels_all = lables_train,
                 batch_size = batch_size)

#--------------------测试------------------------------------------------------

