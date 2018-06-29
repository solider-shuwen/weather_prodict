# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:10:55 2018

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

img_size = 17
num_channels = 28
batch_size = 16
num_classes = 2

#载入数据
test = pd.read_csv(r'E:\python3.6\test\np_bj\test.csv', encoding='gbk')
path = "E:/python3.6/test/np_bj/"
index_now = [path + str(x) + '.npy' for x in test['index']]
index_last = [path + str(x) + '.npy' for x in test['index']-1]

def load_test(now_paths, last_paths):
    images_now = []
    images_last = []

    #print('Going to read images')
    for path in now_paths:   
        image = np.load(path).reshape(-1)
        images_now.append(list(image))
    images_now = np.array(images_now).reshape((-1, img_size, img_size, num_channels))
    
    for path in last_paths:   
        image = np.load(path).reshape(-1)
        images_last.append(list(image))
    images_last = np.array(images_last).reshape((-1, img_size, img_size, num_channels))
    
    
    return images_now, images_last

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./strong_clstm/model.ckpt-449.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, './strong_clstm/model.ckpt-449')

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
pred_sig = graph.get_tensor_by_name("sec_lstm_/pred_sig:0")

## Let's feed the images to the input placeholders
pic_now= graph.get_tensor_by_name("pic_now:0") 
pic_last= graph.get_tensor_by_name("pic_last:0") 
keep_prob= graph.get_tensor_by_name("keep_prob:0") 
training= graph.get_tensor_by_name("training:0") 

pres = []
for step in range((len(test)-1)//batch_size):
    now_paths = index_now[step*batch_size : (step+1)*batch_size]
    last_paths = index_last[step*batch_size : (step+1)*batch_size]
    
    images_now, images_last = load_test(now_paths, last_paths)

    pre=sess.run(pred_sig, feed_dict={pic_now:images_now
                                                ,pic_last:images_last
                                                
                                                ,keep_prob: 1
                                                ,training : False})
    print(step)
    
    pres.extend([x[0] for x in pre])
#------------------------------------------------------------------------------
test['x'] = pd.Series(pres)
xx = test.loc[:3343, ['x','未来雷暴']]
for i in range(len(xx)):
    if xx.loc[i, 'x'] > 0.5:
        xx.loc[i, 'x'] = 1
    else:
        xx.loc[i, 'x'] = 0
        
mat =  confusion_matrix(xx['x'], xx['未来雷暴'])
print(mat[1,1]/(mat[1,1]+mat[0,1]))
print(mat[1,0]/(mat[1,0]+mat[0,0]))