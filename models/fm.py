'''
@Author: 风满楼
@Date: 2020-04-22 19:57:31
@LastEditTime: 2020-04-22 21:44:05
@LastEditors: Please set LastEditors
@Description: 实现FM模型
@FilePath: /eyepetizer_recommends/recommends/frame_sort/models/fm.py
'''
import sys
sys.path.append('../')
import pandas as pd 
from keras import Model
from keras.layers import Input
from layers.one_order_layer import OneOrder

if __name__ == "__main__":

    data = pd.read_csv('../data_set/criteo_sample.txt')
    print(data)
    # sparse_inputs = [
    #     Input(shape=(2,), name='userType'),
    #     Input(shape=(2,), name='gender'),
    #     Input(shape=(10,), name='category'),
    # ]
    # dense_inputs = [
    #     Input(shape=(1,), name='collection_count'),
    #     Input(shape=(1,), name='share_count'),
    #     Input(shape=(1,), name='play_count'),
    #     Input(shape=(1,), name='collect_share_count'),
    # ]
    # y_one_order = OneOrder()([sparse_inputs, dense_inputs])
    # model = Model(inputs=[sparse_inputs, dense_inputs], outputs = y_one_order)
    # model.summary()
    
    
