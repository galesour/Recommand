'''
@Author: 风满楼
@Date: 2020-04-22 19:57:31
@LastEditTime: 2020-04-23 18:47:55
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
from sklearn import preprocessing
from input import SparseClass, DenseClass, get_input_layer

if __name__ == "__main__":
    # 前期的数据处理
    data = pd.read_csv('../data_set/criteo_sample.txt')[0:20] # 读取样例数据
    labels = data['label']
    sparse_inputs = ['I{}'.format(i) for i in range(1,14)] # 连续特征
    dense_inputs = ['C{}'.format(i) for i in range(1,27)] # 离散特征
    data[sparse_inputs] = data[sparse_inputs].fillna('-1', ) # 对数据中缺失值的处理
    data[dense_inputs] = data[dense_inputs].fillna(0, ) # 对数据中缺失值的处理
    for feat in sparse_inputs: 
        print(feat)
        lbe = preprocessing.labelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) # 对离散数据labelencoder编码
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    sparse_input_column = [SparseClass(feat_name=feat, voca_size=data[feat].unique()) for feat in sparse_inputs]
    dense_input_column = [DenseClass(feat_name=feat) for feat in dense_inputs]
    sparse_input_layers, dense_input_layers = get_input_layer(sparse_input_column + dense_input_column)
    print(sparse_input_layers)
    print(dense_input_layers)
    # 模型的构建
    y_one_order = OneOrder()([sparse_input_layers, dense_input_layers])
    model = Model(inputs=[sparse_input_layers, dense_input_layers], outputs=y_one_order)
    model.summary()
