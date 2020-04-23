'''
@Author: 风满楼
@Date: 2020-04-22 19:57:31
@LastEditTime: 2020-04-23 20:46:59
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from input import SparseClass, DenseClass, get_input_layer

if __name__ == "__main__":
    # 前期的数据处理
    data = pd.read_csv('../data_set/criteo_sample.txt')[0:20] # 读取样例数据
    labels = data['label']
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) # standardization of the sparse feature
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features]) # Normalized of the dense feature

    # 2.count #unique features for each sparse field,and record dense feature field name
    sparse_input_column = [SparseClass(feat_name=feat, vocablary_size=data[feat].nunique()) for feat in sparse_features]
    dense_input_column = [DenseClass(feat_name=feat) for feat in dense_features]
    sparse_input_layers, dense_input_layers = get_input_layer(sparse_input_column + dense_input_column)
    # sparse_input_layers is the collection of Input layer
    # building the model 
    y_one_order = OneOrder()([sparse_input_layers, dense_input_layers])
    model = Model(inputs=[sparse_input_layers, dense_input_layers], outputs=y_one_order)
    model.summary()
