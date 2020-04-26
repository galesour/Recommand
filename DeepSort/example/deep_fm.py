'''
@Author: 风满楼
@Date: 2020-04-26 18:12:47
@LastEditTime: 2020-04-26 19:23:10
@LastEditors: Please set LastEditors
@Description: DeepFM实现DeepFM的例子
@FilePath: /frame_sort/DeepSort/example/deep_fm.py
'''
import sys 
sys.path.append('../')
import pandas as pd 
import numpy as np 
from sklearn.metrics import log_loss, roc_auc_score
from models.deep_fm import get_deep_fm_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

if __name__ == "__main__":
    data = pd.read_csv('/home/xddz/code/eyepetizer_recommends/data/dac/train.txt') # load data
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    targets = data['label'].astype(float).values
    # Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) # standardization of the sparse feature
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features]) # Normalized of the dense feature
    train_data = {} # get the train data
    for item in sparse_features + dense_features:
        train_data[item] = np.array([[val] for val in data[item].values.tolist()])
    model = get_deep_fm_model(sparse_features, dense_features, data)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(
        train_data, 
        targets, 
        batch_size=256,
        epochs=10, 
        validation_split=0.2)
    
    
