'''
@Author: 风满楼
@Date: 2020-04-22 19:57:31
@LastEditTime: 2020-04-26 19:17:57
@LastEditors: Please set LastEditors
@Description: 实现FM模型
@FilePath: /eyepetizer_recommends/recommends/frame_sort/models/fm.py
'''
import sys
sys.path.append('../')
import pandas as pd 
import numpy as np 
# import tensorflow as tf 
# tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras import backend as K 
from tensorflow.keras import Model
from layers.one_order_layer import OneOrder
from layers.two_order_layer import TwoOrder
from layers.LR import Combine
from layers.dnn_layer import DeepOrder
from models.input import SparseClass, DenseClass, get_input_layer, get_embedding_layer

def get_deep_fm_model(sparse_features, dense_features, data):
    # get the input class
    sparse_input_column = [SparseClass(feat_name=feat, vocablary_size=data[feat].nunique()) for feat in sparse_features]
    dense_input_column = [DenseClass(feat_name=feat) for feat in dense_features]
    # get the input and embedding layer
    sparse_input_layers, dense_input_layers = get_input_layer(
        sparse_input_column + dense_input_column, 
        )
    sparse_embedding_layers, dense_embedding_layers = get_embedding_layer(
        sparse_input_layers + dense_input_layers,
        sparse_input_column + dense_input_column,
    )
    # build the model
    y_one_order = OneOrder(sparse_input_column)([sparse_input_layers, dense_input_layers])
    y_two_order = TwoOrder()([sparse_embedding_layers, dense_embedding_layers])
    y_dnn_order = DeepOrder(2,128)([sparse_embedding_layers, dense_embedding_layers]) # the dnn has two layer and each layer has 128 neonuals
    y_output = Combine()([y_one_order, y_two_order, y_dnn_order])
    model = Model(inputs = [sparse_input_layers, dense_input_layers], outputs=[y_output])
    return model




