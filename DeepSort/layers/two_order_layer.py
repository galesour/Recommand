'''
@Author: 风满楼
@Date: 2020-04-22 20:01:09
@LastEditTime: 2020-04-26 11:58:08
@LastEditors: Please set LastEditors
@Description: 实现FM算法的二阶项部分
@FilePath: /eyepetizer_recommends/recommends/frame_sort/layers/tow_order_layer.py
'''
# this layers maybe have problems

from tensorflow.keras.layers import Layer, Concatenate, Multiply, Subtract, Lambda
from tensorflow.keras import backend as K 

class TwoOrder(Layer):
    def __init__(self, **kwargs):
        super(TwoOrder, self).__init__(**kwargs)

    def call(self, input):
        sparse_embedding_input, dense_embedding_input = input
        all_fields = Concatenate(axis=1)(sparse_embedding_input + dense_embedding_input) # None * 39 * 10   39:total of all fields 
        sum_square = K.sum(all_fields, axis=1) # None * 10
        sum_square = Multiply()([sum_square, sum_square]) # Q1. how to user Multiply layers
        square_sum = Multiply()([all_fields, all_fields]) # None * 39 * 10 
        square_sum = K.sum(all_fields, axis=1) # None * 10 
        res = Subtract()([sum_square, square_sum]) # None * 10
        res = K.reshape(K.sum(res, axis=1), (-1,1))
        output = Lambda(lambda x: x*0.5)(res)
        return output
        
    def compute_output_shape(self, input_shape):
        return (None, 1)