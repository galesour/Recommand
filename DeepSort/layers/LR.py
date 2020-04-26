'''
@Author: 风满楼
@Date: 2020-04-26 12:00:53
@LastEditTime: 2020-04-26 15:35:24
@LastEditors: Please set LastEditors
@Description: DeepFM最后的组合层,将一阶, 二阶, 高阶组合在一起
@FilePath: /frame_sort/layers/LR.py
'''
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import backend as K 
class Combine(Layer):

    def __init__(self, **kwargs):
        super(Combine, self).__init__(**kwargs)
    
    def call(self, input):
        dense_input = K.concatenate(input, axis=1)
        output = Dense(1)(dense_input)
        print(output.shape)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)