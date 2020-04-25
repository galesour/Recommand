'''
@Author: 风满楼
@Date: 2020-04-22 20:01:09
@LastEditTime: 2020-04-25 17:09:38
@LastEditors: Please set LastEditors
@Description: 实现FM算法的二阶项部分
@FilePath: /eyepetizer_recommends/recommends/frame_sort/layers/tow_order_layer.py
'''
from tensorflow.keras.layers import Layer

class TwoOrder(Layer):
    def __init__(self, **kwargs):
        super(TwoOrder, self).__init__(**kwargs)

    def build(self, input_shape):
        pass 
        super(TwoOrder, self).build(input_shape)

    def call(self, input):
        
        pass 

    def compute_output_shape(self, input_shape):
        pass