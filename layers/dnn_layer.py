'''
@Author: 风满楼
@Date: 2020-04-25 21:51:14
@LastEditTime: 2020-04-26 11:58:56
@LastEditors: Please set LastEditors
@Description: 实现DeepFM中的deep部分
@FilePath: /frame_sort/layers/dnn_layer.py
'''

from tensorflow.keras.layers import Layer, Concatenate, Flatten, Dense

class DeepOrder(Layer):
    
    def __init__(self, height, width, **kwargs):
        self.height = height
        self.width = width
        super(DeepOrder, self).__init__(**kwargs)

    def call(self, input):
        sparse_embedding_input, dense_embedding_input = input
        all_fields = Concatenate(axis=1)(sparse_embedding_input + dense_embedding_input) # None * 39 * 5
        dnn_input = Flatten()(all_fields) # None * (5 * 39)
        output = 0
        for i in range(self.height):
            dnn_input = Dense(self.width)(dnn_input)
        dnn_output = Dense(1)(dnn_input)
        return dnn_output

    def compute_output_shape(self, input_shape):
        return (None, 1)