'''
@Author: 风满楼
@Date: 2020-04-25 21:51:14
@LastEditTime: 2020-04-25 22:33:40
@LastEditors: Please set LastEditors
@Description: 实现DeepFM中的deep部分
@FilePath: /frame_sort/layers/dnn_layer.py
'''

from tensorflow.keras.layers import Layer, Concatenate, Flatten, Dense

class DeepOrder(Layer):

    def __init__(self, height, width):
        self.height = height # 深度部分的网络层数
        self.width = width # 深度部分中每层网络中神经元的个数

    def call(self, input):
        print('高阶项计算....')
        sparse_embedding_input, dense_embedding_input = input
        all_fields = Concatenate(axis=1)(sparse_embedding_input + dense_embedding_input) # None * 39 * 5
        print(all_fields.shape)
        dnn_input = Flatten()(all_fields) # None * (5 * 39)
        print(dnn_input.shape)
        output = 0
        for i in self.height:
            dnn_input = Dense(self.width)(dnn_input)
        dnn_output = Dense(1)(dnn_input)
        print(dnn_output.shape)
        return dnn_output

    def compute_output_shape(self, input_shape):
        return (None, 1)