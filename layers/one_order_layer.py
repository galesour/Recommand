'''
@Author: 风满楼
@Date: 2020-04-22 20:00:48
@LastEditTime: 2020-04-23 23:00:25
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /eyepetizer_recommends/recommends/frame_sort/layers/one_order.py
'''
from keras.engine.topology import Layer
from keras.layers import Embedding, Input
from keras import backend as K 

class OneOrder(Layer):
    '''
    @description: 计算FM中的一阶项
    @param {type} 
    @return: 
    '''    

    def __init__(self, **kwargs):
        super(OneOrder, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.n_sparse = len(input_shape[0])
        self.n_dense = len(input_shape[1])
        self.dense_weights = self.add_weight(
            shape=(self.n_dense,1),
            initializer='glorot_uniform',
            trainable=True,
            name='dense_one_order_weight'
        )
        super(OneOrder, self).build(input_shape)

    def call(self, inputs):
        print(inputs)
        sparse_inputs, dense_inputs = inputs
        output = 0
        for sparse_input in sparse_inputs:
            output += Embedding(sparse_input.shape[-1], 1)(sparse_input)
        dense_inputs = K.concatenate(dense_inputs)
        output += K.dot(dense_inputs, self.dense_weights)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0][0], 1)
    
        