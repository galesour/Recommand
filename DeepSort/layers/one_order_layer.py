'''
@Author: 风满楼
@Date: 2020-04-22 20:00:48
@LastEditTime: 2020-04-27 19:33:32
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /eyepetizer_recommends/recommends/frame_sort/layers/one_order.py
'''
from tensorflow.keras.layers import Layer, Embedding, Input
from tensorflow.keras import backend as K 
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

class OneOrder(Layer):
    '''
    @description: 计算FM中的一阶项
    @param {type} 
    @return: 
    '''    

    def __init__(self, sparse_input_column, **kwargs):
        self.unique_list = [item.vocablary_size for item in sparse_input_column]
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
        sparse_inputs, dense_inputs = inputs
        output = 0
        for num, sparse_input in enumerate(sparse_inputs):
            output += Embedding(self.unique_list[num], 1)(sparse_input)[:, 0, :]
        dense_inputs = K.concatenate(dense_inputs)
        tmp = K.dot(dense_inputs, self.dense_weights) # (None, 1)
        output += tmp
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0][0], 1)
    
        