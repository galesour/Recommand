'''
@Author: 风满楼
@Date: 2020-04-23 17:11:14
@LastEditTime: 2020-04-25 15:19:40
@LastEditors: Please set LastEditors
@Description: 处理输入的类和功能函数
@FilePath: /frame_sort/models/input.py
'''

from tensorflow.keras.layers import Input

class SparseClass():
    '''
    @description: 处理离散数据的类
    @param {
        feat_name: 特征的名字
        vocablary_size: 特征中不重复的个数
        embedding_dim: embedding后向量的维度
    } 
    @return: 
    '''
    def __init__(self, feat_name, vocablary_size, embedding_dim=5):
        self.feat_name = feat_name
        self.vocablary_size = vocablary_size
        self.embedding_dim = 5

class DenseClass():
    '''
    @description: 处理连续数据的类
    @param {
        feat_name: 特征的名字
        embedding_dim:embedding后向量的维度
    } 
    @return: 
    '''     
    def __init__(self, feat_name, embedding_dim=5):
        self.feat_name = feat_name
        self.embedding_dim = 5

def get_input_layer(all_input):
    sparse_input_layer_list = []
    dense_input_layer_list = []
    for item in all_input:
        if isinstance(item, SparseClass):
            input_shape = (item.vocablary_size,)
            sparse_input_layer_list.append(Input(input_shape, name=item.feat_name))
        elif isinstance(item, DenseClass):
            dense_input_layer_list.append(Input(shape=(1,), name=item.feat_name))
    return sparse_input_layer_list, dense_input_layer_list

