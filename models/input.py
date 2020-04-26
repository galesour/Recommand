'''
@Author: 风满楼
@Date: 2020-04-23 17:11:14
@LastEditTime: 2020-04-26 16:42:41
@LastEditors: Please set LastEditors
@Description: 处理输入的类和功能函数
@FilePath: /frame_sort/models/input.py
'''

from tensorflow.keras.layers import Input, Embedding, Dense, RepeatVector

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

def get_input_layer(all_input, embedding=False):
    sparse_layer_list = []
    dense_layer_list = []
    for item in all_input:
        if isinstance(item, SparseClass):
            sparse_layer_list.append(Input(shape=(1,), name=item.feat_name))
        elif isinstance(ite, DenseClass):
            dense_layer_list.append(Input(shape=(1,), name=item.feat_name))
    return sparse_layer_list, dense_layer_list

def get_embedding_layer(all_input_layers, all_input_column):
    sparse_embedding_layers = []
    dense_embedding_layers = []
    for item in all_input_layers: 
        for item_class in all_input_column:
            if isinstance(item_class, SparseClass):
                if item.name.split(":")[0] == item_class.feat_name:
                    sparse_embedding_layers.append(Embedding(item_class.vocablary_size,item_class.embedding_dim)(item))
            if isinstance(item_class, DenseClass):
                if item.name.split(":")[0] == item_class.feat_name:
                    dense_embedding_layers.append(
                        RepeatVector(1)(Dense(item.embedding_dim)(Input(shape=(1,), name=item.feat_name)))
                    )
    return sparse_embedding_layers, dense_embedding_layers