'''
@Author: 风满楼
@Date: 2020-04-23 17:11:14
@LastEditTime: 2020-04-25 19:12:03
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
    if not embedding :
        for item in all_input:
            sparse_layer_list.append(Input(shape=(1,), name=item.feat_name))
            dense_layer_list.append(Input(shape=(1,), name=item.feat_name))
    else: 
        for item in all_input:
            if isinstance(item, SparseClass):
                sparse_layer_list.append(
                    Embedding(item.vocablary_size,item.embedding_dim)(Input(shape=(1,), name=item.feat_name)) # (None, 1, 5)
                    )
            elif isinstance(item, DenseClass):
                dense_layer_list.append(
                    RepeatVector(1)(Dense(item.embedding_dim)(Input(shape=(1,), name=item.feat_name)))
                )
    return sparse_layer_list, dense_layer_list