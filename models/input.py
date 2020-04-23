'''
@Author: your name
@Date: 2020-04-23 17:11:14
@LastEditTime: 2020-04-23 20:02:44
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /frame_sort/models/input.py
'''

from tensorflow.keras.layers import Input

class SparseClass():
    '''
    @description: 处理离散数据的类
    @param {
        feat_name: 特征的名字
        vocablary_size: 特征中不重复的个数
    } 
    @return: 
    '''
    def __init__(self, feat_name, vocablary_size):
        self.feat_name = feat_name
        self.vocablary_size = vocablary_size

class DenseClass():
    '''
    @description: 处理连续数据的类
    @param {
        feat_name: 特征的名字
    } 
    @return: 
    '''     
    def __init__(self, feat_name):
        self.feat_name = feat_name

def get_input_layer(all_input):
    sparse_input_layer_list = []
    dense_input_layer_list = []
    for item in all_input:
        if isinstance(item, SparseClass):
            input_shape = (item.vocablary_size,)
            sparse_input_layer_list.append(Input((shape=input_shape), name=item.feat_name))
        elif isinsatnce(item, DenseClass):
            dense_input_layer_list.append(Input((shape=(1,), name=item.feat_name)))
    return sparse_input_layer_list, dense_input_layer_list

