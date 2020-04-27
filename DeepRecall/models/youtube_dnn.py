'''
@Author: 风满楼
@Date: 2020-04-26 20:14:57
@LastEditTime: 2020-04-26 20:52:20
@LastEditors: Please set LastEditors
@Description: model of the YouTube DNN
@FilePath: /frame_sort/DeepRecall/models/youtube_dnn.py
'''

import pandas as pd 


if __name__ == "__main__":
    data_path = "/home/xddz/tools/jupyterlab/yining/Recommand/data_set/"
    unames = ['user_id','gender','age','occupation','zip']
    user = pd.read_csv(data_path+'ml-1m/users.dat',sep='::',header=None,names=unames)
    rnames = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_csv(data_path+'ml-1m/ratings.dat',sep='::',header=None,names=rnames)
    mnames = ['movie_id','title','genres']
    movies = pd.read_csv(data_path+'ml-1m/movies.dat',sep='::',header=None,names=mnames)