# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:35:30 2020

@author: suyu
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from collections import OrderedDict


'''
def data_reader(path):
    #train part
    data = pd.read_csv(path+"/train.csv")
    id_u = data.UserID
    id_i = data.MovieID
    y = data.Rating
    x = data.drop(['UserID','MovieID','Rating'],1)
    data = pd.concat([x,id_u,id_i,y],1)
    
    #test part
    test = pd.read_csv(path+"/test.csv")
    id_ut = test.UserID
    id_it = test.MovieID
    y_t = test.Rating
    x_t = test.drop(['UserID','MovieID','Rating'],1)
    test = pd.concat([x_t,id_ut,id_it,y_t],1)    
    return data , test
'''
def data_reader(path):
    data = pd.read_csv(path)
    return data
    
def create_meta_info(data):
    #list1 = data.columns
    meta_info = OrderedDict()
    meta_info['uf_1']={'type': 'continues','source':'user'}
    meta_info['uf_2']={'type': 'continues','source':'user'}
    meta_info['uf_3']={'type': 'continues','source':'user'}
    meta_info['uf_4']={'type': 'continues','source':'user'}
    meta_info['uf_5']={'type': 'continues','source':'user'}
    meta_info['if_1']={'type': 'continues','source':'item'}
    meta_info['if_2']={'type': 'continues','source':'item'}
    meta_info['if_3']={'type': 'continues','source':'item'}
    meta_info['if_4']={'type': 'continues','source':'item'}
    meta_info['if_5']={'type': 'continues','source':'item'}
    #meta_info['Occupation']={"type":"categorical"}
    #meta_info['Genres']={"type":"categorical"}
    #meta_info['Gender']={"type":"categorical"}
    #meta_info['Age'] = {"type":"continues"}
    meta_info['user_id']={"type":"id",'source':'user'}
    meta_info['item_id']={"type":"id",'source':'item'}
    meta_info['target']={"type":"target",'source':''}
    return meta_info

def data_index(train_x):
    df = pd.DataFrame(train_x[:,-2:])
    feat_dict = {}
    ui_shape = []
    tc = 0
    for col in range(train_x[:,-2:].shape[1]):
        us = df[col].unique()
        ui_shape.append(us.shape[0])
        feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
        tc += len(us)
        feat_dim = tc
        
    return feat_dict, feat_dim, ui_shape
        
def data_index_get(train_x, feat_dict):
        
    dfi = pd.DataFrame(train_x[:,-2:])
    for col in dfi.columns:
        dfi[col] = dfi[col].map(feat_dict[col])
    Xi = dfi.values.tolist()
    Xi = np.array(Xi)
    
    return Xi

def load_credit_default(data,meta_info,path="./data/",  rand_seed=0):
    #data = pd.read_csv(path + "movie_lens/train.csv", header=1)
    
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    lv_meta_info = {}
    xx = np.zeros(x.shape)
    task_type = "Regression"
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            #enc = OrdinalEncoder()
            #enc.fit(y)
            #y = enc.transform(y)
            #sx = MinMaxScaler((0, 1))
            #y = sx.fit_transform(y)
            continue
            #meta_info[key]["scaler"] = sx
            #meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "categorical":
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]["values"] = enc.categories_[0].tolist()
        elif item['type'] == "id":
            xx[:,[i]] = x[:,[i]]
            lv_meta_info[key] = item
        else:
            sx = MinMaxScaler((0, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    #train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    #return train_x, test_x, train_y, test_y, task_type, meta_info
    return xx.astype(np.float32), y, task_type, meta_info

