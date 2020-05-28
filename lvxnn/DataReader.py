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

def load_meta_info(data,meta_info,task_type,path="./data/",  rand_seed=0):
    #data = pd.read_csv(path + "movie_lens/train.csv", header=1)

    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    xx = np.zeros(x.shape)
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            if task_type == 'Regression':
                continue
            elif task_type == 'Classification':
                
                enc = OrdinalEncoder()
                enc.fit(y)
                y = enc.transform(y)
            #sx = MinMaxScaler((0, 1))
            #y = sx.fit_transform(y)
            
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
        else:
            sx = MinMaxScaler((0, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    #train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=test_ratio, random_state=rand_seed)
    #return train_x, test_x, train_y, test_y, task_type, meta_info
    return xx.astype(np.float32), y, meta_info


def data_initialize(data,test,meta_info,task_type):

    total = pd.concat([data,test],0)
    #meta_info = create_meta_info(total)
    xx_to, y_to, meta_info_to = load_meta_info(total,meta_info,task_type)
    xx, y, meta_info = load_meta_info(data,meta_info,task_type)
    xx_t , y_t, meta_info_t = load_meta_info(test,meta_info,task_type)
    feat_dict, feat_dim, ui_shape= data_index(xx_to)
    Xi = data_index_get(xx,feat_dict)
    Xi_t = data_index_get(xx_t,feat_dict)
    Xi[:,1]=Xi[:,1]-ui_shape[0]
    Xi_t[:,1] = Xi_t[:,1]-ui_shape[0]
    xx = xx[:,:-2]
    xx_t = xx_t[:,:-2]
    meta_info.pop('user_id')
    meta_info.pop('item_id')

    model_info = {}
    model_info['task_type'] = task_type
    model_info['feat_dict'] = feat_dict
    model_info['ui_shape'] = ui_shape


    return xx, Xi, y, xx_t, Xi_t, y_t, meta_info, model_info

