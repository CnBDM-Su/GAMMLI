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


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

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

def cold_index(train_x, test_x, feat_dict):
    train = pd.DataFrame(train_x[:,-2:])
    test = pd.DataFrame(test_x[:,-2:])
    cold_num = []
    for col in train.columns:
        num = 0
        tr = pd.DataFrame(train.iloc[:,col])
        te = pd.DataFrame(test.iloc[:,col])

        tr = tr.drop_duplicates()
        te = te.drop_duplicates()

        one = np.ones(tr.shape)
        tr = pd.concat([tr,pd.DataFrame(one,columns=['signal'])],1)
        re = pd.merge(te,tr,how='left')
        re = re[re.iloc[:,1].isna()]

        for i in re.iloc[:,0]:
            feat_dict[col][i]='cold'
            num += 1
        cold_num.append(num)
    print('cold start user:',cold_num[0])
    print('cold start item:',cold_num[1])
    
    return feat_dict
    

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


def data_initialize(data,test,meta_info_o,task_type):

    #total = pd.concat([data,test],0)
    #meta_info = create_meta_info(total)
    #xx_to, y_to, meta_info_to = load_meta_info(total,meta_info,task_type)
    data = reduce_mem_usage(data)
    test = reduce_mem_usage(test)
    xx, y, meta_info = load_meta_info(data,meta_info_o,task_type)
    xx_t , y_t, meta_info_t = load_meta_info(test,meta_info_o,task_type)
    feat_dict, feat_dim, ui_shape= data_index(xx)
    Xi = data_index_get(xx,feat_dict)
    feat_dict = cold_index(xx,xx_t,feat_dict)
    Xi_t = data_index_get(xx_t,feat_dict)
    Xi[:,1]=Xi[:,1]-ui_shape[0]
    for i in range(Xi_t.shape[0]):
        if Xi_t[i,1] != 'cold':
            Xi_t[i,1] = int(Xi_t[i,1])-ui_shape[0]
    xx = xx[:,:-2]
    xx_t = xx_t[:,:-2]
    meta_info.pop('user_id')
    meta_info.pop('item_id')

    model_info = {}
    model_info['task_type'] = task_type
    model_info['feat_dict'] = feat_dict
    model_info['ui_shape'] = ui_shape


    return xx, Xi, y, xx_t, Xi_t, y_t, meta_info, model_info

