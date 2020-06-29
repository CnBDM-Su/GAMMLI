# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:36:56 2020

@author: suyu
"""

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
import numpy as np
import pandas as pd

def svd(wc, train, test, train_x, tr_Xi, train_y , test_x , te_Xi, test_y, meta_info, model_info, task_type="Regression", val_ratio=0.2, random_state=0):
    
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    Xi = train.iloc[:,-3:-1].values
    Xi_t = test.iloc[:,-3:-1].values
        
    tr_ratings_dict = {'itemID': Xi[:,1].tolist(),
                'userID': Xi[:,0].tolist(),
                'rating': train_y.ravel().tolist()}

    tr_df = pd.DataFrame(tr_ratings_dict)
    reader = Reader(rating_scale=(train_y.min(), train_y.max()))
        
    tr_data = Dataset.load_from_df(tr_df[['userID', 'itemID', 'rating']], reader)
    tr_data = tr_data.build_full_trainset()
    
    if task_type == "Regression":
        idx1, idx2 = train_test_split(indices, test_size=val_ratio, random_state=random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1
                
        base = SVD(n_factors=5)
        
        cold_mae = []
        cold_rmse = []
        warm_mae = []
        warm_rmse = []
        
        for j in range(10):
            base.fit(tr_data)

            pred = []
        
            for i in range(Xi_t.shape[0]):
                pred.append(base.predict(Xi_t[i,0],Xi_t[i,1],Xi_t[i,0]).est)
    
            pred2 = np.array(pred).ravel()
            
            if wc == 'warm':
            
                warm_y = test_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_pred = pred2[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_mae.append(mean_absolute_error(warm_y,warm_pred))
                warm_rmse.append(mean_squared_error(warm_y,warm_pred)**0.5)
                
            if wc == 'cold':
                
                cold_y = test_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred2[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_mae.append(mean_absolute_error(cold_y,cold_pred))
                cold_rmse.append(mean_squared_error(cold_y,cold_pred)**0.5)

        if wc == 'warm':
            
            i_result = np.array(['SVD',np.mean(warm_mae),np.mean(warm_rmse),np.std(warm_mae),np.std(warm_rmse)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','warm_mae','warm_rmse','std_warm_mae','std_warm_rmse'])

        if wc == 'cold':
            
            i_result = np.array(['SVD',np.mean(cold_mae),np.mean(cold_rmse),np.std(cold_mae),np.std(cold_rmse)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','cold_mae','cold_rmse','std_cold_mae','std_cold_rmse'])
            

        return result
    
    elif task_type == "Classification":
        idx1, idx2 = train_test_split(indices, test_size=val_ratio, stratify=train_y, random_state=random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1

        base = SVD(n_factors=5)
        
        cold_auc = []
        cold_logloss = []
        warm_auc = []
        warm_logloss = []
        
        for j in range(10):
            base.fit(tr_data)

            pred = []
        
            for i in range(Xi_t.shape[0]):
                pred.append(base.predict(Xi_t[i,0],Xi_t[i,1],Xi_t[i,0]).est)
    
            pred2 = np.array(pred).ravel()
            
            if wc == 'warm':
                
                warm_y = test_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_pred = pred2[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_auc.append(roc_auc_score(warm_y,warm_pred))
                warm_logloss.append(log_loss(warm_y,warm_pred))
                
            if wc == 'cold':
                
                cold_y = test_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred2[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]            
                cold_auc.append(roc_auc_score(cold_y,cold_pred))
                cold_logloss.append(log_loss(cold_y,cold_pred))
 
        if wc == 'warm':
            i_result = np.array(['SVD',np.mean(warm_auc),np.mean(warm_logloss),np.std(warm_auc),np.std(warm_logloss)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','warm_auc','warm_logloss','std_warm_auc','std_warm_logloss'])

        if wc == 'cold':
            i_result = np.array(['SVD',np.mean(cold_auc),np.mean(cold_logloss),np.std(cold_auc),np.std(cold_logloss)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','cold_auc','cold_logloss','std_cold_auc','std_cold_logloss'])
            

        return result        
