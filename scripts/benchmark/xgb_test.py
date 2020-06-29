# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:59:51 2020

@author: suyu
"""
from sklearn.model_selection import train_test_split,GridSearchCV,PredefinedSplit
from sklearn.metrics import make_scorer,mean_squared_error,roc_auc_score,mean_absolute_error,log_loss
from xgboost import XGBClassifier,XGBRegressor
import numpy as np
import pandas as pd

def xgb(wc, train, test, train_x, tr_Xi, train_y , test_x , te_Xi, test_y, meta_info, model_info, task_type="Regression", val_ratio=0.2, random_state=0):
    
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    if task_type == "Regression":
        idx1, idx2 = train_test_split(indices, test_size=val_ratio, random_state=random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1

        base = XGBRegressor(n_estimators=100, random_state=random_state)
        grid = GridSearchCV(base, param_grid={"max_depth": (3, 4, 5, 6, 7, 8)},
                            scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
                            cv=PredefinedSplit(val_fold), refit=False, n_jobs=-1, error_score=np.nan)
        grid.fit(train_x, train_y.ravel())
        model = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
        cold_mae = []
        cold_rmse = []
        warm_mae = []
        warm_rmse = []
        for times in range(10):
            model.random_state=times
            model.fit(train_x[idx1, :], train_y[idx1, :].ravel())
            pred_test = model.predict(test_x).reshape([-1, 1])
            
            if wc == 'warm':
                
                warm_y = test_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_pred = pred_test[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_mae.append(mean_absolute_error(warm_y,warm_pred))
                warm_rmse.append(mean_squared_error(warm_y,warm_pred)**0.5)
                
            if wc == 'cold':
            
                cold_y = test_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred_test[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_mae.append(mean_absolute_error(cold_y,cold_pred))
                cold_rmse.append(mean_squared_error(cold_y,cold_pred)**0.5)

        if wc == 'warm':
            
            i_result = np.array(['xgboost',np.mean(warm_mae),np.mean(warm_rmse),np.std(warm_mae),np.std(warm_rmse)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','warm_mae','warm_rmse','std_warm_mae','std_warm_rmse'])

        if wc == 'cold':  
            i_result = np.array(['xgboost',np.mean(cold_mae),np.mean(cold_rmse),np.std(cold_mae),np.std(cold_rmse)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','cold_mae','cold_rmse','std_cold_mae','std_cold_rmse',])
        

        return result


    elif task_type == "Classification":
        idx1, idx2 = train_test_split(indices, test_size=val_ratio, stratify=train_y, random_state=random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1

        base = XGBClassifier(n_estimators=100, random_state=random_state)
        grid = GridSearchCV(base, param_grid={"max_depth": (3, 4, 5, 6, 7, 8)},
                            scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)},
                            cv=PredefinedSplit(val_fold), refit=False, n_jobs=-1, error_score=np.nan)
        grid.fit(train_x, train_y.ravel())
        model = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_auc"] == 1))[0][0]])
        
        cold_auc = []
        cold_logloss = []
        warm_auc = []
        warm_logloss = []
        for times in range(10):
            
            model.random_state=times
            model.fit(train_x, train_y.ravel())
            pred_test = model.predict_proba(test_x)[:,-1].reshape([-1, 1])
            
            if wc == 'warm':

                warm_y = test_y[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_pred = pred_test[(te_Xi[:,1] != 'cold') & (te_Xi[:,0] != 'cold')]
                warm_auc.append(roc_auc_score(warm_y,warm_pred))
                warm_logloss.append(log_loss(warm_y,warm_pred))   
                
            if wc == 'cold':
                
                cold_y = test_y[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_pred = pred_test[(te_Xi[:,1] == 'cold') | (te_Xi[:,0] == 'cold')]
                cold_auc.append(roc_auc_score(cold_y,cold_pred))
                cold_logloss.append(log_loss(cold_y,cold_pred))

        if wc == 'warm':
            i_result = np.array(['xgboost',np.mean(warm_auc),np.mean(warm_logloss),np.std(warm_auc),np.std(warm_logloss)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','warm_auc','warm_logloss','std_warm_auc','std_warm_logloss'])

        if wc == 'cold':
            i_result = np.array(['xgboost',np.mean(cold_auc),np.mean(cold_logloss),np.std(cold_auc),np.std(cold_logloss)]).reshape(1,-1)
            result = pd.DataFrame(i_result,columns=['model','cold_auc','cold_logloss','std_cold_auc','std_cold_logloss'])
            

        return result