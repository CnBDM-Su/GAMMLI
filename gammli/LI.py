# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:56:00 2020

@author: suyu
"""

import numpy as np 
from .soft_impute import SoftImpute
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
from joblib import Parallel, delayed 
from itertools import product
import pandas as pd

class LatentVariable:
    def __init__(self,
                 task_type=None,
                 mf_type='als',
                 shrinkage_value=None,
                 convergence_threshold=0.001,
                 max_iters=20,
                 max_rank=None,
                 n_power_iterations=1,
                 n_oversamples=10,
                 val_ratio=0.2,
                 init_fill_method="zero",
                 min_value=None,
                 max_value=None,
                 change_mode = False,
                 normalizer=None,
                 verbose=True,
                 u_group = 0,
                 i_group = 0,
                 val_u_group = 0,
                 val_i_group = 0,
                 scale_ratio=1,
                 beta=0.5,
                 auto_tune=False,
                 pred_tr=None,
                 tr_y=None,
                 pred_val=None,
                 val_y=None,
                 tr_Xi=None,
                 val_Xi=None,
                 random_state =0,
                 combine_range=0.99,
                 wc = None):
        """
        mf_type: string
            type two algorithms are implements, type="svd" or the default type="als". The
            "svd" algorithm repeatedly computes the svd of the completed matrix, and soft
            thresholds its singular values. Each new soft-thresholded svd is used to reimpute 
            the missing entries. For large matrices of class "Incomplete", the svd
            is achieved by an efficient form of alternating orthogonal ridge regression. The
            softImpute 11 "als" algorithm uses this same alternating ridge regression, but updates 
            the imputation at each step, leading to quite substantial speedups in some cases. The
            "als" approach does not currently have the same theoretical convergence guarantees as the "svd" approach.
            thresh convergence threshold, measured as the relative change in the Frobenius norm
            between two successive estimates.
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 100.
        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.
        max_iters : int
            Maximum number of SVD iterations
        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.
        n_power_iterations : int
            Number of power iterations to perform with randomized SVD
        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.
        min_value : float
            Smallest allowable value in the solution
        max_value : float
            Largest allowable value in the solution
        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods
        verbose : bool
            Print debugging info
        """
        super(LatentVariable, self).__init__()
        self.task_type = task_type
        self.mf_type = mf_type
        self.fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.change_mode =change_mode
        self.n_power_iterations = n_power_iterations
        self.n_oversamples = n_oversamples
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.u_group = u_group
        self.i_group = i_group
        self.val_u_group = val_u_group
        self.val_i_group = val_i_group
        self.scale_ratio = scale_ratio

        self.beta = beta
        self.auto_tune = auto_tune
        
        self.pred_tr=pred_tr
        self.tr_y=tr_y
        self.pred_val=pred_val
        self.val_y=val_y
        self.tr_Xi = tr_Xi
        self.val_Xi = val_Xi
        self.random_state = random_state
        self.combine_range= combine_range
        self.wc = wc


    def fit(self,Xi,val_Xi,residual,residual_val,ui_shape):

        train_index = Xi
        val_index = val_Xi
        residual_train = residual
        residual_val = residual_val
        matrix =np.zeros(shape=[ui_shape[0],ui_shape[1]])
        for i in range(train_index.shape[0]):
            matrix[int(train_index[i,0]),int(train_index[i,1])] = residual_train[i]
        #for i in range(val_index.shape[0]):
         #   matrix[int(val_index[i,0]),int(val_index[i,1])] = residual_val[i]  
        matrix[matrix==0] = np.nan
        #matrix = BiScaler(tolerance=0.1).fit_transform(matrix)

        print('missing value counts:',np.isnan(matrix).sum())
        if self.auto_tune:
            self.auto_tuning(5,matrix)
        else:
            self.best_ratio = self.scale_ratio
            self.best_combine_range = self.combine_range
        X_filled_softimpute, self.u, self.v, self.s, self.mf_mae, self.mf_valmae, self.match_u, self.match_i, self.var_u, self.var_i, var_whole_u, var_whole_i, self.pre_u, self.pre_i = SoftImpute(task_type = self.task_type,
                                                                                                                                                                                           combine = True,
                                                                                                                                                                                           auto_tune = False,
                                                                                                                                                                                           verbose = self.verbose,
                                                                                                                                                                                           shrinkage_value=self.shrinkage_value,
                                                                                                                                                                                           convergence_threshold=self.convergence_threshold,
                                                                                                                                                                                           max_iters=self.max_iters,
                                                                                                                                                                                           max_rank=self.max_rank,
                                                                                                                                                                                           n_oversamples = self.n_oversamples,
                                                                                                                                                                                           n_power_iterations=self.n_power_iterations,
                                                                                                                                                                                           init_fill_method=self.fill_method,
                                                                                                                                                                                           min_value=self.min_value,
                                                                                                                                                                                           max_value=self.max_value,
                                                                                                                                                                                           change_mode = self.change_mode,
                                                                                                                                                                                           normalizer=self.normalizer,
                                                                                                                                                                                           u_group = self.u_group,
                                                                                                                                                                                           i_group = self.i_group,
                                                                                                                                                                                           val_u_group = self.val_u_group,
                                                                                                                                                                                           val_i_group = self.val_i_group,
                                                                                                                                                                                           scale_ratio = self.best_ratio,
                                                                                                                                                                                           pred_tr = self.pred_tr,
                                                                                                                                                                                           tr_y = self.tr_y,
                                                                                                                                                                                           pred_val=self.pred_val,
                                                                                                                                                                                           val_y=self.val_y,
                                                                                                                                                                                           tr_Xi=self.tr_Xi,
                                                                                                                                                                                           val_Xi=self.val_Xi,
                                                                                                                                                                                           combine_range = self.best_combine_range,
                                                                                                                                                                                           wc = self.wc).fit_transform(matrix)
        self.filled_matrix = X_filled_softimpute
        current_rank = self.u.shape[1]
        self.cur_rank = current_rank        

    
    def predict(self,Xi):

        pred2 = []
        for i in range(Xi.shape[0]):
            pred2.append(self.filled_matrix[int(Xi[i,0]),int(Xi[i,1])])
        pred2 = np.ravel(np.array(pred2))
        final_pred = pred2
        #final_pred = self.filled_matrix[int(Xi[0,0]),int(Xi[0,1])]
        return final_pred
    
    
    def predictval(self, Xi, pre_u, pre_i, u, v, s, match_u, match_i):
        
        v=v.T
            
        pred1 = self.pred_val
            
        pred2 = []
        for i in range(Xi.shape[0]):
            if Xi[i,0] == 'cold':
                g = self.val_u_group[i]
                group_pre_u = pre_u
                for i in range(1000):
                    if g in list(group_pre_u.keys()):
                        g = group_pre_u[g]
                    else:
                        break
                u_ = match_u[g]
                #u=np.zeros(u.shape)
            else:
                u_ = u[int(Xi[i,0])]
            if Xi[i,1] == 'cold':
                g = self.val_i_group[i]
                group_pre_i = pre_i
                for i in range(1000):
                    if g in list(group_pre_i.keys()):
                        g = group_pre_i[g]
                    else:
                        break
                v_ = match_i[g]
                #v =np.zeros(v.shape)
            else:
                v_ =v[int(Xi[i,1])]

            pred_mf = np.dot(u_, np.multiply(np.diag(s), v_))
            pred2.append(pred_mf)
        pred2 = np.array(pred2)

        pred = pred1.ravel()+ pred2.ravel()
            
            
        if self.task_type == 'Classification':
            #print(pred.shape)
            #pred = tf.nn.softmax(pred).numpy()
            #print(pred.shape)
            pred[pred>1]=1
            pred[pred<0]=0


        return pred
    
    def dispersion(self, avg, radius):
        sep = []
        for i in range(avg.shape[0]):
            distance = []
            rad_sum = []
            for j in range(avg.shape[0]):
                distance.append(np.sqrt(np.sum(np.square(avg[i]-avg[j]))))
                rad_sum.append(radius[i]+radius[j])
            sepa = np.array(distance) - np.array(rad_sum)
            sep.append(avg.shape[0] - sepa[sepa>0].shape[0])
        sep = np.array(sep).mean()

        return sep

    def evaluate(self, y, pred_y):
        if self.task_type == 'Regression':
            eva = mean_absolute_error(y,pred_y)
        elif self.task_type == 'Classification':
            eva = roc_auc_score(y, pred_y)
            
        return eva
    
    


    def auto_tuning(self,times,matrix):
        
        def once(ratio,dis):
            print(np.unique(self.u_group).shape)
            X_filled_softimpute, u, v, s, mf_mae, mf_valmae, match_u, match_i, var_u, var_i, var_whole_u, var_whole_i, pre_u, pre_i = SoftImpute(task_type = self.task_type,
                                                                                                                                   auto_tune = True,
                                                                                                                                   combine=True,
                                                                                                                                   verbose = self.verbose,
                                                                                                                                   shrinkage_value=self.shrinkage_value,
                                                                                                                                   convergence_threshold=self.convergence_threshold,
                                                                                                                                   max_iters=self.max_iters,
                                                                                                                                   max_rank=self.max_rank,
                                                                                                                                   n_oversamples = self.n_oversamples,
                                                                                                                                   n_power_iterations=self.n_power_iterations,
                                                                                                                                   init_fill_method=self.fill_method,
                                                                                                                                   min_value=self.min_value,
                                                                                                                                   max_value=self.max_value,
                                                                                                                                   change_mode = self.change_mode,
                                                                                                                                   normalizer=self.normalizer,
                                                                                                                                   u_group = self.u_group,
                                                                                                                                   i_group = self.i_group,
                                                                                                                                   val_u_group = self.val_u_group,
                                                                                                                                   val_i_group = self.val_i_group,
                                                                                                                                   scale_ratio = ratio,
                                                                                                                                   pred_tr = self.pred_tr,
                                                                                                                                   tr_y = self.tr_y,
                                                                                                                                   pred_val=self.pred_val,
                                                                                                                                   val_y=self.val_y,
                                                                                                                                   tr_Xi=self.tr_Xi,
                                                                                                                                   val_Xi=self.val_Xi,
                                                                                                                                   combine_range=dis,
                                                                                                                                  wc = self.wc).fit_transform(matrix)
            #val_mae = mf_valmae[-1]
            '''
            var_mean = 0
            for i in var_u.keys():
                var_mean += var_u[i].mean()
            var_u = var_mean/len(var_u)
            var_mean =0
            for i in var_i.keys():
                var_mean += var_i[i].mean()
            var_i = var_mean/len(var_i)
            val_var = (var_u + var_i)/2
            
            mean_class_u = list(match_u.values())
            class_var_u = np.var(mean_class_u)
            
            mean_class_i = list(match_i.values())
            class_var_i = np.var(mean_class_i)
            
            class_var = ((class_var_u + class_var_i)/2).mean()
            
            print(val_mae,val_var,val_w_var)
            
            res_stat = pd.DataFrame(np.vstack([val_mae,val_var,class_var, ratio, dis]).T, 
                            columns = ['val_mae', "val_var", "val_w_var", "ratio", "dis"])
            '''
            
            pred_at = self.predictval(self.val_Xi,pre_u,pre_i, u, v, s, match_u, match_i)
            val_eva = self.evaluate(self.val_y, pred_at)
            
            
            res_stat = pd.DataFrame(np.vstack([val_eva, ratio, dis]).T, 
                            columns = ['val_eva', "ratio", "dis"])
            return res_stat
            
        print('#####start auto_tuning#####')
        start_r = 0
        end_r = 1
        start_d = 0.4
        end_d = 1        
        for i in range(times-1):
            candidate_r = np.linspace(start_r,end_r,times)
            candidate_d = np.linspace(start_d,end_d,times)
            stat = Parallel(n_jobs=-1,timeout=3000)(delayed(once)(j,k) for j in candidate_r for k in candidate_d)
            stat = pd.concat(stat)
            evaluation = stat.val_eva.tolist()
            if self.task_type=='Regression':
                r = stat.iloc[np.argmin(evaluation),-2]
                d = stat.iloc[np.argmin(evaluation),-1]
            elif self.task_type == 'Classification':
                r = stat.iloc[np.argmax(evaluation),-2]
                d = stat.iloc[np.argmax(evaluation),-1]
            if r-((end_r-start_r)/times) > start_r:
                start_r = r-((end_r-start_r)/times)
            if r + ((end_r-start_r)/times) < end_r:
                end_r = r + ((end_r-start_r)/times)
            if d-((end_d-start_d)/times) > start_d:
                start_d = d-((end_d-start_d)/times)
            if d + ((end_d-start_d)/times) < end_d:
                end_d = d + ((end_d-start_d)/times)                 
                    
            self.best_ratio = r
            self.best_combine_range = d
        print('the best shrinkage is %f' %self.best_ratio)
        print('the best combination is %f' %self.best_combine_range)
        
    def rank_norm(self,Xi):
               
        std = []
        v = self.v.T
        u = self.u
        s = self.s
        for i in range(self.u.shape[1]):
            res = []
            for j in range(Xi.shape[0]):
                
                res.append(u[Xi[j,0],i] * v[Xi[j,1],i] *s[i,i])
    
            std.append(np.std(res,ddof=1))
        
        return std
        
        
            
        
            



