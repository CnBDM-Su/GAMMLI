# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:56:00 2020

@author: suyu
"""

import numpy as np 
from .soft_impute import SoftImpute
from sklearn.preprocessing import MinMaxScaler


class LatentVariable:
    def __init__(self,
                 task_type=None,
                 mf_type='als',
                 shrinkage_value=None,
                 convergence_threshold=0.001,
                 max_iters=100,
                 max_tuning_iters=50,
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
                 scale_ratio=1,
                 alpha=0.5,
                 auto_tune=False,
                 pred_tr=None,
                 tr_y=None,
                 pred_val=None,
                 val_y=None,
                 tr_Xi=None,
                 val_Xi=None):
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
        self.scale_ratio = scale_ratio
        self.max_tuning_iters = max_tuning_iters

        self.alpha = alpha
        self.auto_tune = auto_tune
        
        self.pred_tr=pred_tr
        self.tr_y=tr_y
        self.pred_val=pred_val
        self.val_y=val_y
        self.tr_Xi = tr_Xi
        self.val_Xi = val_Xi


    def fit(self,Xi,val_Xi,residual,residual_val,ui_shape):

        train_index = Xi
        val_index= val_Xi
        residual_train = residual
        residual_val = residual_val
        matrix =np.zeros(shape=[ui_shape[0],ui_shape[1]])
        for i in range(train_index.shape[0]):
            matrix[int(train_index[i,0]),int(train_index[i,1])] = residual_train[i]
        matrix_val =np.zeros(shape=[ui_shape[0],ui_shape[1]])
        for i in range(val_index.shape[0]):
            matrix_val[int(val_index[i,0]),int(val_index[i,1])] = residual_val[i]
        matrix_val[matrix_val==0] = np.nan
        matrix[matrix==0] = np.nan
        #matrix = BiScaler(tolerance=0.1).fit_transform(matrix)

        matrix[matrix==0] = np.nan
        print('missing value counts:',np.isnan(matrix).sum())
        if self.auto_tune:
            self.auto_tuning(5,matrix,matrix_val)
        self.best_ratio = self.scale_ratio
        X_filled_softimpute, self.u, self.v, self.s, self.mf_mae, self.mf_valmae, self.match_u, self.match_i, self.ini_u,self.var_u, self.var_i, self.radius_u, self.radius_i = SoftImpute(task_type = self.task_type,
                                                                                                                                                                                           shrinkage_value=self.shrinkage_value,
                                                                                                                                                                                           convergence_threshold=self.convergence_threshold,
                                                                                                                                                                                           max_iters=self.max_iters,
                                                                                                                                                                                           max_tuning_iters = self.max_tuning_iters,
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
                                                                                                                                                                                           scale_ratio = self.best_ratio,
                                                                                                                                                                                           pred_tr = self.pred_tr,
                                                                                                                                                                                           tr_y = self.tr_y,
                                                                                                                                                                                           pred_val=self.pred_val,
                                                                                                                                                                                           val_y=self.val_y,
                                                                                                                                                                                           tr_Xi=self.tr_Xi,
                                                                                                                                                                                           val_Xi=self.val_Xi).fit_transform(matrix,matrix_val)
        self.filled_matrix = X_filled_softimpute
        current_rank = self.u.shape[1]
        self.cur_rank = current_rank        



    def predict(self,Xi):

        pred2 = []
        for i in range(Xi.shape[0]):
            pred2.append(self.filled_matrix[int(Xi[i,0]),int(Xi[i,1])])
        pred2 = np.ravel(np.array(pred2))
        final_pred = pred2
        return final_pred

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

    def evaluate(self, var, mae, alpha):
        model = MinMaxScaler()
        var = model.fit_transform(var.reshape(-1,1))
        mae = model.fit_transform(mae.reshape(-1,1))
        badness = alpha * mae + (1-alpha) * var

        return badness


    def auto_tuning(self,times,matrix,matrix_val):
        start = 0
        end = 1
        for i in range(times-1):
            val_mae = []
            val_var = []
            candidate = np.linspace(start,end,times)
            for i in candidate:
                X_filled_softimpute, self.u, self.v, self.s, self.mf_mae, self.mf_valmae, self.match_u, self.match_i, self.ini_u, self.var_u, self.var_i, self.radius_u, self.radius_i = SoftImpute(task_type = self.task_type,
                                                                                                                                                                                                    shrinkage_value=self.shrinkage_value,
                                                                                                                                                                                                    convergence_threshold=self.convergence_threshold,
                                                                                                                                                                                                    max_iters=self.max_iters,
                                                                                                                                                                                                    max_tuning_iters = self.max_tuning_iters,
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
                                                                                                                                                                                                    scale_ratio = i,
                                                                                                                                                                                                    pred_tr = self.pred_tr,
                                                                                                                                                                                                    tr_y = self.tr_y,
                                                                                                                                                                                                    pred_val=self.pred_val,
                                                                                                                                                                                                    val_y=self.val_y,
                                                                                                                                                                                                    tr_Xi=self.tr_Xi,
                                                                                                                                                                                                    val_Xi=self.val_Xi).fit_transform(matrix,matrix_val)
                val_mae.append(self.mf_valmae[-1])
                mean_var = 0
                for i in range(len(self.var_u)):
                    mean_var += self.var_u[i].mean()
                var_u = mean_var/len(self.var_u)
                mean_var = 0
                for i in range(len(self.var_u)):
                    mean_var += self.var_u[i].mean()
                var_i = mean_var/len(self.var_u)
                val_var.append((var_u + var_i)/2)
            badness = self.evaluate(np.array(val_var),np.array(val_mae),self.alpha).tolist()
            start = candidate[np.argmin(badness)-1 if np.argmin(badness)!=0 else np.argmin(badness)]
            end = candidate[np.argmin(badness)+1 if np.argmin(badness)!=times-1 else np.argmin(badness)]
            self.best_ratio = candidate[np.argmin(badness)]







