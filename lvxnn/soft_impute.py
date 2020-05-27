# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array

from .common import masked_mae
from .solver import Solver

F32PREC = np.finfo(np.float32).eps


class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_value=None,
            convergence_threshold=0.001,
            max_iters=100,
            max_tuning_iters=20,
            max_rank=None,
            n_power_iterations=1,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None,
            change_mode=False,
            verbose=True,
            u_group = 0,
            i_group = 0,
            scale_ratio=1,
            n_oversamples=10):
        """
        Parameters
        ----------
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
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer,
            change_mode = change_mode,
            u_group = 0,
            i_group = 0,
            scale_ratio=1)

        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose
        self.u_group = u_group
        self.i_group = i_group
        self.scale_ratio = scale_ratio
        self.max_tuning_iters = max_tuning_iters
        self.n_oversamples = n_oversamples

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
        # edge cases
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value,step,tuning=False, max_rank=None):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                power_iteration_normalizer='QR',
                n_oversamples=self.n_oversamples,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)

        if tuning:
            U,V ,self.match_u, self.match_i,self.var_u, self.var_i, self.radius_u, self.radius_i = self.cluster_mean(U,V,self.u_group,self.i_group,self.u_max_d,self.i_max_d,self.scale_ratio)
        else:
            self.u_max_d,self.i_max_d = self.get_max_dis(U,V,self.u_group,self.i_group)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)

        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        
        return X_reconstruction, rank, U_thresh, V_thresh, S_thresh

    def _max_singular_value(self, X_filled):
        # quick decomposition of X_filled into rank-1 SVD
        _, s, _ = randomized_svd(
            X_filled,
            1,
            n_iter=5)
        return s[0]
    
    
    def solve(self, X, X_val, missing_mask, missing_val):
        X = check_array(X, force_all_finite=False)
        X_val = check_array(X_val, force_all_finite=False)
        
        X_init = X.copy()
        self.mae_record = []
        self.valmae_record = []

        X_filled = X
        observed_mask = ~missing_mask
        val_mask = ~missing_val
        max_singular_value = self._max_singular_value(X_filled)
        if self.verbose:
            print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if self.shrinkage_value:
            shrinkage_value = self.shrinkage_value
        else:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            shrinkage_value = max_singular_value / 50.0

        print('#####mf_training#####')
        for i in range(self.max_iters):
            X_reconstruction, rank, U_thresh, V_thresh, S_thresh = self._svd_step(
                X_filled,
                shrinkage_value,
                tuning=False,
                step=1,
                max_rank=self.max_rank)
            X_reconstruction = self.clip(X_reconstruction)

            # print error on observed data
            if self.verbose:
                mae = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask)
                self.mae_record.append(mae)
                
                val_mae = masked_mae(
                        X_true = X_val,
                        X_pred = X_reconstruction,
                        mask = val_mask)
                self.valmae_record.append(val_mae)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f validation MAE=%0.6f,rank=%d" % (
                        i + 1,
                        mae,
                        val_mae,
                        rank))

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            
            
            if converged:
                break
        #X_filled[missing_mask]=0
        # print(X_reconstruction[observed_mask])


        self.ini_u = X_filled
        print('######start tuning######')
        
        for i in range(self.max_tuning_iters):
            X_reconstruction, rank, U_thresh, V_thresh, S_thresh = self._svd_step(
                X_filled,
                shrinkage_value,
                tuning=True,
                step=1,
                max_rank=self.max_rank)
            X_reconstruction = self.clip(X_reconstruction)

            # print error on observed data
            if self.verbose:
                mae = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask)
                self.mae_record.append(mae)
                
                val_mae = masked_mae(
                        X_true = X_val,
                        X_pred = X_reconstruction,
                        mask = val_mask)
                self.valmae_record.append(val_mae)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f validation MAE=%0.6f,rank=%d" % (
                        i + 1,
                        mae,
                        val_mae,
                        rank))

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstruction[missing_mask]
           # print(X_reconstruction[observed_mask])

            if converged:
                break
        
        
        
        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))
        
            
        if self.change_mode:
            X_filled = X_reconstruction
        return X_filled, U_thresh, V_thresh, S_thresh ,self.mae_record, self.valmae_record, self.match_u, self.match_i, self.ini_u, self.var_u, self.var_i, self.radius_u, self.radius_i
    
    def cluster_mean(self, u,v,u_group,i_group,u_max_d,i_max_d,scale_ratio):
        
        v=v.T
        match_u = dict()
        match_i = dict()
        var_u = dict()
        var_i = dict()
        radius_u = dict()
        radius_i = dict()
        
        def center_move(point_t,avg,r):
            if point_t.shape[0] ==1:
                new = point_t
                return new
                    
            new = (r*point_t+(1-r)*avg).reshape(-1,1,3)
    
            return new
        
        
        def center_restrict(point_t,max_d,avg,r):
            new = point_t
            if point_t.shape[0] ==1:
                return new

            d=[]
            for i in range(point_t.shape[0]):
                d.append(np.sqrt(np.sum(np.square(point_t[i]-avg))))    
            d=np.array(d).reshape(-1,1)
            count=0
            for j in range(point_t.shape[0]):
                
                if d[j] > max_d*r:
                    count = count+1
                    new[j] = ((1-r)*(avg-point_t[j]))+point_t[j]
            print(count)
            new = new.reshape(-1,1,3)
            
                    
            #new = (avg+(d/r*d.max())*(point_t-avg)).reshape(-1,1,3)

    
            return new
        
        if type(u_group) != int :
            for i in np.unique(u_group):
                cus = np.argwhere(u_group==i)
                group = u[cus,:].reshape(-1,u.shape[1])
                avg = np.mean(group,axis=0)
                var = np.var(group,axis=0)
                point_t = u[cus].reshape(-1,3)
                #u[cus] = center_move(point_t,u_max_d[i],avg,scale_ratio)
                u[cus] = center_restrict(point_t,u_max_d[i],avg,scale_ratio)
                #u[cus] = avg
                match_u[i] = avg
                var_u[i] = var
                radius_u[i] =u_max_d[i]*scale_ratio
            
                    
        if type(i_group) != int :
            for j in np.unique(i_group):
                cus = np.argwhere(i_group==j)
                group = v[cus,:].reshape(-1,v.shape[1])
                avg = np.mean(group,axis=0)
                var = np.var(group,axis=0)
                point_t = v[cus].reshape(-1,3)
                #v[cus] = center_move(point_t,i_max_d[j],avg,scale_ratio)
                v[cus] = center_restrict(point_t,i_max_d[j],avg,scale_ratio)
                #v[cus] = avg
                match_i[j] = avg
                var_i[j] = var
                radius_i[i] =i_max_d[j]*scale_ratio
        v=v.T
                
                
        return u,v,match_u,match_i,var_u,var_i, radius_u, radius_i
    
    def get_max_dis(self, u,v,u_group,i_group):
        v=v.T
        
        def max_distance(point_t,avg):
            d=[]
            for i in range(point_t.shape[0]):
                d.append(np.sqrt(np.sum(np.square(point_t[i]-avg))))    
            d=np.array(d).reshape(-1,1)
            return d.max()
        
        u_max_d = []
        if type(u_group) != int :
            for i in np.unique(u_group):
                cus = np.argwhere(u_group==i)
                group = u[cus,:].reshape(-1,u.shape[1])
                avg = np.mean(group,axis=0)           
                point_t = u[cus].reshape(-1,3)
                u_max_d.append(max_distance(point_t,avg))
            
                    
        i_max_d = []
        if type(i_group) != int :            
            for j in np.unique(i_group):
                cus = np.argwhere(i_group==j)
                group = v[cus,:].reshape(-1,v.shape[1])
                avg = np.mean(group,axis=0)
                point_t = v[cus].reshape(-1,3)
                i_max_d.append(max_distance(point_t,avg))
                #v[cus] = avg

                                
        return u_max_d,i_max_d
    


                    
                    
                
