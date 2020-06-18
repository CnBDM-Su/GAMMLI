# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:56:00 2020

@author: suyu
"""
import os
import time
import numpy as np
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from .LV import LatentVariable
from .gaminet import GAMINet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans 
import networkx as nx

class LV_XNN:
    def __init__(self,
                 meta_info=None,
                 model_info=None,
                 subnet_arch=[10, 6],
                 interact_arch=[20, 10],
                 activation_func=tf.tanh,
                 bn_flag=True,
                 lr_bp=0.001,
                 loss_threshold_main=0.01,
                 loss_threshold_inter=0.01,
                 main_grid_size=41,
                 interact_grid_size=41,
                 batch_size=1000,
                 main_effect_epochs=10000, 
                 tuning_epochs=500,
                 interaction_epochs=20,
                 interact_num=20,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=100,
                 shrinkage_value=None,
                 convergence_threshold=0.001,
                 mf_training_iters=1,
                 mf_tuning_iters=50,
                 max_rank=None,
                 n_power_iterations=1,
                 n_oversamples=10,
                 init_fill_method="zero",
                 min_value=None,
                 max_value=None,
                 change_mode = False,
                 normalizer=None,
                 multi_type_num=0,
                 u_group_num=0,
                 i_group_num=0,
                 scale_ratio=1,
                 alpha=0.5,
                 auto_tune=False,
                 epsilon = 0.5,
                 random_state = 0,
                 combine_range=0.99):

        super(LV_XNN, self).__init__()

        self.meta_info = meta_info
        self.model_info = model_info

        self.bn_flag = bn_flag
        self.subnet_arch = subnet_arch
        self.interact_arch = interact_arch
        self.activation_func = activation_func

        self.lr_bp = lr_bp
        self.loss_threshold_main = loss_threshold_main
        self.loss_threshold_inter = loss_threshold_inter
        self.main_grid_size = main_grid_size
        self.interact_grid_size = interact_grid_size
        self.batch_size = batch_size
        self.tuning_epochs = tuning_epochs
        self.main_effect_epochs = main_effect_epochs
        self.interaction_epochs = interaction_epochs
        self.interact_num = interact_num

        self.verbose = verbose
        self.val_ratio = val_ratio
        self.early_stop_thres = early_stop_thres

        self.fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.mf_max_iters = mf_training_iters
        self.max_tuning_iters = mf_tuning_iters
        self.max_rank = max_rank
        self.change_mode =change_mode
        self.n_power_iterations = n_power_iterations
        self.n_oversamples = n_oversamples

        self.multi_type_num = multi_type_num
        self.u_group_num = u_group_num
        self.i_group_num = i_group_num
        self.scale_ratio = scale_ratio
        self.alpha = alpha
        self.auto_tune = auto_tune
        self.epsilon = epsilon
        self.random_state = random_state
        self.combine_range = combine_range


        tf.random.set_seed(self.random_state)
        simu_dir = "./results/gaminet/"
        #path = 'data/simulation/sim_0.9.csv'
        if not os.path.exists(simu_dir):
            os.makedirs(simu_dir)

        self.task_type = self.model_info['task_type']
        self.feat_dict = self.model_info['feat_dict']
        self.ui_shape = self.model_info['ui_shape']

        if self.task_type == "Regression":
            #self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    #gam first mf second    
    def fit(self,xx,Xi,y):

        #initial cluster training
        self.user_feature_list = []
        self.item_feature_list = []
        for indice, (feature_name, feature_info) in enumerate(self.meta_info.items()):
            if feature_info["source"] == "user":
                self.user_feature_list.append(indice)
            elif feature_info["source"] == "item":
                self.item_feature_list.append(indice)


        user_feature = np.concatenate([xx[:,self.user_feature_list],Xi[:,0].reshape(-1,1)],1)
        item_feature = np.concatenate([xx[:,self.item_feature_list],Xi[:,1].reshape(-1,1)],1)
        user_feature = np.unique(user_feature,axis=0)
        item_feature = np.unique(item_feature,axis=0)
        user_feature = user_feature[np.argsort(user_feature[:,-1])]
        item_feature = item_feature[np.argsort(item_feature[:,-1])]

        user_feature = user_feature[:,:-1]
        item_feature = item_feature[:,:-1]

        if self.u_group_num != 0:
            self.u_group, self.u_group_model = self.main_effect_cluster(user_feature,self.u_group_num)
        else:
            self.u_group=0
        if self.i_group_num != 0:
            self.i_group, self.i_group_model = self.main_effect_cluster(item_feature,self.i_group_num)
        else:
            self.i_group = 0

        error1=[]
        val_error1=[]
        error2=[]
        val_error2 =[]
        val_error = []


        #gam fit
        self.gami_model = GAMINet(meta_info=self.meta_info,interact_num=self.interact_num,interact_arch=self.interact_arch,
                                  subnet_arch=self.subnet_arch, task_type=self.task_type,main_grid_size=self.main_grid_size,
                                  interact_grid_size=self.interact_grid_size,activation_func=tf.tanh, batch_size=self.batch_size, lr_bp=self.lr_bp,
                                  main_effect_epochs=self.main_effect_epochs,tuning_epochs=self.tuning_epochs, multi_type_num = self.multi_type_num,
                                  loss_threshold_main=self.loss_threshold_main,loss_threshold_inter=self.loss_threshold_inter,interaction_epochs=self.interaction_epochs,
                                  verbose=self.verbose, val_ratio=self.val_ratio, early_stop_thres=self.early_stop_thres,random_state=self.random_state)


        model = self.gami_model
        st_time = time.time()

        model.fit(xx, y)
        val_x = xx[model.val_idx, :]
        val_y = y[model.val_idx, :]
        val_Xi = Xi[model.val_idx, :]
        tr_x = xx[model.tr_idx, :]
        tr_y = y[model.tr_idx, :]
        tr_Xi = Xi[model.tr_idx, :]

        fi_time = time.time()
        print('time cost:',fi_time-st_time)

        pred_train = model.predict(tr_x)
        pred_val = model.predict(val_x)           

        error1.append(self.loss_fn(tr_y.ravel(),pred_train.ravel()).numpy())
        val_error1.append(self.loss_fn(val_y.ravel(),pred_val.ravel()).numpy())

        print('After the gam stage, training error is %0.5f , validation error is %0.5f' %(error1[-1],val_error1[-1]))
        residual = (tr_y.ravel() - pred_train.ravel()).reshape(-1,1)
        residual_val = (val_y.ravel() - pred_val.ravel()).reshape(-1,1)
        
        if self.task_type == 'Classification':
            residual[abs(residual)<self.epsilon]=0
            residual_val[abs(residual_val)<self.epsilon]=0
            self.residual = residual



        #mf fit
        if self.mf_max_iters !=0:

            self.lv_model = LatentVariable(verbose = self.verbose,task_type=self.task_type,max_rank=self.max_rank,max_iters=self.mf_max_iters,alpha=self.alpha,
                                           max_tuning_iters=self.max_tuning_iters,change_mode=self.change_mode,auto_tune=self.auto_tune,
                                           convergence_threshold=self.convergence_threshold,n_oversamples=self.n_oversamples
                                           ,u_group = self.u_group,i_group = self.i_group,scale_ratio=self.scale_ratio,pred_tr=pred_train,
                                           tr_y=tr_y,pred_val=pred_val,val_y=val_y, tr_Xi=tr_Xi,val_Xi=val_Xi,random_state=self.random_state,combine_range=self.combine_range)
            model1 = self.lv_model
            st_time = time.time()
            model1.fit(tr_Xi,val_Xi,residual,residual_val,self.ui_shape)
            fi_time = time.time()
            print('time cost:',fi_time-st_time)

            pred = model1.predict(tr_Xi)
            predval = model1.predict(val_Xi)
            error2.append(self.loss_fn(tr_y.ravel(),pred.ravel()+pred_train.ravel()).numpy())
            val_error2.append(self.loss_fn(val_y.ravel(),predval.ravel()+pred_val.ravel()).numpy())
            self.mf_tr_err = error2[-1]
            self.mf_val_err = val_error2[-1]
            print('After the matrix factor stage, training error is %0.5f, validation error is %0.5f' %(error2[-1],val_error2[-1]))


            val_error_bi = [val_error1[-1],val_error2[-1]]
            val_error = val_error + val_error_bi

        '''
        #selection stage
        if self.mf_max_iters !=0:
            best_choice = val_error.index(min(val_error))

        if best_choice == 0:
            self.final_gam_model = self.gami_model
            self.final_mf_model = None
            print('select best model: stop at gami model' )
            return

        else:
            self.final_gam_model = self.gami_model
            self.final_mf_model = self.lv_model
            print('select best model: stop at lv model')
        '''
        self.final_gam_model = self.gami_model
        self.final_mf_model = self.lv_model

        self.cur_rank = self.final_mf_model.cur_rank
        self.match_i = self.final_mf_model.match_i
        self.match_u = self.final_mf_model.match_u
        self.var_u = self.final_mf_model.var_u
        self.var_i = self.final_mf_model.var_i
        
        self.s = np.diag(self.final_mf_model.s)
        self.u = self.final_mf_model.u
        self.v = self.final_mf_model.v.T


    def predict(self,xx,Xi):

        if self.mf_max_iters == 0 or self.final_mf_model==None:

            pred = self.final_gam_model.predict(xx)

            return pred

        else:
            
            pred1 = self.final_gam_model.predict(xx)
            
            pred2 = []
            for i in range(Xi.shape[0]):
                if Xi[i,0] == 'cold':
                    g = self.u_group_model.predict(xx[i,self.user_feature_list].reshape(1,-1))[0]
                    group_pre_u = self.final_mf_model.pre_u
                    for i in range(1000):
                        if g in list(group_pre_u.keys()):
                            g = group_pre_u[g]
                        else:
                            break
                    u = self.match_u[g]
                    #u=np.zeros(u.shape)
                else:
                    u = self.u[int(Xi[i,0])]
                if Xi[i,1] == 'cold':
                    g = self.i_group_model.predict(xx[i,self.item_feature_list].reshape(1,-1))[0]
                    group_pre_i = self.final_mf_model.pre_i
                    for i in range(1000):
                        if g in list(group_pre_i.keys()):
                            g = group_pre_i[g]
                        else:
                            break
                    v = self.match_i[g]
                    #v =np.zeros(v.shape)
                else:
                    v =self.v[int(Xi[i,1])]

                pred_mf = np.dot(u, np.multiply(self.s, v))
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

    def linear_global_explain(self):
        self.final_gam_model.global_explain(folder="./results", name="demo", cols_per_row=3, main_density=3, save_png=False, save_eps=False)

    def local_explain(self,class_,ex_idx,xx,Xi,y,simu_dir = 'result'):


        mf_output = self.final_mf_model.predict(Xi[ex_idx].reshape(1,-1))
        data_dict_local = self.final_gam_model.local_explain(class_ ,mf_output,xx[[ex_idx],:], y[[ex_idx],:],save_dict=False)
        return data_dict_local



    def mf_distance(self,threshold,u_i='user'):

        if u_i == 'user':
            user = self.match_u
            dis,closest = self.get_distance(user)

            self.draw_net(dis,closest,threshold)                      

        elif u_i == 'item':
            item = self.match_i
            dis,closest = self.get_distance(item)
            self.draw_net(dis,closest,threshold)



    def main_effect_cluster(self,x,group_num):



        gmm_pu = KMeans(group_num,n_jobs=-1).fit(x) 
        labels = gmm_pu.predict(x)

        return labels, gmm_pu

    def get_distance(self,x):
        dis = {}
        closest = {}
        #group = {}
        adjusted = np.mean(np.array(list(x.values())),axis=0)
        for i in x.keys():
            x[i] = x[i] - adjusted
        for i in x.keys():            
            sim = []
            for j in x.keys():
                sim.append(cosine_similarity(x[i].reshape(1,-1),x[j].reshape(1,-1))[0][0])
            sim = np.array(sim)
            similarity = np.concatenate([np.array([np.array(list(x.keys()))]).T,abs(sim).reshape(-1,1)],axis=1).T
            #sorted_sim = similarity.T[np.lexsort(similarity)][1:,:]
            sorted_sim = similarity.T[np.lexsort(similarity)][:-1,:]
            dis[i] = sorted_sim
            closest[i] = similarity.T[np.lexsort(similarity)][-2,0]
            #group[i] = similarity.T[np.lexsort(similarity)][:-1,:]

        return dis,closest

    def draw_net(self, dis, closest, threshold):

        group ={}
        for i in dis:
            group[i] = dis[i][dis[i][:,1]>threshold]

        G = nx.Graph()
        for i in dis.keys():
            G.add_node(i)
        for i in dis.keys():
            for j in dis.keys():
                if i != j:
                    G.add_edge(i,dis[i][dis[i][:,0]==j][0][0],weight=np.round(dis[i][dis[i][:,0]==j][0][1],3))

        pos = nx.spring_layout(G,iterations=1000)
        n_labels = {} 
        e_labels = nx.get_edge_attributes(G,'weight')
        draw_edge= {}

        for i,j in e_labels:
            if group[i].shape!=(0,2):
                for k in group[i]:
                    try:
                        draw_edge[i,int(k[0])] = e_labels[i,int(k[0])]
                    except:
                        draw_edge[i,int(k[0])] = e_labels[int(k[0]),i]

        for node in G.nodes():

            n_labels[node] = node

        nx.draw_networkx_nodes(G,pos,node_size=300,node_color='r',label=n_labels)
        nx.draw_networkx_labels(G,pos,n_labels,font_size=10,font_color='black')
        nx.draw_networkx_edges(G,pos,edgelist=draw_edge,width=2)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=draw_edge)
        nx.draw_networkx_edges(G,pos,edgelist=e_labels,width=1,alpha=0.5,edge_color='b',style='dashed')


    def cold_start_analysis(self,x,u_i,confi):
        if u_i == 'user':
            group = self.u_group_model.predict(x)
            mean_g = self.match_u[group[0]]
            var_g = self.var_u[group[0]]**0.5

        if u_i == 'item':
            group = self.i_group_model.predict(x)
            mean_g = self.match_i[group[0]]
            var_g = self.var_i[group[0]]**0.5

        upper = mean_g + confi * var_g
        lower = mean_g - confi * var_g
        
        print('The new '+u_i+' belong to group '+str(group)+'\n mean is '+str(mean_g)+' and std is '+ str(var_g)+
        '\n the confidence interval is ['+str(lower)+','+str(upper)+']')

        return mean_g, var_g, upper, lower








