# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:56:00 2020

@author: suyu
"""
import os
import time
import numpy as np
import tensorflow as tf
from .LI import LatentVariable
from .gaminet import GAMINet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans 
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from .utils import global_visualize_density
from itertools import product

class GAMMLI:
    """
    Generalized Addtive Model with Manifest and Latent Interactions
    
    :param dict model_info: model basic information.
    :param array subnet_arch: subnetwork architecture.
    :param array interact_arch: interact subnetwork architecture.
    :param func activation_func: activation_function.
    :param float lr_bp: learning rate.
    :param float loss_threshold_main: main_effect loss threshold.
    :param float loss_threshold_inter: interact_effect loss threshold.
    :param int main_grid_size: number of the sampling points for main_effect training..
    :param int interact_grid_size: number of the sampling points for interact_effect training.
    :param int batch_size: size of batch.
    :param int main_effect_epochs: main effect training stage epochs.
    :param int tuning_epochs: tuning stage epochs.
    :param int interaction_epochs: interact effect training stage epochs.
    :param int interact_num: the max interact pair number.
    :param str interaction_restrict: interaction restrict settings.
    :param int early_stop_thres: epoch for starting the early stop.
    :param float convergence_threshold: convergence threshold for latent effect training.
    :param int mf_training_iters: latent effect training stage epochs.
    :param int max_rank: max rank for the latent variable.
    :param bool change_mode: whether change the initial value for latent effect training.
    :param int u_group_num: number of user group.
    :param int i_group_num: number of item group.
    :param float scale_ratio: group range shrinkage ratio.
    :param float combine_range: group combination range.
    :param bool auto_tune: whether auto tune the hyperparameter.
    :param int random_state: number of user group.
    :param str wc: build model for 'warm start' or 'cold start'


    """
    def __init__(self,
                 meta_info=None,
                 model_info=None,
                 subnet_arch=[10, 6],
                 interact_arch=[20, 10],
                 activation_func=tf.tanh,
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
                 interaction_restrict=None,
                 verbose=False,
                 early_stop_thres=100,
                 shrinkage_value=None,
                 convergence_threshold=0.001,
                 mf_training_iters=20,
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
                 auto_tune=False,
                 random_state = 0,
                 combine_range=0.99,
                 wc = None):

        super(GAMMLI, self).__init__()

        self.meta_info = meta_info
        self.model_info = model_info
        
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
        self.interaction_restrict = interaction_restrict

        self.verbose = verbose
        self.early_stop_thres = early_stop_thres

        self.fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.mf_max_iters = mf_training_iters
        self.max_rank = max_rank
        self.change_mode =change_mode
        self.n_power_iterations = n_power_iterations
        self.n_oversamples = n_oversamples

        self.multi_type_num = multi_type_num
        self.u_group_num = u_group_num
        self.i_group_num = i_group_num
        self.scale_ratio = scale_ratio
        self.auto_tune = auto_tune
        self.random_state = random_state
        self.combine_range = combine_range
        self.wc = wc


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
    def fit(self,tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx):
        
        """
        Build a GAMMLI model from the dataset (tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx).

        :param array tr_x: explict effect feature in training set.
        :param array val_x: explict effect feature in validation set.
        :param array tr_y: target variable in training set.
        :param array val_y: target variable in validation set.
        :param array tr_Xi: implicit effect feature in training set.
        :param array val_Xi: implicit effect feature in validation set.
        :param array tr_idx: training set index.
        :param array tr_idx: validation set index.
        
        :return: fitted GAMMLI model
        """
        def clip(x):
            y=deepcopy(x)
            for i in range(y.shape[0]):
                if y[i] == 0:
                    y[i] = 0.0001
                elif y[i] == 1:
                    y[i] = 0.9999
            return y

        #initial cluster training
        self.user_feature_list = []
        self.item_feature_list = []
        for indice, (feature_name, feature_info) in enumerate(self.meta_info.items()):
            if feature_info["source"] == "user":
                self.user_feature_list.append(indice)
            elif feature_info["source"] == "item":
                self.item_feature_list.append(indice)


        user_feature = np.concatenate([tr_x[:,self.user_feature_list],tr_Xi[:,0].reshape(-1,1)],1)
        item_feature = np.concatenate([tr_x[:,self.item_feature_list],tr_Xi[:,1].reshape(-1,1)],1)
        user_feature = np.unique(user_feature,axis=0)
        item_feature = np.unique(item_feature,axis=0)
        user_feature = user_feature[np.argsort(user_feature[:,-1])]
        item_feature = item_feature[np.argsort(item_feature[:,-1])]
        self.user_id = user_feature[:,-1]
        self.item_id = item_feature[:,-1]
        user_feature = user_feature[:,:-1]
        item_feature = item_feature[:,:-1]
        
        val_user_feature = np.concatenate([val_x[:,self.user_feature_list],val_Xi[:,0].reshape(-1,1)],1)
        val_item_feature = np.concatenate([val_x[:,self.item_feature_list],val_Xi[:,1].reshape(-1,1)],1)
        val_user_feature = np.unique(val_user_feature,axis=0)
        val_item_feature = np.unique(val_item_feature,axis=0)
        val_user_feature = val_user_feature[np.argsort(val_user_feature[:,-1])]
        val_item_feature = val_item_feature[np.argsort(val_item_feature[:,-1])]
        val_user_feature = val_user_feature[:,:-1]
        val_item_feature = val_item_feature[:,:-1]

        if self.u_group_num != 0:
            self.u_group, self.u_group_model = self.main_effect_cluster(user_feature,self.u_group_num)
            self.val_u_group = self.u_group_model.predict(val_user_feature)
        else:
            self.u_group=0
        if self.i_group_num != 0:
            self.i_group, self.i_group_model = self.main_effect_cluster(item_feature,self.i_group_num)
            self.val_i_group = self.i_group_model.predict(val_item_feature)
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
                                  verbose=self.verbose, early_stop_thres=self.early_stop_thres,random_state=self.random_state,interaction_restrict=self.interaction_restrict)


        model = self.gami_model
        st_time = time.time()

        model.fit(tr_x, val_x, tr_y, val_y, tr_idx, val_idx)

        fi_time = time.time()
        print('time cost:',fi_time-st_time)

        pred_train = model.predict(tr_x)
        pred_val = model.predict(val_x)   
        error1.append(self.loss_fn(tr_y.ravel(),pred_train.ravel()).numpy())
        val_error1.append(self.loss_fn(val_y.ravel(),pred_val.ravel()).numpy())
        if self.task_type == 'Classification':
            pred_train_initial = model.predict_initial(tr_x).numpy()
            pred_train_initial[pred_train_initial>np.log(9999)] = np.log(9999)
            pred_train_initial[pred_train_initial<np.log(1/9999)] = np.log(1/9999)
            pred_val_initial = model.predict_initial(val_x).numpy()
            pred_val_initial[pred_val_initial>np.log(9999)] = np.log(9999)
            pred_val_initial[pred_val_initial<np.log(1/9999)] = np.log(1/9999)

        print('After the gam stage, training error is %0.5f , validation error is %0.5f' %(error1[-1],val_error1[-1]))
        if self.task_type == 'Regression':
            residual = (tr_y.ravel() - pred_train.ravel()).reshape(-1,1)
            residual_val = (val_y.ravel() - pred_val.ravel()).reshape(-1,1)
        elif self.task_type == 'Classification':
            residual = (np.log(clip(tr_y.ravel())/(1-clip(tr_y.ravel()))) - pred_train_initial.ravel()).reshape(-1,1)
            residual_val = (np.log(clip(val_y.ravel())/(1-clip(val_y.ravel()))) - pred_val_initial.ravel()).reshape(-1,1)

        #mf fit
        if self.mf_max_iters !=0:
            
            if self.task_type == 'Classification':
                pred_train = pred_train_initial
                pred_val = pred_val_initial
            self.lv_model = LatentVariable(verbose = self.verbose,task_type=self.task_type,max_rank=self.max_rank,max_iters=self.mf_max_iters,
                                           change_mode=self.change_mode,auto_tune=self.auto_tune,
                                           convergence_threshold=self.convergence_threshold,n_oversamples=self.n_oversamples
                                           ,u_group = self.u_group,i_group = self.i_group,val_u_group = self.val_u_group,val_i_group = self.val_i_group
                                           ,scale_ratio=self.scale_ratio,pred_tr=pred_train,shrinkage_value=self.shrinkage_value,
                                           tr_y=tr_y,pred_val=pred_val,val_y=val_y, tr_Xi=tr_Xi,val_Xi=val_Xi,random_state=self.random_state
                                           ,combine_range=self.combine_range, wc = self.wc)
            model1 = self.lv_model
            st_time = time.time()
            model1.fit(tr_Xi,val_Xi,residual,residual_val,self.ui_shape)
            fi_time = time.time()
            print('time cost:',fi_time-st_time)

            pred = model1.predict(tr_Xi)
            predval = model1.predict(val_Xi)
            if self.task_type == 'Classification':
                error2.append(self.loss_fn(tr_y.ravel(),tf.sigmoid(pred.ravel()+pred_train.ravel()).numpy()).numpy())
                val_error2.append(self.loss_fn(val_y.ravel(),tf.sigmoid(predval.ravel()+pred_val.ravel()).numpy()).numpy())
            else:
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
            if self.task_type == 'Classification':
                pred1 = self.final_gam_model.predict_initial(xx).numpy()
            
            pred2 = []
            for i in range(Xi.shape[0]):
                if Xi[i,0] == 'cold':
                    g = self.u_group_model.predict(xx[i,self.user_feature_list].reshape(1,-1))[0]
                    group_pre_u = self.final_mf_model.pre_u
                    g = self.new_group(g,group_pre_u)
                    u = self.match_u[g]
                    #u=np.zeros(u.shape)
                else:
                    u = self.u[int(Xi[i,0])]
                if Xi[i,1] == 'cold':
                    g = self.i_group_model.predict(xx[i,self.item_feature_list].reshape(1,-1))[0]
                    group_pre_i = self.final_mf_model.pre_i
                    g = self.new_group(g,group_pre_i)
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
                pred = tf.sigmoid(pred).numpy()


            return pred

    def new_group(self,g, group_pre):
        for i in range(1000):
            if g in list(group_pre.keys()):
                g = group_pre[g]
            else:
                break
        return g
    
    def group_back(self,g,group_pre):
        kid_group = []
        pre = dict(zip(group_pre.values(), group_pre.keys()))
        for i in range(1000):
            if g in list(pre.keys()):
                g = pre[g]
                kid_group.append(g)
            else:
                break
        return kid_group
    
    def linear_global_explain(self):
        self.final_gam_model.global_explain(folder="./results", name="demo", cols_per_row=3, main_density=3, save_png=False, save_eps=False, threshold=0.02)

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
    
    def group_explain(self,g, u_i):
    
        def group_num(u, group):
            num = 0
            contain_id = []
            for j in range(group.shape[0]):
                if u == group[j]:
                    contain_id.append(j)
                    num += 1
            return num, contain_id
        
        if u_i == 'user':
            kid_group = self.group_back(g, self.final_mf_model.pre_u)
            num, contain_id = group_num(g,self.u_group)
            contain_id = self.user_id[contain_id]
            mean_g = self.match_u[g]
            std_g = self.var_u[g]**0.5
            print('kid group:',kid_group)
            print('contain users:',num)
            print('mean :',mean_g)
            print('std :',std_g)
            v=self.v
            preference = []
            for ex_item in range(v.shape[0]):
                pred_mf = np.dot(mean_g, np.multiply(self.s, v[ex_item,:]))
                preference.append(pred_mf)

            #数据
            
            name=['ind_cco_fin_ult1',
                  'ind_cder_fin_ult1',
                  'ind_cno_fin_ult1',
                  'ind_ctju_fin_ult1',
                  'ind_ctma_fin_ult1',
                  'ind_ctop_fin_ult1',
                  'ind_ctpp_fin_ult1',
                  'ind_deco_fin_ult1',
                  'ind_deme_fin_ult1',
                  'ind_dela_fin_ult1',
                  'ind_ecue_fin_ult1',
                  'ind_fond_fin_ult1',
                  'ind_hip_fin_ult1',
                  'ind_plan_fin_ult1',
                  'ind_pres_fin_ult1',
                  'ind_reca_fin_ult1',
                  'ind_tjcr_fin_ult1',
                  'ind_valo_fin_ult1',
                  'ind_viv_fin_ult1',
                  'ind_nomina_ult1',
                  'ind_nom_pens_ult1',
                  'ind_recibo_ult1']
            score=preference

            #图像绘制
            fig,ax=plt.subplots()
            b=ax.barh(range(len(name)),score,color='#6699CC')

            #设置Y轴刻度线标签
            ax.set_yticks(range(len(name)))
            #font=FontProperties(fname=r'/Library/Fonts/Songti.ttc')
            ax.set_yticklabels(name)

            plt.show()
            
            
        elif u_i == 'item':
            
            kid_group = self.group_back(g, self.final_mf_model.pre_i)
            num, contain_id = group_num(g,self.i_group)
            contain_id = self.item_id[contain_id]
            mean_g = self.match_i[g]
            std_g = self.var_i[g]**0.5
            
            
    def digraph(self, target, related_width):
    
        def group_num(u, group):
            num = 0
            contain_id = []
            for j in range(group.shape[0]):
                if u == group[j]:
                    contain_id.append(j)
                    num += 1
            return num, contain_id
        
        
        edges = {}
        u_node=[]
        i_node=[]
        if target == 'implicit':
        
            avg =0
            for i in self.match_u.keys():
                user_g = self.match_u[i]
                u_node.append('u'+str(i))
                for j in self.match_i.keys():
                    item_g = self.match_i[j]
                    pred_mf = np.dot(user_g, np.multiply(self.s, item_g))
                    edges[('u'+str(i),'i'+str(j))] = pred_mf
                    avg = avg+pred_mf
            avg = avg/(len(self.match_u)+len(self.match_i))
            for j in self.match_i.keys():
                i_node.append('i'+str(j))
                    
        if target == 'explicit':
            return
                    
        G = nx.DiGraph()
        n_labels = {} 
        for i in u_node + i_node:
            G.add_node(i)
        for node in G.nodes():
            n_labels[node] = node
                
        pos={}
        for i in range(len(u_node)):
            un = u_node[i]
            step = 2/len(u_node)
            pos[un] = np.array([-1,1-step*i])
                
        for i in range(len(i_node)):
            un = i_node[i]
            step = 2/len(i_node)
            pos[un] = np.array([1,1-step*i])
            
        for i,j in edges.items():
            G.add_edges_from([i], weight=j)
        
        e_labels = nx.get_edge_attributes(G,'weight')
        showedge = deepcopy(e_labels)
        for i,j in e_labels.items():
            if (j < avg) or (j<0):
                showedge.pop(i)

        nx.draw_networkx_nodes(G,pos,node_size=300,node_color='r',label=n_labels)
        nx.draw_networkx_labels(G,pos,n_labels,font_size=10,font_color='black')
        nx.draw_networkx_edges(G,pos,edgelist=showedge,width=(np.array(list(showedge.values()),dtype=np.float)*related_width).tolist(),arrows=True)
        #nx.draw_networkx_edges(G,pos,edgelist=G.edges,width=1,alpha=0.5,edge_color='b',style='dashed')
            
            

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


    def cold_start_analysis(self,xx,u_i,confi):
        
        if u_i == 'user':
            g = self.u_group_model.predict(xx[:,self.user_feature_list].reshape(1,-1))[0]
            group_pre_u = self.final_mf_model.pre_u
            g = self.new_group(g,group_pre_u)
            mean_g = self.match_u[g]
            std_g = self.var_u[g]**0.5

        if u_i == 'item':
            g = self.i_group_model.predict(xx[:,self.item_feature_list].reshape(1,-1))[0]
            group_pre_i = self.final_mf_model.pre_i
            g = self.new_group(g,group_pre_i)
            mean_g = self.match_i[g]
            std_g = self.var_i[g]**0.5

        upper = mean_g + confi * std_g
        lower = mean_g - confi * std_g
        
        print('The new '+u_i+' belong to group '+str(g)+'\n mean is '+str(mean_g)+'\n and std is '+ str(std_g)+
        '\n the confidence interval is ['+str(lower)+','+str(upper)+']')

        return mean_g, std_g, upper, lower
    
    def dash_board(self,data_dict, importance, simu_dir, save_eps=True):
        
        im_list = importance.tolist()
        for i,j in data_dict.items():
            importance_ = im_list.pop(0)
            if data_dict[i]['importance'] !=0:
                data_dict[i]['importance'] = importance_
                
        result = {}
        name_i = {}
        name_u = {}
        #imp = self.final_mf_model.s
        #impp = imp/ imp.sum()
        impp = im_list
        una = np.array(list(self.match_u.keys()))
        moddd_hor = []
        moddd_ver = []
        for i in range(3):
            u1=np.array(list(self.match_u.values()))[:,i]
            i1=np.array(list(self.match_i.values()))[:,i]
    
            ina = np.array(list(self.match_i.keys()))
            una = np.array(list(self.match_u.keys()))
    
            res = []
            for a,b in product(u1, i1):
                res.append(a*b)
            res = np.array(res).reshape(u1.shape[0],-1)
    
            modd_hor = []
            modd_ver = []
            for j in range(res.shape[1]):
                modd_hor.append(np.linalg.norm(res[:,j]))
            for j in range(res.shape[0]):
                modd_ver.append(np.linalg.norm(res[j,:]))            
            modd_hor=np.array(modd_hor)*impp[i]
            modd_ver=np.array(modd_ver)*impp[i]
            moddd_hor.append(modd_hor)
            moddd_ver.append(modd_ver)
            result[i] = res
            name_i[i] = ina
            name_u[i] = una
            
        moddd_hor = moddd_hor[0] + moddd_hor[1]+ moddd_hor[2]
        moddd_ver = moddd_ver[0] + moddd_ver[1]+ moddd_ver[2]
        for i in range(3):
            result[i] = result[i][:,np.argsort(moddd_hor)[::-1]]
            result[i] = result[i][np.argsort(moddd_ver)[::-1],:]
            name_i[i] = name_i[i][np.argsort(moddd_hor)[::-1]]
            name_u[i] = name_u[i][np.argsort(moddd_ver)[::-1]]
        
        global_visualize_density(data_dict, save_png=False,save_eps=save_eps, folder=simu_dir, name='s1_global')

        left_x,left_y=0.5,0.1
        width,height=0.8,0.5
        left_xh=left_x+width+0.1
        left_xhh=left_xh+width+0.1
        #left_yh = left_y + height + 0.1

        scatter_area=[left_x,left_y,1,height]
        hist_y=[left_xh,left_y,1,height]
        hist_yy = [left_xhh,left_y,1,height]
        #hist_g = [left_xh,left_yh,1,height]

        plt.figure(figsize=(6, round(14 * 0.45)))
        plt.suptitle('Latent Interactions',fontsize=14,x=1.9,y=0.7)

        area_1=plt.axes(scatter_area)
        area_2=plt.axes(hist_y)
        area_3=plt.axes(hist_yy)

        ax1 = area_1.matshow(result[0])
        area_1.xaxis.set_ticks_position('top') #设置x轴刻度到上方
        area_1.set_xticks(np.arange(result[0].shape[1])) #设置x轴刻度
        area_1.set_yticks(np.arange(result[0].shape[0])) #设置y轴刻度
        area_1.set_xticklabels(name_i[0]) #设置x轴刻度标签
        area_1.set_yticklabels(name_u[0]) #设置y轴刻度标签
        area_1.set_title("%.2f%%" % (impp[0] * 100),y=-0.2) #设置标题以及其位置和字体大小 

        ax2 = area_2.matshow(result[1])
        area_2.xaxis.set_ticks_position('top') #设置x轴刻度到上方
        area_2.set_xticks(np.arange(result[1].shape[1])) #设置x轴刻度
        area_2.set_yticks(np.arange(result[1].shape[0])) #设置y轴刻度
        area_2.set_xticklabels(name_i[1]) #设置x轴刻度标签
        area_2.set_yticklabels(name_u[1]) #设置y轴刻度标签
        area_2.set_title("%.2f%%" % (impp[1] * 100),y=-0.2) #设置标题以及其位置和字体大小 

        ax3 = area_3.matshow(result[2])
        area_3.xaxis.set_ticks_position('top') #设置x轴刻度到上方
        area_3.set_xticks(np.arange(result[2].shape[1])) #设置x轴刻度
        area_3.set_yticks(np.arange(result[2].shape[0])) #设置y轴刻度
        area_3.set_xticklabels(name_i[2]) #设置x轴刻度标签
        area_3.set_yticklabels(name_u[1]) #设置y轴刻度标签
        area_3.set_title("%.2f%%" % (impp[2] * 100),y=-0.2) #设置标题以及其位置和字体大小 
        plt.colorbar(ax1,ax=area_1)
        plt.colorbar(ax2,ax=area_2)
        plt.colorbar(ax3,ax=area_3)
        
        if save_eps:
            plt.savefig("heatmap.eps", bbox_inches="tight", dpi=100)
        
        
        
    def get_all_rank(self,Xi):
        beta, gamma = self.final_gam_model.get_component()
        delta = np.array(self.final_mf_model.rank_norm(Xi)).reshape(-1,1)
        componment_coefs= np.vstack([beta, gamma, delta])
        
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        
        return componment_scales
    
    def relation_plot(self, similarity_thres, draw_s, save_eps):
        plt.figure(figsize=(15, 5))
        plt.suptitle('Latent Relation',fontsize=14)
        
        ax1=plt.subplot(131)
        ax1.set_title('user_relation')
        self.mf_distance(similarity_thres, 'user')
        
        ax2=plt.subplot(132)
        ax2.set_title('item_relation')
        self.mf_distance(similarity_thres, 'item')
        
        ax3=plt.subplot(133)
        ax3.set_title('user_item_relation')
        self.digraph('implicit', draw_s)
        
        if save_eps:
            plt.savefig("relation.eps", bbox_inches="tight", dpi=100)
        
        








