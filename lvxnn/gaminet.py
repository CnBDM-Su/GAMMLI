import os
import numpy as np
import pandas as pd 
import tensorflow as tf
from scipy import stats
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

from .layers import *
from .utils import get_interaction_list


class GAMINet(tf.keras.Model):

    def __init__(self, meta_info,
                 subnet_arch=[20, 10],
                 interact_num=10,
                 interact_arch=[20, 10],
                 task_type="Regression",
                 activation_func=tf.tanh,
                 main_grid_size=41,
                 interact_grid_size=41,
                 lr_bp=0.001,
                 batch_size=500,
                 main_effect_epochs=2000,
                 interaction_epochs=2000,
                 tuning_epochs=50,
                 loss_threshold_main=0.01,
                 loss_threshold_inter=0.01,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=100,
                 random_state=0,
                 threshold =0.5,
                 multi_type_num=0):

        super(GAMINet, self).__init__()
        # Parameter initiation
        self.meta_info = meta_info
        self.input_num = len(meta_info) - 1
        
        self.task_type = task_type
        self.subnet_arch = subnet_arch
        self.main_grid_size = main_grid_size
        self.interact_grid_size = interact_grid_size
        self.activation_func = activation_func
        self.interact_arch = interact_arch
        self.max_interact_num = int(round(self.input_num * (self.input_num - 1) / 2))
        self.interact_num = min(interact_num, self.max_interact_num)
        self.interact_num_added = 0
        self.interaction_list = []
        self.loss_threshold_main = loss_threshold_main
        self.loss_threshold_inter = loss_threshold_inter
        
        self.lr_bp = lr_bp
        self.batch_size = batch_size
        self.tuning_epochs = tuning_epochs
        self.main_effect_epochs = main_effect_epochs
        self.interaction_epochs = interaction_epochs

        self.verbose = verbose
        self.val_ratio = val_ratio
        self.early_stop_thres = early_stop_thres
        self.random_state = random_state
        self.threshold = threshold
        
        self.multi_type_num = multi_type_num
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.categ_variable_num = 0
        self.numerical_input_num = 0
        self.categ_variable_list = []
        self.categ_index_list = []
        self.numerical_index_list = []
        self.numerical_variable_list = []
        self.variables_names = []
        self.feature_type_list = []
        self.interaction_status = False
        for indice, (feature_name, feature_info) in enumerate(self.meta_info.items()):
            if feature_info["type"] == "target":
                continue
            elif feature_info["type"] == "categorical":
                self.categ_variable_num += 1
                self.categ_index_list.append(indice)
                self.feature_type_list.append("categorical")
                self.categ_variable_list.append(feature_name)
            elif feature_info["type"] == "id":
                continue
            else:
                self.numerical_input_num +=1
                self.numerical_index_list.append(indice)
                self.feature_type_list.append("continuous")
                self.numerical_variable_list.append(feature_name)
            self.variables_names.append(feature_name)
        print(self.variables_names)
            
        # build
        self.maineffect_blocks = MainEffectBlock(meta_info=self.meta_info,
                                 numerical_index_list=list(self.numerical_index_list),
                                 categ_index_list=self.categ_index_list,
                                 subnet_arch=self.subnet_arch,
                                 activation_func=self.activation_func,
                                 grid_size=self.main_grid_size)
        self.interact_blocks = InteractionBlock(interact_num=self.interact_num,
                                meta_info=self.meta_info,
                                interact_arch=self.interact_arch,
                                activation_func=self.activation_func,
                                grid_size=self.interact_grid_size)
        self.output_layer = OutputLayer(input_num=self.input_num,
                                interact_num=self.interact_num,
                                task_type=self.task_type,
                                multi_type_num = self.multi_type_num)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_bp)
        if self.task_type == "Regression":
            #self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif self.task_type == "MultiClassification":
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()        
        elif self.task_type == "Ordinal_Regression":
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        else:
            print(self.task_type)
            raise ValueError("The task type is not supported")

    def call(self, inputs, main_effect_training=False, interaction_training=False):

        self.maineffect_outputs = self.maineffect_blocks(inputs, training=main_effect_training)
        if self.interaction_status:
            self.interact_outputs = self.interact_blocks(inputs, training=interaction_training)
        else:
            self.interact_outputs = tf.zeros([inputs.shape[0], self.interact_num])

        concat_list = [self.maineffect_outputs]
        if self.interact_num > 0:
            concat_list.append(self.interact_outputs)

        if self.task_type == "Regression":
            output = self.output_layer(tf.concat(concat_list, 1))
        elif self.task_type == "Classification":
            output = tf.nn.sigmoid(self.output_layer(tf.concat(concat_list, 1)))
        elif self.task_type == "Ordinal_Regression":
            output = tf.nn.sigmoid(self.output_layer(tf.concat(concat_list, 1)))
        elif self.task_type == "MultiClassification":
            output = tf.nn.softmax(self.output_layer(tf.concat(concat_list, 1)))
        else:
            raise ValueError("The task type is not supported")

        return output

    @tf.function
    def predict_graph(self, x, main_effect_training=False, interaction_training=False):
        return self.__call__(tf.cast(x, tf.float32), 
                      main_effect_training=main_effect_training,
                      interaction_training=interaction_training)

    def predict(self, x):
        if self.task_type == "Ordinal_Regression":
            ind = self.scan(self.predict_graph(x).numpy(),self.threshold)
            return tf.keras.backend.eval(ind)
        if self.task_type == "MultiClassification":
            ind = tf.argmax(self.predict_graph(x).numpy(),axis=1)
            return tf.keras.backend.eval(ind)
        return self.predict_graph(x).numpy()

    @tf.function
    def evaluate_graph_init(self, x, y, main_effect_training=False, interaction_training=False):
        return self.loss_fn(y, self.__call__(tf.cast(x, tf.float32),
                               main_effect_training=main_effect_training,
                               interaction_training=interaction_training))

    @tf.function
    def evaluate_graph_inter(self, x, y, main_effect_training=False, interaction_training=False):
        return self.loss_fn(y, self.__call__(tf.cast(x, tf.float32),
                               main_effect_training=main_effect_training,
                               interaction_training=interaction_training))

    def evaluate(self, x, y, main_effect_training=False, interaction_training=False):
        if self.interaction_status:
            return self.evaluate_graph_inter(x, y,
                                  main_effect_training=main_effect_training,
                                  interaction_training=interaction_training).numpy()
        else:
            return self.evaluate_graph_init(x, y,
                                  main_effect_training=main_effect_training,
                                  interaction_training=interaction_training).numpy()

    @tf.function
    def train_main_effect(self, inputs, labels, main_effect_training=True, interaction_training=False):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, main_effect_training=main_effect_training,
                          interaction_training=interaction_training)
            total_loss = self.loss_fn(labels, pred)


        if self.task_type == "Ordinal_Regression":
            train_weights = self.maineffect_blocks.weights
            train_weights.append(self.output_layer.main_effect_weights)            
            train_weights.append(self.output_layer.ordinal_bias)
        else:
            train_weights = self.maineffect_blocks.weights
            train_weights.append(self.output_layer.main_effect_weights)
            train_weights.append(self.output_layer.main_effect_output_bias)
        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_interaction(self, inputs, labels, main_effect_training=False, interaction_training=True):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, main_effect_training=main_effect_training,
                          interaction_training=interaction_training)
            total_loss = self.loss_fn(labels, pred)


        if self.task_type == "Ordinal_Regression":
            train_weights = self.interact_blocks.weights
            train_weights.append(self.output_layer.interaction_weights)
            train_weights.append(self.output_layer.interaction_output_bias)
        else:
            train_weights = self.interact_blocks.weights
            train_weights.append(self.output_layer.interaction_weights)
            train_weights.append(self.output_layer.interaction_output_bias)
        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    def get_main_effect_rank(self,j):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.input_num)]
        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights[:,j].numpy() * np.array([main_effect_norm]))
        if np.sum(np.abs(beta)) > 10**(-10):
            componment_scales = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
            sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales
    
    def get_interaction_rank(self,j):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.interact_num_added)]
        if self.interact_num_added > 0:
            interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_added)]
            gamma = (self.output_layer.interaction_weights[:,j].numpy()[:self.interact_num_added] 
                  * np.array([interaction_norm]).reshape([-1, 1]))[0]
            if np.sum(np.abs(gamma)) > 10**(-10):
                componment_scales = (np.abs(gamma) / np.sum(np.abs(gamma))).reshape([-1])
                sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales
    
    def get_all_active_rank(self,class_):

        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights[:,class_].numpy() * np.array([main_effect_norm])
             * self.output_layer.main_effect_switcher[:,class_].numpy()).reshape([-1, 1]) 

        interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_added)]
        gamma = (self.output_layer.interaction_weights[:,class_].numpy()[:self.interact_num_added] 
              * np.array([interaction_norm])
              * self.output_layer.interaction_switcher[:,class_].numpy()[:self.interact_num_added]).reshape([-1, 1]) 
        gamma = np.vstack([gamma, np.zeros((self.interact_num - self.interact_num_added, 1)).reshape([-1, 1]) ]) 

        componment_coefs = np.vstack([beta, gamma])
        if np.sum(np.abs(componment_coefs)) > 10**(-10):
            componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        else:
            componment_scales = [0 for i in range(self.input_num + self.interact_num_added)]
        return componment_scales

    def estimate_density(self, x):
        
        n_samples = x.shape[0]
        self.data_dict_density = {}
        for indice in range(self.input_num):
            feature_name = list(self.variables_names)[indice]
            if indice in self.numerical_index_list:
                sx = self.meta_info[feature_name]["scaler"]
                density, bins = np.histogram(sx.inverse_transform(x[:,[indice]]), bins=10, density=True)
                self.data_dict_density.update({feature_name:{"density":{"names":bins,"scores":density}}})
            elif indice in self.categ_index_list:
                unique, counts = np.unique(x[:, indice], return_counts=True)
                density = np.zeros((len(self.meta_info[feature_name]["values"])))
                density[unique.astype(int)] = counts / n_samples
                self.data_dict_density.update({feature_name:{"density":{"names":np.arange(len(self.meta_info[feature_name]["values"])),
                                                     "scores":density}}})
    def coding(self,y):
        re = np.zeros((y.shape[0],4))
        for i in range(y.shape[0]):
            if y[i]== 1:
                re[i] = np.array([0,0,0,0])
            elif y[i] ==2:
                re[i] = np.array([1,0,0,0])
            elif y[i] ==3:
                re[i] = np.array([1,1,0,0])
            elif y[i] ==4:
                re[i] = np.array([1,1,1,0])
            elif y[i] ==5:
                re[i] = np.array([1,1,1,1])
        return re
    
    def scan(self, x, threshold):
        res = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            res[i] = 5
            for j in range(x.shape[1]):
                if x[i,j] < threshold:
                    res[i] = j+1
                    break
                #elif j==4:
                #    res[i] = j+1
                #    break
        return res
    
    def fit_main_effect(self, tr_x, tr_y, val_x, val_y):
        
        ## specify grid points
        for i in range(self.input_num):
            if i in self.categ_index_list:
                length = len(self.meta_info[self.variables_names[i]]["values"])
                input_grid = np.arange(len(self.meta_info[self.variables_names[i]]["values"]))
            else:
                length = self.main_grid_size
                input_grid = np.linspace(0, 1, length)
            pdf_grid = np.ones([length]) / length    
            self.maineffect_blocks.subnets[i].set_pdf(np.array(input_grid, dtype=np.float32).reshape([-1, 1]),
                                        np.array(pdf_grid, dtype=np.float32).reshape([1, -1]))

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        for epoch in range(self.main_effect_epochs):
            
            if self.task_type != "Ordinal_Regression":
                shuffle_index = np.arange(tr_x.shape[0])
                np.random.shuffle(shuffle_index)
                tr_x = tr_x[shuffle_index]
                tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_main_effect(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train_main_effect_training.append(self.evaluate(tr_x, tr_y, main_effect_training=False, interaction_training=False))
            self.err_val_main_effect_training.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Main effects training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_main_effect_training[-1], self.err_val_main_effect_training[-1]))

            if self.err_val_main_effect_training[-1] < best_validation:
                best_validation = self.err_val_main_effect_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_main_effect_training[-1]))
                break

    def prune_main_effect(self, val_x, val_y):

        if self.multi_type_num == 0:
            self.main_effect_val_loss = []
            sorted_index, componment_scales = self.get_main_effect_rank(0)        
            self.output_layer.main_effect_switcher.assign(tf.constant(np.zeros((self.input_num, 1)), dtype=tf.float32))
            self.main_effect_val_loss.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False) )
            for idx in range(self.input_num):
                selected_index = sorted_index[:(idx + 1)]
                main_effect_switcher = np.zeros((self.input_num, 1))
                main_effect_switcher[selected_index] = 1
                self.output_layer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))
                val_loss = self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False)
                self.main_effect_val_loss.append(val_loss)

            best_loss = np.min(self.main_effect_val_loss)
            if np.sum((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold_main) > 0:
                best_idx = np.where((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold_main)[0][0]
            else:
                best_idx = np.argmin(self.main_effect_val_loss)
            self.active_main_effect_index = sorted_index[:best_idx]
            main_effect_switcher = np.zeros((self.input_num, 1))
            main_effect_switcher[self.active_main_effect_index] = 1
            self.output_layer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))
        
        else:
            self.active_main_effect_index = []
        
            for i in range(self.multi_type_num):
                tmp1 = self.output_layer.main_effect_switcher.numpy()
                tmp1[:,i] = np.zeros(self.input_num).ravel()
                self.output_layer.main_effect_switcher.assign(tf.constant(tmp1, dtype=tf.float32))
                sorted_index, componment_scales = self.get_main_effect_rank(i)        
                self.main_effect_val_loss = []
                self.main_effect_val_loss.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False) )
                for idx in range(self.input_num):
                    selected_index = sorted_index[:(idx + 1)]
                    main_effect_switcher = np.zeros((self.input_num, 1))
                    main_effect_switcher[selected_index] = 1
                    tmp = self.output_layer.main_effect_switcher.numpy()
                    tmp[:,i] = main_effect_switcher.ravel()
                    self.output_layer.main_effect_switcher.assign(tf.constant(tmp, dtype=tf.float32))
                    val_loss = self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False)
                    self.main_effect_val_loss.append(val_loss)
            
            
                best_loss = np.min(self.main_effect_val_loss)
                if np.sum((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold_main) > 0:
                    best_idx = np.where((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold_main)[0][0]
                else:
                    best_idx = np.argmin(self.main_effect_val_loss)
                self.active_main_effect_index.append(sorted_index[:best_idx])
                main_effect_switcher = np.zeros((self.input_num, 1))
                main_effect_switcher[self.active_main_effect_index[-1].astype(int)] = 1
                tmp2 = self.output_layer.main_effect_switcher.numpy()
                tmp2[:,i] = main_effect_switcher.ravel()
            self.output_layer.main_effect_switcher.assign(tf.constant(tmp2, dtype=tf.float32))

    def fine_tune_main_effect(self, tr_x, tr_y, val_x, val_y):
        
        train_size = tr_x.shape[0]
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_main_effect(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train_main_effect_tuning.append(self.evaluate(tr_x, tr_y, main_effect_training=False, interaction_training=False))
            self.err_val_main_effect_tuning.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Main effects tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_main_effect_tuning[-1], self.err_val_main_effect_tuning[-1]))

    def add_interaction(self, tr_x, tr_y, val_x, val_y):
        
        tr_pred = self.__call__(tf.cast(tr_x, tf.float32), main_effect_training=False, interaction_training=False).numpy().astype(np.float64)
        val_pred = self.__call__(tf.cast(val_x, tf.float32), main_effect_training=False, interaction_training=False).numpy().astype(np.float64)

        if self.multi_type_num == 0:
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                                      tr_pred.ravel(), val_pred.ravel(),
                                      self.variables_names,
                                      self.feature_type_list,
                                      task_type=self.task_type,
                                      active_main_effect_index=self.active_main_effect_index)

            self.interaction_list = interaction_list_all[:self.interact_num]
            self.interact_num_added = len(self.interaction_list)
            interaction_switcher = np.zeros((self.interact_num, 1))
            interaction_switcher[:self.interact_num_added] = 1
            self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
            self.interact_blocks.set_interaction_list(self.interaction_list)            
        
        else:
            active_index_inter = []
            for fe_num in range(self.input_num):
                count_int = 0
                for num in range(self.multi_type_num):
                    if (self.active_main_effect_index[num]==fe_num).sum()==1:
                        count_int = count_int +1
                    if count_int > self.multi_type_num/5:
                        active_index_inter.append(fe_num)
                    
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                                      tr_pred.ravel(), val_pred.ravel(),
                                      self.variables_names,
                                      self.feature_type_list,
                                      task_type=self.task_type,
                                      active_main_effect_index=active_index_inter)

            self.interaction_list = interaction_list_all[:self.interact_num]
            self.interact_num_added = len(self.interaction_list)
            interaction_switcher = np.zeros((self.interact_num, 1))
            interaction_switcher[:self.interact_num_added] = 1
            for i in range(self.multi_type_num):

                tmp = self.output_layer.interaction_switcher.numpy()
                tmp[:,i] = interaction_switcher.ravel()
                self.output_layer.interaction_switcher.assign(tf.constant(tmp, dtype=tf.float32))
                self.interact_blocks.set_interaction_list(self.interaction_list)

    def fit_interaction(self, tr_x, tr_y, val_x, val_y):
        
        # specify grid points
        for interact_id, (idx1, idx2) in enumerate(self.interaction_list):

            feature_name1 = self.variables_names[idx1]
            feature_name2 = self.variables_names[idx2]
            if feature_name1 in self.categ_variable_list:
                length1 = len(self.meta_info[feature_name1]["values"]) 
                length1_grid = np.arange(length1)
            else:
                length1 = self.interact_grid_size
                length1_grid = np.linspace(0, 1, length1)
            if feature_name2 in self.categ_variable_list:
                length2 = len(self.meta_info[feature_name2]["values"]) 
                length2_grid = np.arange(length2)
            else:
                length2 = self.interact_grid_size
                length2_grid = np.linspace(0, 1, length2)

            x1, x2 = np.meshgrid(length1_grid, length2_grid)
            input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
            pdf_grid = np.ones([length1, length2]) / (length1 * length2)
            self.interact_blocks.interacts[interact_id].set_pdf(np.array(input_grid, dtype=np.float32),
                                               np.array(pdf_grid, dtype=np.float32).T)

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        self.interaction_status = True 
        for epoch in range(self.interaction_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_interaction(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train_interaction_training.append(self.evaluate(tr_x, tr_y, main_effect_training=False, interaction_training=False))
            self.err_val_interaction_training.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Interaction training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_interaction_training[-1], self.err_val_interaction_training[-1]))

            if self.err_val_interaction_training[-1] < best_validation:
                best_validation = self.err_val_interaction_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_interaction_training[-1]))
                break

    def prune_interaction(self, val_x, val_y):
        
        if self.multi_type_num ==0:
            self.interaction_val_loss = []
            sorted_index, componment_scales = self.get_interaction_rank(0)        
            self.output_layer.interaction_switcher.assign(tf.constant(np.zeros((self.interact_num, 1)), dtype=tf.float32))
            self.interaction_val_loss.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False))
            for idx in range(self.interact_num_added) :
                selected_index = sorted_index[:(idx + 1)]
                interaction_switcher = np.zeros((self.interact_num, 1))
                interaction_switcher[selected_index] = 1
                self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
                val_loss = self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False)
                self.interaction_val_loss.append(val_loss)

            best_loss = np.min(self.interaction_val_loss)
            if np.sum((self.interaction_val_loss / best_loss - 1) < self.loss_threshold_inter) > 0:
                best_idx = np.where((self.interaction_val_loss / best_loss - 1) < self.loss_threshold_inter)[0][0]
            else:
                best_idx = np.argmin(self.interaction_val_loss)
            self.active_interaction_index = sorted_index[:best_idx]
            interaction_switcher = np.zeros((self.interact_num, 1))
            interaction_switcher[self.active_interaction_index] = 1
            self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
            
        else:
            for i in range(self.multi_type_num):
                self.interaction_val_loss = []
                self.active_interaction_index = []
                sorted_index, componment_scales = self.get_interaction_rank(i)    
                tmp = self.output_layer.interaction_switcher.numpy()
                tmp[:,i] = np.zeros(self.interact_num).ravel()
                self.output_layer.interaction_switcher.assign(tf.constant(tmp, dtype=tf.float32))
                self.interaction_val_loss.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False))
                for idx in range(self.interact_num_added) :
                    selected_index = sorted_index[:(idx + 1)]
                    interaction_switcher = np.zeros((self.interact_num, 1))
                    interaction_switcher[selected_index] = 1
                    tmp1 = self.output_layer.interaction_switcher.numpy()
                    tmp1[:,i] = interaction_switcher.ravel()
                    self.output_layer.interaction_switcher.assign(tf.constant(tmp1, dtype=tf.float32))
                    val_loss = self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False)
                    self.interaction_val_loss.append(val_loss)

                best_loss = np.min(self.interaction_val_loss)
                if np.sum((self.interaction_val_loss / best_loss - 1) < self.loss_threshold_inter) > 0:
                    best_idx = np.where((self.interaction_val_loss / best_loss - 1) < self.loss_threshold_inter)[0][0]
                else:
                    best_idx = np.argmin(self.interaction_val_loss)
                self.active_interaction_index.append(sorted_index[:best_idx])
                interaction_switcher = np.zeros((self.interact_num, 1))
                interaction_switcher[self.active_interaction_index[-1].astype(int)] = 1
                tmp2 = self.output_layer.interaction_switcher.numpy()
                tmp2[:,i] = interaction_switcher.ravel()
            self.output_layer.interaction_switcher.assign(tf.constant(tmp2, dtype=tf.float32))
       
    def fine_tune_interaction(self, tr_x, tr_y, val_x, val_y):
        
        train_size = tr_x.shape[0]
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(train_size)
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_interaction(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train_interaction_tuning.append(self.evaluate(tr_x, tr_y, main_effect_training=False, interaction_training=False))
            self.err_val_interaction_tuning.append(self.evaluate(val_x, val_y, main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Interaction tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_interaction_tuning[-1], self.err_val_interaction_tuning[-1]))

    def fit(self, train_x, train_y):
        
        ## data loading
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)
        if self.task_type == "Regression":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                          random_state=self.random_state)
        elif self.task_type == "Classification":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                      stratify=train_y, random_state=self.random_state)
            
            
        elif self.task_type == "Ordinal_Regression":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                          random_state=self.random_state)
            tr_y_c = self.coding(tr_y)
            idx = np.argsort(tr_y.reshape(1,-1))[0][::-1]
            tr_y = tr_y_c[idx]
            tr_x = tr_x[idx]
            
            val_y = self.coding(val_y)
            
        elif self.task_type == "MultiClassification":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                          random_state=self.random_state)
            enc = OneHotEncoder()
            enc.fit(tr_y.reshape(-1,1))
            tr_y = enc.transform(tr_y.reshape(-1,1)).toarray()
            val_y = enc.transform(val_y.reshape(-1,1)).toarray()

        self.tr_idx = tr_idx
        self.val_idx = val_idx

        ## initialization
        self.data_dict_density = {}
        self.err_train_main_effect_training = []
        self.err_val_main_effect_training = []
        self.err_train_main_effect_tuning = []
        self.err_val_main_effect_tuning = []
        self.err_train_interaction_training = []
        self.err_val_interaction_training = []
        self.err_train_interaction_tuning = []
        self.err_val_interaction_tuning = []
        self.interaction_list = []
        self.active_main_effect_index = []
        self.active_interaction_index = []
        self.main_effect_val_loss = []
        self.interaction_val_loss = []
        
        self.estimate_density(tr_x)
        if self.verbose:
            print("#" * 20 + "GAMI-Net training start." + "#" * 20)
        ## step 1: main effects
        if self.verbose:
            print("#" * 10 + "Stage 1: main effect training start." + "#" * 10)
        self.fit_main_effect(tr_x, tr_y, val_x, val_y)
        if self.verbose:
            print("#" * 10 + "Stage 1: main effect training stop." + "#" * 10)
        
        self.prune_main_effect(val_x, val_y)
        
        if self.multi_type_num ==0:
            if len(self.active_main_effect_index) == 0:
                if self.verbose:
                    print("#" * 10 + "No main effect is selected, training stop." + "#" * 10)
                    return 
            elif len(self.active_main_effect_index) < self.input_num:
                if self.verbose:
                    print(str(self.input_num - len(self.active_main_effect_index)) + " main effects are pruned, start tuning." + "#" * 10)
                self.fine_tune_main_effect(tr_x, tr_y, val_x, val_y)
            else:
                if self.verbose:
                    print("#" * 10 + "No main effect is pruned, the tuning step is skipped." + "#" * 10)
        
        else:
            ame_count1=0
            ame_count2=0
            class_=-1
            for ame_index in self.active_main_effect_index:
                class_ = class_+1
                if len(ame_index) == 0:
                    ame_count1 = ame_count1+1
                    print("#" * 10 + "No main effect is selected for class "+str(class_)+ "#" * 10)
                    continue
        
                elif len(ame_index) < self.input_num:
                    ame_count2 = ame_count2+1
                    if self.verbose:
                        print(str(self.input_num - len(ame_index)) + " main effects are pruned for class "+str(class_) + "#" * 10)
                        continue
            
                else:
                    if self.verbose:
                        print("#" * 10 + "No main effect is pruned, the tuning step is skipped." + "#" * 10)
                        continue
                
            if ame_count1 == self.multi_type_num:
                if self.verbose:
                    print("#" * 10 + "No main effect is selected, training stop." + "#" * 10)
                    return 
            elif ame_count2 >0:
                print("start tuning.")
                self.fine_tune_main_effect(tr_x, tr_y, val_x, val_y)
            
        
        ## step2: interaction
        if self.interact_num == 0:
            if self.verbose:
                print("#" * 10 + "Max interaction is specified to zero, training stop." + "#" * 10)
            return 
        if self.verbose:
            print("#" * 10 + "Stage 2: interaction training start." + "#" * 10)
        self.add_interaction(tr_x, tr_y, val_x, val_y)
        self.fit_interaction(tr_x, tr_y, val_x, val_y)
        if self.verbose:
            print("#" * 10 + "Stage 2: interaction training stop." + "#" * 10)
        self.prune_interaction(val_x, val_y)
        if len(self.active_interaction_index) == 0:
            if self.verbose:
                print("#" * 10 + "No interaction is selected, the model returns to GAM." + "#" * 10)
            self.output_layer.interaction_output_bias.assign(tf.constant(np.zeros([self.multi_type_num+1]).ravel(), dtype=tf.float32))
        elif len(self.active_interaction_index) < len(self.interaction_list):
            if self.verbose:
                print("#" * 10 + str(len(self.interaction_list) - len(self.active_interaction_index))
                      + " interactions are pruned, start tuning." + "#" * 10)
            self.fine_tune_interaction(tr_x, tr_y, val_x, val_y)
        else:
            if self.verbose:
                print("#" * 10 + "No main interaction is pruned, the tuning step is skipped.")
        if self.verbose:
            print("#" * 20 + "GAMI-Net training finished." + "#" * 20)
    
    def summary_logs(self, save_dict=False, folder="./", name="summary_logs"):
    
        data_dict_log = {}
        data_dict_log.update({"err_train_main_effect_training":self.err_train_main_effect_training,
                       "err_val_main_effect_training":self.err_val_main_effect_training,
                       "err_train_main_effect_tuning":self.err_train_main_effect_tuning,
                       "err_val_main_effect_tuning":self.err_val_main_effect_tuning,
                       "err_train_interaction_training":self.err_train_interaction_training,
                       "err_val_interaction_training":self.err_val_interaction_training,
                       "err_train_interaction_tuning":self.err_train_interaction_tuning,
                       "err_val_interaction_tuning":self.err_val_interaction_tuning,
                       "interaction_list":self.interaction_list,
                       "active_main_effect_index":self.active_main_effect_index,
                       "active_interaction_index":self.active_interaction_index,
                       "main_effect_val_loss":self.main_effect_val_loss,
                       "interaction_val_loss":self.interaction_val_loss})
        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_log)
            
        return data_dict_log

    def global_explain(self, class_=0, main_grid_size=None, interact_grid_size=None, save_dict=False, folder="./", name="global_explain"):

        ## By default, we use the same main_grid_size and interact_grid_size as that of the zero mean constraint
        ## Alternatively, we can also specify it manually, e.g., when we want to have the same grid size as EBM (256).        
        if main_grid_size is None:
            main_grid_size = self.main_grid_size
        if interact_grid_size is None:
            interact_grid_size = self.interact_grid_size      

        data_dict_global = self.data_dict_density
        componment_scales = self.get_all_active_rank(class_)
        for indice in range(self.input_num):
            feature_name = list(self.variables_names)[indice]
            subnet = self.maineffect_blocks.subnets[indice]
            if indice in self.numerical_index_list:
                sx = self.meta_info[feature_name]["scaler"]
                main_effect_inputs = np.linspace(0, 1, main_grid_size).reshape([-1, 1])
                main_effect_inputs_original = sx.inverse_transform(main_effect_inputs)
                main_effect_outputs = (self.output_layer.main_effect_weights[:,class_].numpy()[indice]
                            * self.output_layer.main_effect_switcher[:,class_].numpy()[indice]
                            * subnet.__call__(tf.cast(tf.constant(main_effect_inputs), tf.float32)).numpy())
                data_dict_global[feature_name].update({"type":"continuous",
                                      "importance":componment_scales[indice],
                                      "inputs":main_effect_inputs_original.ravel(),
                                      "outputs":main_effect_outputs.ravel()})

            elif indice in self.categ_index_list:
                main_effect_inputs_original = self.meta_info[feature_name]["values"]
                main_effect_inputs = np.arange(len(main_effect_inputs_original)).reshape([-1, 1])
                main_effect_outputs = (self.output_layer.main_effect_weights[:,class_].numpy()[indice]
                            * self.output_layer.main_effect_switcher[:,class_].numpy()[indice]
                            * subnet.__call__(tf.cast(main_effect_inputs, tf.float32)).numpy())
                
                main_effect_input_ticks = (main_effect_inputs.ravel().astype(int) if len(main_effect_inputs_original) <= 6 else 
                              np.linspace(0.1 * len(main_effect_inputs_original), len(main_effect_inputs_original) * 0.9, 4).astype(int))
                main_effect_input_labels = [main_effect_inputs_original[i] for i in main_effect_input_ticks]
                if len("".join(list(map(str, main_effect_input_labels)))) > 30:
                    main_effect_input_labels = [str(main_effect_inputs_original[i])[:4] for i in main_effect_input_ticks]

                data_dict_global[feature_name].update({"type":"categorical",
                                      "importance":componment_scales[indice],
                                      "inputs":main_effect_inputs_original,
                                      "outputs":main_effect_outputs.ravel(),
                                      "input_ticks":main_effect_input_ticks,
                                      "input_labels":main_effect_input_labels})

        for indice in range(self.interact_num_added):
            
            response = []
            inter_net = self.interact_blocks.interacts[indice]
            feature_name1 = self.variables_names[self.interaction_list[indice][0]]
            feature_name2 = self.variables_names[self.interaction_list[indice][1]]
            feature_type1 = "categorical" if feature_name1 in self.categ_variable_list else "continuous"
            feature_type2 = "categorical" if feature_name2 in self.categ_variable_list else "continuous"
            
            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.categ_variable_list:
                interact_input1_original = self.meta_info[feature_name1]["values"]
                interact_input1 = np.arange(len(interact_input1_original), dtype=np.float32)
                interact_input1_ticks = (interact_input1.astype(int) if len(interact_input1) <= 6 else 
                                 np.linspace(0.1 * len(interact_input1), len(interact_input1) * 0.9, 4).astype(int))
                interact_input1_labels = [interact_input1_original[i] for i in interact_input1_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [str(interact_input1_original[i])[:4] for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(interact_input1_original) - 0.5])
            else:
                sx1 = self.meta_info[feature_name1]["scaler"]
                interact_input1 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input1_original = sx1.inverse_transform(interact_input1.reshape([-1, 1])).ravel()
                interact_input1_ticks = []
                interact_input1_labels = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.categ_variable_list:
                interact_input2_original = self.meta_info[feature_name2]["values"]
                interact_input2 = np.arange(len(interact_input2_original), dtype=np.float32)
                interact_input2_ticks = (interact_input2.astype(int) if len(interact_input2) <= 6 else 
                                 np.linspace(0.1 * len(interact_input2), len(interact_input2) * 0.9, 4).astype(int))
                interact_input2_labels = [interact_input2_original[i] for i in interact_input2_ticks]
                if len("".join(list(map(str, interact_input2_labels)))) > 30:
                    interact_input2_labels = [str(interact_input2_original[i])[:4] for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(interact_input2_original) - 0.5])
            else:
                sx2 = self.meta_info[feature_name2]["scaler"]
                interact_input2 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input2_original = sx2.inverse_transform(interact_input2.reshape([-1, 1])).ravel()
                interact_input2_ticks = []
                interact_input2_labels = []
                interact_input_list.append(interact_input2)
                axis_extent.extend([interact_input2_original.min(), interact_input2_original.max()])
                
            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
            
            interact_outputs = (self.output_layer.interaction_weights[:,class_].numpy()[indice]
                        * self.output_layer.interaction_switcher[:,class_].numpy()[indice]
                        * inter_net.__call__(input_grid, training=False).numpy().reshape(x1.shape))
            data_dict_global.update({feature_name1 + " vs. " + feature_name2:{"type":"pairwise",
                                                       "xtype":feature_type1,
                                                       "ytype":feature_type2,
                                                       "importance":componment_scales[self.input_num + indice],
                                                       "input1":interact_input1_original,
                                                       "input2":interact_input2_original,
                                                       "outputs":interact_outputs,
                                                       "input1_ticks": interact_input1_ticks,
                                                       "input2_ticks": interact_input2_ticks,
                                                       "input1_labels": interact_input1_labels,
                                                       "input2_labels": interact_input2_labels,
                                                       "axis_extent":axis_extent}})

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_global)
            
        return data_dict_global
        
    def local_explain(self, class_, mf_output,x, y=None, save_dict=False, folder="./", name="local_explain"):
        
        predicted = self.predict(x)
        intercept = self.output_layer.main_effect_output_bias[class_].numpy() + self.output_layer.interaction_output_bias[class_].numpy()
        main_effect_output = self.maineffect_blocks.__call__(tf.cast(tf.constant(x), tf.float32)).numpy().ravel()
        if self.interact_num > 0:
            interaction_output = self.interact_blocks.__call__(tf.cast(tf.constant(x), tf.float32)).numpy().ravel()
        else:
            interaction_output = np.array([])

        main_effect_weights = ((self.output_layer.main_effect_weights[:,class_].numpy()) * self.output_layer.main_effect_switcher[:,class_].numpy()).ravel()
        interaction_weights = ((self.output_layer.interaction_weights[:,class_].numpy()[:self.interact_num_added])
                              * self.output_layer.interaction_switcher[:,class_].numpy()[:self.interact_num_added]).ravel()
        interaction_weights = np.hstack([interaction_weights, np.zeros((self.interact_num - self.interact_num_added))]) 

        scores = np.hstack([mf_output,intercept, np.hstack([main_effect_weights, interaction_weights]) 
                                          * np.hstack([main_effect_output, interaction_output])])
        active_indice = 2 + np.hstack([-2,-1, self.active_main_effect_index[class_], self.input_num + self.active_interaction_index[0].astype(int)])
        effect_names = np.hstack(["Latent_effect","Intercept", 
                          np.array(self.variables_names),
                          [self.variables_names[self.interaction_list[i][0]] + " x " 
                          + self.variables_names[self.interaction_list[i][1]] for i in range(len(self.interaction_list))]])
        
        if self.task_type == "Regression":
            data_dict_local = {"active_indice": active_indice.astype(int),
                     "scores": scores,
                     "effect_names": effect_names,
                     "predicted": predicted+mf_output, 
                     "actual": y}
        else:
            data_dict_local = {"active_indice": active_indice.astype(int),
                     "scores": scores,
                     "effect_names": effect_names,
                     "predicted": predicted, 
                     "actual": y}
        
        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_local)

        return data_dict_local