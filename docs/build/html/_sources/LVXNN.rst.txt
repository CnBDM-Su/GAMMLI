LVXNN
=============
Introduce the API of LVXNN


LVXNN model
----------------

The LVXNN model setting:

meta_info=None, data basic information.

model_info=None, model basic information.

subnet_arch=[10, 6], subnetwork architecture.

interact_arch=[20, 10], interact subnetwork architecture.

activation_func=tf.tanh, activation_function.

lr_bp=0.001, learning rate.

loss_threshold_main=0.01, main_effect loss threshold.

loss_threshold_inter=0.01, interact_effect loss threshold.

main_grid_size=41, number of the sampling points for main_effect training.

interact_grid_size=41, number of the sampling points for interact_effect training.

batch_size=1000, batch size.

main_effect_epochs=10000, main effect training stage epochs.

tuning_epochs=500, tuning stage epochs.

interaction_epochs=20, interact effect training stage epochs.

interact_num=20, the max interact pair number.

interaction_restrict=False, whether restrict the user feature and item feature to interact.

verbose=False, whether show the training state.

early_stop_thres=100, epoch for starting the early stop.

shrinkage_value=None, shrinkage value in each epoch of latent effect training.

convergence_threshold=0.001, convergence threshold for latent effect training.

mf_training_iters=20, main effect training stage epochs.

max_rank=None, max rank for the latent variable.

change_mode = False, whether change the initial value for latent effect training.

u_group_num=0, number of user group.

i_group_num=0, number of item group.

scale_ratio=1, group range shrinkage ratio.

combine_range=0.99, group combination range.

auto_tune=False, whether auto tune.

random_state = 0, random state.

wc = None, control warm or cold start training.


method
----------------
**fit**

LVXNN.fit(tr_x, val_x, tr_y, val_y, tr_Xi, val_Xi, tr_idx, val_idx), training the model.
return the trained model.

**predict**

LVXNN.predict(xx,Xi), predict result.
return predict result.

