#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

############ CIFAR10 Dataset ###########

######### DFedAvg for CIFAR10 dataset Remove Adversarial (No Agent Selection) #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "DFedAvg_cifar10_fake_malicious_agents_prop_source_class_0.5_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--fake_malicious_agents \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.5 \
# 	--target_class 3 \
# 	--source_class 5 \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 1 \
# 	--N 50 \
# 	--M 25 \
# 	--num_participant_per_round 50 \
# 	--T 301 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--uniform_weight \
# 	--only_weighted_avg \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done


################ EMNIST Dataset ##################

######## DFedAvg for EMNIST dataset #####
for seed in 10 12 16
do
 	python "${SRC}"/main.py \
 	--experiment_name "DFedAvg_emnist_oracle_min_agents_mali_prop_0.3_SC_prop_1.0_seed_${seed}" \
 	--data_name 'emnist' \
 	--num_classes 47 \
 	--adversarial \
 	--adversarial_aug_mali \
 	--remove_malicious_agents \
 	--num_local_data 500 \
 	--num_mali_local_data 1200 \
 	--malicious_prop 0.3 \
 	--prop_source_class 1.0 \
 	--target_class 0 \
 	--source_class 24 \
 	--model_name 'CNN_EMNIST' \
 	--p 1 \
 	--N 50 \
 	--M 20 \
 	--num_participant_per_round 50 \
 	--T 151 \
 	--seed "$seed" \
 	--batch_size 64 \
  --bias \
 	--is_communication \
 	--benign_agent_local_epochs 5 \
 	--optimizer 'SimpleSGD' \
 	--lr 0.004 \
 	--momentum 0.9 \
 	--uniform_weight \
 	--only_weighted_avg \
 	--val_set_prop 0.2 \
 	--gpu_ids '0'
done