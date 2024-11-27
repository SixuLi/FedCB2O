#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

############ CIFAR10 Dataset ##########

######## WITH ATTACK ########

####### FedCBO for rotated CIFAR10 dataset with Random Agent Selections #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_rotated_cifar10_attack_random_agent_selection_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.6 \
# 	--target_class 3 \
# 	--source_class 5 \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 2 \
# 	--N 50 \
# 	--M 20 \
# 	--num_participant_per_round 50 \
# 	--T 301 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 10 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--record_time_freq 10 \
# 	--only_weighted_avg \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done

####### FedCBO for rotated CIFAR10 dataset with Agent Selections based on Clustering #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_rotated_cifar10_attack_agent_selection_ranking_type_alpha_1_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.6 \
# 	--target_class 3 \
# 	--source_class 5 \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 2 \
# 	--N 20 \
# 	--M 10 \
# 	--num_participant_per_round 20 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--Temp 0.25 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--record_time_freq 10 \
# 	--agent_selection_method 'clustering' \
# 	--moving_avg_alpha 0.5 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done

####### FedCBO for rotated CIFAR10 dataset with Agent Selections based on Probability #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_rotated_cifar10_attack_large_scale_agent_selection_prob_alpha_10_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.5 \
# 	--target_class 3 \
# 	--source_class 5 \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 2 \
# 	--N 100 \
# 	--M 25 \
# 	--num_participant_per_round 100 \
# 	--T 301 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 10 \
# 	--Temp 0.25 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--record_time_freq 10 \
# 	--agent_selection_method 'prob' \
# 	--moving_avg_alpha 0.5 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done



############ EMNIST Dataset ##########

######## WITH ATTACK ########

####### FedCBO for rotated EMNIST dataset with Agent Selections based on Probability #####
for seed in 10 13 16
do
 	python "${SRC}"/main.py \
 	--experiment_name "FedCBO_REMNIST_mali_prop_0.3_mali_data_1200_SC_prop_1.0_seed_${seed}" \
 	--data_name 'emnist' \
 	--num_classes 47 \
 	--alg 'FedCBO' \
 	--adversarial \
 	--adversarial_aug_mali \
 	--num_local_data 500 \
 	--num_mali_local_data 1200 \
 	--malicious_prop 0.3 \
 	--prop_source_class 1.0 \
 	--target_class 0 \
 	--source_class 24 \
 	--model_name 'CNN_EMNIST' \
 	--p 2 \
 	--N 100 \
 	--M 20 \
 	--num_participant_per_round 100 \
 	--T 151 \
 	--Lambda 1 \
 	--Sigma 0 \
 	--Alpha 10 \
 	--Temp 0.5 \
 	--seed "$seed" \
 	--batch_size 64 \
  --bias \
 	--is_communication \
 	--benign_agent_local_epochs 5 \
 	--mali_agent_local_epochs 5 \
 	--optimizer 'SimpleSGD' \
 	--lr 0.004 \
 	--momentum 0.9 \
 	--record_time_freq 10 \
 	--agent_selection_method 'prob' \
 	--moving_avg_alpha 0.5 \
 	--val_set_prop 0.2 \
 	--gpu_ids '0'
done