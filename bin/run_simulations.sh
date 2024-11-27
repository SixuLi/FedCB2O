#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

######### CBO for Bilevel Optimization #########

##### Synthetic Dataset #####

#python "${SRC}"/main.py \
#  --experiment_name "CBO_for_Bilevel_Optimization_1d_hard_beta_0.15" \
#  --N 200 \
#  --M 100 \
#  --T 201 \
#  --Lambda 1 \
#  --Sigma 10 \
#  --Alpha 20 \
#  --Gamma 0.01 \
#  --Beta 0.15 \
#  --num_seeds 1

#python "${SRC}"/main.py \
#  --experiment_name "CBO_for_Bilevel_Optimization_2d_beta_0.25" \
#  --N 200 \
#  --M 100 \
#  --T 1001 \
#  --Lambda 1 \
#  --Sigma 10 \
#  --Alpha 20 \
#  --d 2 \
#  --Gamma 0.01 \
#  --Beta 0.25 \
#  --num_seeds 10

#python "${SRC}"/main.py \
#  --experiment_name "CBO_for_constrained_optimization_2d_beta_0.1" \
#  --data_name 'synthetic_data' \
#  --N 200 \
#  --M 200 \
#  --T 251 \
#  --Lambda 1 \
#  --Sigma 10 \
#  --Alpha 20 \
#  --d 2 \
#  --Gamma 0.01 \
#  --opt_type 'constrained' \
#  --Beta 0.1 \
#  --num_seeds 2


##### CIFAR10 Dataset #####

###### FedCBO for rotated CIFAR10 dataset #####
##### Train with communications
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_rotated_cifar10_full_scale_Temp_0.5_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 2 \
# 	--N 200 \
# 	--M 20 \
# 	--num_participant_per_round 200 \
# 	--num_local_data 500 \
# 	--T 301 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--Temp 0.5 \
# 	--seed "$seed" \
# 	--batch_size 50 \
# 	--test_batch_size 128 \
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
# 	--agent_selection_method 'clustering' \
# 	--moving_avg_alpha 0.5 \
# 	--val_set_prop 0.2 \
# 	--record_time_freq 10 \
# 	--gpu_ids '0'
#done


###### DFedAvg for rotated CIFAR10 dataset #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "DFedAvg_cifar10_full_scale_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 1 \
# 	--N 100 \
# 	--M 10 \
# 	--num_participant_per_round 100 \
# 	--num_local_data 500 \
# 	--T 301 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--Temp 0.25 \
# 	--seed "$seed" \
# 	--batch_size 50 \
# 	--test_batch_size 128 \
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
# 	--moving_avg_alpha 0.5 \
# 	--val_set_prop 0.0 \
# 	--record_time_freq 10 \
# 	--gpu_ids '0'
#done

###### FedCBO for CIFAR10 dataset WITH Label Flipping Attack #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_rotated_cifar10_label_flipping_attack_test_small_scale_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--adversarial \
# 	--malicious_prop 0.2 \
# 	--prop_source_class 1.0 \
# 	--target_class 2 \
# 	--source_class 0 \
# 	--model_name 'CNN' \
# 	--p 1 \
# 	--N 50 \
# 	--M 20 \
# 	--num_participant_per_round 50 \
# 	--T 100 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--local_epochs 5 \
#  --prop_to_full_dataset 0.1 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--only_weighted_avg \
# 	--agent_selection_ranking_type 'clustering' \
# 	--moving_avg_alpha 0.25 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done

####### FedCBO_Bilevel for CIFAR10 dataset WITH Label Flipping Attack (No Agent Selection) #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_Bilevel_cifar10_attack_small_scale_prop_source_class_0.5_seed_${seed}" \
# 	--alg 'FedCBO_Bilevel' \
# 	--data_name 'cifar10' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1200 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.5 \
# 	--target_class 2 \
# 	--source_class 0 \
# 	--model_name 'CNN' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 10 \
# 	--Beta 1.0 \
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
# 	--only_weighted_avg \
# 	--val_set_prop 0.2 \
# 	--record_time_freq 10 \
# 	--gpu_ids '0'
#done

####### FedCBO for rotated CIFAR10 dataset WITH Label Flipping Attack #####
for seed in 10
do
 	python "${SRC}"/main.py \
 	--experiment_name "FedCBO_rotated_cifar10_attack_agent_selection_ranking_type_mali_1000_seed_${seed}" \
 	--data_name 'cifar10' \
 	--alg 'FedCBO' \
 	--adversarial \
 	--adversarial_aug_mali \
 	--num_local_data 500 \
 	--num_mali_local_data 1000 \
 	--malicious_prop 0.3 \
 	--prop_source_class 0.6 \
 	--target_class 3 \
 	--source_class 5 \
 	--model_name 'CNN_CIFAR10' \
 	--p 2 \
 	--N 20 \
 	--M 10 \
 	--num_participant_per_round 20 \
 	--T 201 \
 	--Lambda 1 \
 	--Sigma 0 \
 	--Alpha 10 \
 	--seed "$seed" \
 	--batch_size 50 \
  --bias \
 	--is_communication \
 	--benign_agent_local_epochs 5 \
 	--mali_agent_local_epochs 5 \
 	--optimizer 'SGD' \
 	--lr 0.05 \
 	--momentum 0.9 \
 	--lr_step_size=1 \
 	--lr_scheduler='StepLR' \
 	--lr_gamma=0.99 \
 	--weight_decay=0.0001 \
 	--lr_decay_per_round 0.99 \
 	--record_time_freq 10 \
 	--agent_selection_method 'clustering' \
 	--moving_avg_alpha 0.5 \
 	--val_set_prop 0.2 \
 	--gpu_ids '0'
done

####### FedCBO for rotated CIFAR10 dataset Remove Malicious Agents #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_rotated_cifar10_remove_malicious_agents_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--remove_malicious_agents \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.6 \
# 	--target_class 3 \
# 	--source_class 5 \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 2 \
# 	--N 100 \
# 	--M 20 \
# 	--num_participant_per_round 100 \
# 	--T 201 \
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
# 	--agent_selection_method 'clustering' \
# 	--moving_avg_alpha 0.5 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done


######## FedCBO for CIFAR10 dataset Remove Adversarial Data Points (No Agent Selection) #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_cifar10_remove_adversarial_effect_small_scale_test_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--remove_adversarial_effect \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1500 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--target_class 2 \
# 	--source_class 0 \
# 	--model_name 'CNN' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 101 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--local_epochs 5 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--only_weighted_avg \
# 	--val_set_prop 0.2 \
# 	--record_time_freq 10 \
# 	--gpu_ids '0'
#done

####### FedCBO for CIFAR10 dataset Remove Malicious Agents (No Agent Selection) #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_cifar10_attack_small_scale_prop_source_class_0.5_Alpha_1_remove_malicious_agents_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--remove_malicious_agents \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1200 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.5 \
# 	--target_class 2 \
# 	--source_class 0 \
# 	--model_name 'CNN' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
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
# 	--only_weighted_avg \
# 	--record_time_freq 10 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done


######## DFedAvg for CIFAR10 dataset Remove Adversarial (No Agent Selection) #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "DFedAvg_cifar10_fake_malicious_agents_prop_source_class_0.6_seed_${seed}" \
# 	--data_name 'cifar10' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--fake_malicious_agents \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 0.6 \
# 	--target_class 3 \
# 	--source_class 5 \
# 	--model_name 'CNN_CIFAR10' \
# 	--p 1 \
# 	--N 50 \
# 	--M 20 \
# 	--num_participant_per_round 50 \
# 	--T 201 \
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


##### EMNIST Dataset #####
###### No Malicious Agents #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_emnist_small_scale_seed_${seed}" \
# 	--data_name 'emnist' \
# 	--alg 'FedCBO' \
# 	--num_classes 47 \
# 	--num_local_data 500 \
# 	--prop_source_class 1.0 \
# 	--model_name 'CNN_EMNIST' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 101 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1.0 \
# 	--seed "$seed" \
# 	--batch_size 64 \
# 	--test_batch_size 1000 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--optimizer 'SimpleSGD' \
# 	--lr 0.004 \
# 	--momentum 0.9 \
# 	--only_weighted_avg \
# 	--record_time_freq 5 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done


###### Bilevel FedCBO for EMNIST dataset WITH Label Flipping Attack (No Agent Selection) #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "Bilevel_FedCBO_clustered_emnist_attack_max_diff_small_scale_seed_${seed}" \
# 	--data_name 'emnist' \
# 	--alg 'FedCBO_Bilevel' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_classes 47 \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--source_class 24 \
# 	--target_class 0 \
# 	--model_name 'CNN_EMNIST' \
# 	--p 2 \
# 	--N 20 \
# 	--M 19 \
# 	--num_participant_per_round 20 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 10.0 \
# 	--Beta 0.5 \
# 	--G_func 'max_diff' \
# 	--seed "$seed" \
# 	--batch_size 64 \
# 	--test_batch_size 1000 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SimpleSGD' \
# 	--lr 0.004 \
# 	--momentum 0.9 \
# 	--only_weighted_avg \
# 	--record_time_freq 10 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done

#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "Bilevel_FedCBO_emnist_attack_small_scale_entropy_oracle_seed_${seed}" \
# 	--data_name 'emnist' \
# 	--alg 'FedCBO_Bilevel' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_classes 47 \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--source_class 24 \
# 	--target_class 0 \
# 	--model_name 'CNN_EMNIST' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 10.0 \
# 	--Beta 1.0 \
# 	--G_func 'entropy' \
# 	--is_oracle \
# 	--seed "$seed" \
# 	--batch_size 64 \
# 	--test_batch_size 1000 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SimpleSGD' \
# 	--lr 0.004 \
# 	--momentum 0.9 \
# 	--only_weighted_avg \
# 	--record_time_freq 10 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done

###### FedCBO for EMNIST dataset WITH Label Flipping Attack (No Agent Selection) #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_clustered_emnist_attack_small_scale_Alpha_10_seed_${seed}" \
# 	--data_name 'emnist' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_classes 47 \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--source_class 24 \
# 	--target_class 0 \
# 	--model_name 'CNN_EMNIST' \
# 	--p 2 \
# 	--N 20 \
# 	--M 19 \
# 	--num_participant_per_round 20 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 10.0 \
# 	--seed "$seed" \
# 	--batch_size 64 \
# 	--test_batch_size 1000 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SimpleSGD' \
# 	--lr 0.004 \
# 	--momentum 0.9 \
# 	--only_weighted_avg \
# 	--record_time_freq 10 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done


##### FedCBO for EMNIST dataset Remove Malicious Agents (No Agent Selection) #####
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "FedCBO_emnist_remove_mali_agents_small_scale_Alpha_1.0_debug_seed_${seed}" \
# 	--data_name 'emnist' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--remove_malicious_agents \
# 	--num_classes 47 \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--source_class 24 \
# 	--target_class 0 \
# 	--model_name 'CNN_EMNIST' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1.0 \
# 	--seed "$seed" \
# 	--batch_size 64 \
# 	--test_batch_size 1000 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SimpleSGD' \
# 	--lr 0.004 \
# 	--momentum 0.9 \
# 	--only_weighted_avg \
# 	--record_time_freq 10 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done


##### DFedAvg for EMNIST dataset Fake Malicious Agents (No Agent Selection) #####
#for seed in {10..12}
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "DFedAvg_emnist_fake_mali_agents_small_scale_seed_${seed}" \
# 	--data_name 'emnist' \
# 	--alg 'FedCBO' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--fake_malicious_agents \
# 	--num_classes 47 \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1000 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--source_class 24 \
# 	--target_class 0 \
# 	--model_name 'CNN_EMNIST' \
# 	--p 1 \
# 	--N 10 \
# 	--M 9 \
# 	--num_participant_per_round 10 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1.0 \
# 	--seed "$seed" \
# 	--batch_size 64 \
# 	--test_batch_size 1000 \
#  --bias \
# 	--is_communication \
# 	--benign_agent_local_epochs 5 \
# 	--mali_agent_local_epochs 5 \
# 	--optimizer 'SimpleSGD' \
# 	--lr 0.004 \
# 	--momentum 0.9 \
# 	--uniform_weight \
# 	--record_time_freq 10 \
# 	--val_set_prop 0.2 \
# 	--gpu_ids '0'
#done



###### Dataset initialization test #######
#for seed in 10
#do
# 	python "${SRC}"/main.py \
# 	--experiment_name "dataset_initialization_test" \
# 	--alg 'test' \
# 	--data_name 'cifar10' \
# 	--adversarial \
# 	--adversarial_aug_mali \
# 	--num_local_data 500 \
# 	--num_mali_local_data 1500 \
# 	--malicious_prop 0.3 \
# 	--prop_source_class 1.0 \
# 	--target_class 2 \
# 	--source_class 0 \
# 	--model_name 'CNN' \
# 	--p 2 \
# 	--N 20 \
# 	--M 19 \
# 	--num_participant_per_round 20 \
# 	--T 201 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 20 \
# 	--Beta 0.5 \
# 	--seed "$seed" \
# 	--batch_size 50 \
#  --bias \
# 	--is_communication \
# 	--local_epochs 5 \
# 	--optimizer 'SGD' \
# 	--lr 0.05 \
# 	--momentum 0.9 \
# 	--lr_step_size=1 \
# 	--lr_scheduler='StepLR' \
# 	--lr_gamma=0.99 \
# 	--weight_decay=0.0001 \
# 	--lr_decay_per_round 0.99 \
# 	--only_weighted_avg \
# 	--val_set_prop 0.2 \
# 	--record_time_freq 5 \
# 	--gpu_ids '0'
#done