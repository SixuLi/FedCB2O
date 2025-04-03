#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

################ EMNIST Dataset ##################

######## DFedAvg for EMNIST dataset #####
for seed in 10 12 16
do
 	python "${SRC}"/main.py \
 	--experiment_name "DFedAvg_emnist_oracle_min_agents_mali_prop_0.2_seed_${seed}" \
 	--data_name 'emnist' \
 	--num_classes 47 \
 	--adversarial \
 	--adversarial_aug_mali \
 	--remove_malicious_agents \
 	--num_local_data 500 \
 	--num_mali_local_data 1200 \
 	--malicious_prop 0.2 \
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