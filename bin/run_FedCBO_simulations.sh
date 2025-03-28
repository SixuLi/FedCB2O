#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

############ EMNIST Dataset ##########

######## WITH ATTACK ########

####### FedCBO for rotated EMNIST dataset with Agent Selections based on Probability #####
for seed in 10 12 16
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