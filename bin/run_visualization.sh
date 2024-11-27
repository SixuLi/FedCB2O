#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

python "${SRC}"/visualization.py \
  --experiment_name 'visualizations' \
  --result_path './results/visualization' \
  --alg 'Bilevel_FedCBO' \
  --data_name 'emnist' \
  --vis_object 'max_diff' \
  --vis_type 'agentwise' \
  --agent_idx 5 \
  --file_path './results/visualization/loss_Bilevel_FedCBO_max_diff_attack_small_scale.txt'