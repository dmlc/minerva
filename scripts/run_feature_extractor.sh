#!/bin/bash

# Provide file paths in this external file

if [ $# -ne 5 ]; then
  echo "Usage: ./run_trainer.sh <solver-path> <layer-name> <feature-path> <snapshot> <gpu-idx>"
  echo "Available network type: alex, goognet, vggnet"
  exit 1
fi

solver_file=$1
layer_name=$2
feature_path=$3
snapshot=$4
gpu_idx=$5

python feature_extractor.py $solver_file $layer_name $feature_path\
  --snapshot=$snapshot \
  --gpu_idx=$gpu_idx
