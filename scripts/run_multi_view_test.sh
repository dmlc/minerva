#!/bin/bash

# Provide file paths in this external file

if [ $# -ne 3 ]; then
  echo "Usage: ./run_multi_view_test.sh <solver-path> <snapshot> <num-gpu>"
  echo "Available network type: alex, goognet, vggnet"
  exit 1
fi


solver_file=$1
snapshot=$2
num_gpu=$3

python multiview_test.py $solver_file \
  --snapshot=$snapshot \
  --num_gpu=$num_gpu
