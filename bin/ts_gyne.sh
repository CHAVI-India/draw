#! /bin/bash

# Set variables
source ./bin/env.sh

# Preprocess the dataset
DATASET_ID=800

# preprocess
# python seg.py \
#     --root_dir /data1/students/sandip/biomedical/code/data/raw/TS_Gyne \
#     --dataset_id $DATASET_ID \
#     --dataset_name TSGyne \
#     --sample_start 0

# NNUnet preprocess
nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity
