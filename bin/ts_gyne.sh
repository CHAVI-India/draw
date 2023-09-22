#! /bin/bash

# Set variables
source ./bin/env.sh

# Preprocess the dataset
DATASET_ID=800

# preprocess
# python seg.py \
#     --root_dir /data1/students/sandip/biomedical/code/data/raw/TS_Gyne_2 \
#     --dataset_id $DATASET_ID \
#     --dataset_name TSGyne \
#     --sample_start 9

# # NNUnet preprocess
# nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity

# Train
nnUNetv2_train $DATASET_ID 3d_fullres 0 \
    -tr nnUNetTrainerDiceCELoss_noSmooth
