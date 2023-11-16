#! /bin/bash

# Set variables
source ./bin/env.sh

#!/bin/bash
export BASE_DIR=./data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

# Preprocess the dataset
DATASET_ID=820
DATASET_NAME=TSGyne

# python ./main.py preprocess \
#     --root-dir data/raw/TS_PRIME_3 \
#     --dataset-id $DATASET_ID \
#     --dataset-name $DATASET_NAME

# echo "Completed Preprocessing of data"

# nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

# Train
nnUNetv2_train $DATASET_ID 3d_fullres 0 -tr nnUNetTrainerNoMirroring

nnUNetv2_predict \
-i $nnUNet_raw/Dataset800_TSGyne/imagesTr \
-o $nnUNet_results/Dataset800_TSGyne/imagesTr_predhighres \
-c 3d_fullres \
-d 800 \
-f 0 \
--verbose \
-chk "checkpoint_best.pth" \
-npp 1 \
-nps 1 \
-num_parts 1 \
-part_id 0 \
-device 'cuda'
