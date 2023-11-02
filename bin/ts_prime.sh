#!/bin/bash
export BASE_DIR=./data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

# Preprocess the dataset
DATASET_ID=722
DATASET_NAME=TSPrimeCTVN

python ./main.py preprocess \
    --root-dir data/raw/TS_PRIME_3 \
    --dataset-id $DATASET_ID \
    --dataset-name $DATASET_NAME

echo "Completed Preprocessing of data"

nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

# Train
nnUNetv2_train $DATASET_ID 3d_fullres 0
