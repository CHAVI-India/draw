#! /bin/bash

export BASE_DIR=data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

# Preprocess the dataset
DATASET_ID=820
DATASET_NAME=TSGyne
DATA_PATH=data/raw/TSGyneRaw

python ./main.py preprocess \
    --root-dir  $DATA_PATH\
    --dataset-id $DATASET_ID \
    --dataset-name $DATASET_NAME

nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
echo "Completed Preprocessing of data"

echo "Starting DDP Training"
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train $DATASET_ID 3d_lowres 0 -num_gpus 2
