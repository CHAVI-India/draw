#!/bin/bash
export BASE_DIR=./data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

# Preprocess the dataset
DATASET_ID=920
DATASET_NAME=TSCns

python main.py preprocess \
    --root-dir data/raw/CNSV1 \
    --dataset-id $DATASET_ID \
    --dataset-name $DATASET_NAME

python main.py train-single-gpu \
    --model-fold 0 \
    --gpu-id 0 \
    --model-name $DATASET_NAME\
    --dataset-id $DATASET_ID \
    --determine-postprocessing
