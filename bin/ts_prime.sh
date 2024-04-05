#!/bin/bash
export BASE_DIR=./data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

# Preprocess the dataset
DATASET_ID=722
DATASET_NAME=TSPrime

# python main.py preprocess \
#     --root-dir data/raw/TS_PRIME_3 \
#     --dataset-id $DATASET_ID \
#     --dataset-name $DATASET_NAME

python main.py prep-train \
    --model-fold 0 \
    --gpu-id 0 \
    --model-name TSPrime \
    --dataset-id 722 \
    --determine-postprocessing \
    --train-continue
