#!/bin/bash
export BASE_DIR=./data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

DATASET_ID=721

python ./main.py preprocess \
    --root-dir data/raw/TS_PRIME_3 \
    --dataset-id $DATASET_ID \
    --dataset-name TSPrimeCTVP

echo "Completed Preprocessing of data"

nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
