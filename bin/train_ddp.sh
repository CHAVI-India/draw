#!/bin/bash
# Multi-GPU (DistributedDataParallel) training example.
#
# DRAW's `train-single-gpu` command covers single-GPU training. For multi-GPU you
# drive nnU-Net's native DDP trainer directly, as shown here. Adjust the dataset id,
# name, config, and GPU list to your setup.
#
# Usage: bash bin/train_ddp.sh
set -euo pipefail

export nnUNet_raw=data/nnUNet_raw
export nnUNet_preprocessed=data/nnUNet_preprocessed
export nnUNet_results=data/nnUNet_results

DATASET_ID=820
DATASET_NAME=TSGyne
DATA_PATH=data/raw/TSGyneRaw
MODEL_CONFIG=3d_lowres

# 1) DICOM -> nnU-Net dataset (uses the model's seg map from config_yaml/).
draw preprocess \
    --root-dir "$DATA_PATH" \
    --dataset-id "$DATASET_ID" \
    --dataset-name "$DATASET_NAME"

# 2) nnU-Net planning + fingerprinting.
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

# 3) DDP training across 2 GPUs (fold 0).
echo "Starting DDP training on GPUs 0,1"
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_train "$DATASET_ID" "$MODEL_CONFIG" 0 -num_gpus 2
