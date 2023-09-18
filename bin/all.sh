#!/bin/bash
export BASE_DIR=./data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results

DATASET_ID=720

python ./preprocess_data.py \
    --root_dir /data1/students/sandip/biomedical/code/data/raw/TS_Prime \
    --dataset_id $DATASET_ID \
    --dataset_name TSPrime \
    --sample_start 0

python ./preprocess_data.py \
    --root_dir /data1/students/sandip/biomedical/code/data/raw/TS_Prime_2 \
    --dataset_id $DATASET_ID \
    --dataset_name TSPrime \
    --sample_start 15

echo "Completed Preprocessing of data"

nnUNetv2_predict \
    -i $nnUNet_raw/Dataset720_TSPrime/imagesTr \
    -o $nnUNet_results/Dataset720_TSPrime/imagesTr_predhighres \
    -c 3d_fullres \
    -d 720 \
    -f 0 \
    -npp 1 \
    -nps 1 \
    -num_parts 1 \
    -part_id 0 \
    -device 'cuda'

python ./nifti2rt.py
