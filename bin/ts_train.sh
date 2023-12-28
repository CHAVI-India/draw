#!/bin/bash
PREPROCESS=false
VERIFY=false
MODEL_CONFIG="3d_lowres"

# Function to display help information
function display_help {
    echo "Usage: ts_train -n <model_name> -i <dataset_id> -d <cuda_device_id> -p -v"
    echo "Options:"
    echo "  -n    Model name like TSGyne, TSPrime etc..."
    echo "  -i    Dataset id like 820, 920 etc..."
    echo "  -d    Cuda device id, e.g., 0, 1, 2..."
    echo "  -p    Preprocess data (optional). Default False. Boolean Flag"
    echo "  -v    Verify data (optional). Default False. Boolean Flag"
    echo "  -r    Root data path. Eg 'data/raw/TSGyne'. Put Path in double Quotes"
    echo "  -c    Model Config like '3d_lowres', '3d_fullres'. Put in double Quotes"
    echo "  -h    Display this help message"
}

# Parse command line options
while getopts ":n:i:d:r:c:pvh" opt; do
    case $opt in
        n)
            model_name=$OPTARG
            ;;
        i)
            dataset_id=$OPTARG
            ;;
        d)
            cuda_device_id=$OPTARG
            ;;
        p)
            PREPROCESS=true
            ;;
        v)
            VERIFY=true
            ;;
        h)
            display_help
            exit 0
            ;;
        r)
            ROOT_DIR="$OPTARG"
            ;;
        c)
            MODEL_CONFIG="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            display_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            display_help
            exit 1
            ;;
    esac
done

# Validate required options
if [ -z "$model_name" ] || [ -z "$dataset_id" ] || [ -z "$cuda_device_id" ]; then
    echo "Error: Model name, dataset id, and cuda device id are required."
    display_help
    exit 1
fi

# Display selected options
echo "Model Name: $model_name"
echo "Dataset ID: $dataset_id"
echo "Cuda Device ID: $cuda_device_id"
echo "Preprocess: $PREPROCESS"
echo "Verify: $VERIFY"
echo "ROOT DIR: $ROOT_DIR"
echo "MODEL_CONFIG: $MODEL_CONFIG"

# Environment Set
export BASE_DIR=data
export nnUNet_raw=$BASE_DIR/nnUNet_raw
export nnUNet_preprocessed=$BASE_DIR/nnUNet_preprocessed
export nnUNet_results=$BASE_DIR/nnUNet_results
export nnUNet_compile=1

# CLI Args
DATASET_ID=$dataset_id
DATASET_NAME=$model_name

echo -e "\nSTARTING..."

# Preprocess
if ${PREPROCESS}; then
    echo "Starting Preprocessing..."

    python ./main.py preprocess \
    --root-dir "$ROOT_DIR" \
    --dataset-id "$DATASET_ID" \
    --dataset-name "$DATASET_NAME"

    echo "Completed Preprocessing of data"
fi

# Verify and generate plans
if ${VERIFY}; then
    echo "Starting nnUNet Verification..."
    nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity --clean
fi

# Train
echo "Training Started..."
CUDA_VISIBLE_DEVICES=$cuda_device_id nnUNetv2_train "$DATASET_ID" "$MODEL_CONFIG" 0

echo "Process Completed..."
#nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits
