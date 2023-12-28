# Predict CLI Command

Generates prediction from Trained NNUNet model

## Usage

```bash
python script_name.py predict \
    --preds-dir OUTPUT_DIR \
    --root-dir ROOT_DIR \
    --dataset-name DATASET_NAME \
    [--only-original]
```

## Options

- `--preds-dir, -p`: _(required)_ Output Directory that will contain the final labels. It must exist and be writable.

- `--root-dir, -r`: _(required)_ Directory containing other DICOM parent directories. It must exist and be readable.

- `--dataset-name, -n`: _(required)_ Name of the dataset. Choose from a predefined set of dataset names.

- `--only-original`: _(optional)_ If present, the script will convert only DICOM images. This disables RTStruct file searching and parsing.

## Example

```bash
python main.py predict -p /path/to/output_dir -r /path/to/dicom_dir -n dataset_name --only-original
```

This command will generate predictions using the trained model for the specified dataset and save the results in the specified output directory. If the `--only-original` flag is present, only original DICOM files will be processed, skipping RTStruct file searching and parsing.
