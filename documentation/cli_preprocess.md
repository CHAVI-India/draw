# CLI Preprocess Command

Preprocesses DICOM data into the nnUNet format

## Usage

```bash
python main.py preprocess \
    --root-dir ROOT_DIR \
    --dataset-id DATASET_ID \
    --dataset-name DATASET_NAME \
    [--start START] \
    [--only-original]
```

## Options

- `--root-dir, -d`: _(required)_ Parent Directory containing other DICOM directories. It must exist.

- `--dataset-id, -i`: _(required)_ 3-digit ID of the dataset.

- `--dataset-name, -n`: _(required)_ Name of the dataset. Choose from a predefined set of dataset names.

- `--start, -s`: _(optional)_ The sample number to start putting data from. Defaults to 0.

- `--only-original`: _(optional)_ If present, the script will convert only original DICOM files. This disables RTStruct file searching and parsing.

## Example

```bash
python main.py preprocess -d /path/to/dicom_dir -i 720 -n TSPrime --start 5 --only-original
```

This command will preprocess DICOM data into the nnUNet format for the specified dataset and save the results in the appropriate directory. The processing starts from the sample number specified. If the `--only-original` flag is present, only original DICOM files will be processed, skipping RTStruct file searching and parsing.
