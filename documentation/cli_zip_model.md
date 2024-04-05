# CLI Export Command

Convertis a model into a ZIP archive.

## Usage

```bash
python main.py export --dataset-id DATASET_ID --model-name MODEL_NAME
```

## Options

- `--dataset-id`: _(required)_ 3-digit ID for the dataset.

- `--model-name`: _(required)_ Model name. Choose from a predefined set of model names.

## Example

```bash
python main.py export --dataset-id 720 --model-name TSPrime
```

This command will convert the specified model for the given dataset into a ZIP archive.
