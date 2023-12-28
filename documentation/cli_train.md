# Train Model CLI Command

This CLI command is designed for preparing and training a model on a single GPU. It offers various options to customize the training process based on user requirements.

## Command Usage

```bash
python train_model.py [OPTIONS]
```

## Options

The following options are available to customize the training process:

- **`--model-fold`**: Specify the fold of data to train. Choose from available folds `[0, 1, 2, 3, 4]`.

  - _Type_: Choice
  - _Default_: `0`

- **`--gpu-id`**: Specify the GPU id to use for training.

  - _Type_: Integer
  - _Default_: 0

- **`--model-name`**: Specify the name of the model to be trained. Choose from available model names: **TSPrime, TSGyne**

  - _Type_: Choice
  - _Required_: Yes

- **`--dataset-id`**: Specify the 3-digit dataset ID for training. ID should be present under model name in YAML file

  - _Type_: Integer
  - _Required_: Yes

- **`--gpu-space`**: Specify the GPU space in GB. `WARN Use carefully. If you dont know what you are doing, leave this as is.`

  - _Type_: Integer
  - _Default_: None

- **`--email-address`**: Specify the email address to receive notifications. `WARN: Not implemented yet`

  - _Type_: String
  - _Default_: None

- **`--determine-postprocessing`**: Enable or disable postprocessing determination. If you have no idea, specify this.

  - _Type_: Flag

- **`--train-continue`**: Resume training from where it was left off
  - _Type_: Flag

## Example

```bash
# TSPrime CTVN model on fold 1 GPU id 0
python main.py train-single-gpu \
    --model-fold 1 \
    --gpu-id 0 \
    --model-name TSPrime \
    --dataset-id 722 \
    --gpu-space 8 \
    --email-address user@example.com \
    --determine-postprocessing \
    --train-continue
```
