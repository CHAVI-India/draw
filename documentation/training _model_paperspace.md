

# DRAW Training in Paperspace Manual

## Introduction

This manual provides step-by-step instructions for training a model on Paperspace. It covers cloning the Git DRAW repository, preparing training data, configuring YAML files, setting up Paperspace machines, preprocessing data, and running training procedures.

## Steps

### 1. Clone the Git DRAW Repository

Clone the DRAW repository from GitHub to your local system using the following command:

```bash
git clone https://github.com/CHAVI-India/draw.git
```
Before moving to step 2, first, check out the docs at https://docs.google.com/document/d/1Obqj3BlNTIeGFuM5Mieg6k1NcoeU48yi0aS8QO5LxYM/edit
### 2. Prepare Training Data

Place the prepared training data inside the `data/raw` folder of the cloned repository.

### 3. Create YAML Configuration File

Create a specific YAML configuration file in the `config_yaml` folder according to your requirements.

### 4. Zip and Upload Folder

Zip the parent folder containing the DRAW repository and upload it using the Paperspace file browser.

### 5. Start Paperspace Machine

Start the Paperspace machine that you intend to use for training.

### 6. Download and Extract Uploaded Zip

Download the uploaded zip file from the Paperspace file browser and extract its contents.

### 7. Install Requirements

Navigate to the root directory of the extracted folder and install all required packages by running:

```bash
pip install -r requirements.txt
```

### 8. Check PyTorch GPU Availability

Check if PyTorch can run on the GPU by executing the following command:

```python
import torch
torch.cuda.is_available()
```

If the result is `True`, PyTorch can utilize the GPU. Otherwise, install PyTorch for CUDA with the specific version.

### 9. Create Screen Session

Create a screen session using the following command, replacing `<model-id>` with the ID obtained from the YAML file:

```bash
screen -S <model-id>
```

### 10. Preprocess Data

If preprocessing hasn't been done on the local system, run the preprocessing code with the appropriate parameters:

```bash
python main.py preprocess -d /path/to/dicom_dir -i 620 -n TSBreast
```

### 11. Run Training

If preprocessing is completed without errors, run the training code using the following command:

```bash
python main.py train-single-gpu \
    --model-fold 0 \
    --gpu-id 0 \
    --model-name TSBreast \
    --dataset-id 620 \
    --determine-postprocessing
```

Note: Adjust GPU settings and specify higher GPU space if necessary using `--gpu-space`.

### 12. Screen Off

Exit the screen session by pressing `Ctrl + d` and then `a`.

### 13. Check Training Log

Check the training log located at `data/nnUNet_results/Dataset620_<>/nnUNetTrainer_<>/fold_0`.

### 14. Repeat Steps 11 to 13

Repeat steps 11 to 13 for preprocessing and training other models based on their IDs, GPU IDs, and root directories.

## Conclusion

This manual provides a systematic guide for training models using the DRAW framework on Paperspace. Follow these steps to efficiently train models for your specific tasks.
