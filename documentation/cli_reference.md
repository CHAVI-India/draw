# CLI Reference

All commands are invoked via `python main.py <command> [options]`.

---

## `preprocess`

Converts raw DICOM data (CT scans + RT-Struct files) into the nnU-Net dataset format for training.

### Usage

```bash
python main.py preprocess \
    --root-dir /path/to/dicom/parent \
    --dataset-id 720 \
    --dataset-name TSPrime \
    [--start 0] \
    [--only-original]
```

### Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--root-dir` | `-d` | Yes | — | Parent directory containing DICOM subdirectories. Each subdirectory should contain one patient's CT series (`.dcm` files) and optionally an RT-Struct file. |
| `--dataset-id` | `-i` | Yes | — | 3-digit nnU-Net dataset ID. Must match the ID in the cancer-site YAML config. |
| `--dataset-name` | `-n` | Yes | — | Cancer-site model name (e.g., `TSPrime`, `TSGyne`). Must match a name in `config_yaml/`. |
| `--start` | `-s` | No | `0` | Sample numbering offset. Use when appending data to an existing dataset. |
| `--only-original` | — | No | `false` | If set, only converts CT images without parsing RT-Struct labels. Use for inference-only preprocessing. |

### What It Does

1. Iterates over each subdirectory in `--root-dir`
2. For each patient directory:
   - Converts DICOM CT → NIfTI image (`imagesTr/seg_NNN_0000.nii.gz`)
   - If RT-Struct present and `--only-original` not set: extracts structure masks → combines into multi-label NIfTI (`labelsTr/seg_NNN.nii.gz`)
3. Generates `dataset.json` (nnU-Net metadata)
4. Records sample-to-DICOM-path mapping in `db.json`

### Output Structure

```
data/nnUNet_raw/Dataset720_TSPrime/
├── imagesTr/
│   ├── seg_000_0000.nii.gz
│   ├── seg_001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── seg_000.nii.gz
│   ├── seg_001.nii.gz
│   └── ...
├── dataset.json
└── db.json
```

### Example

```bash
# Preprocess 48 training cases for prostate OARs
python main.py preprocess \
    -d /data/prostate_training/ \
    -i 720 \
    -n TSPrime

# Preprocess inference-only data (no RT-Struct labels)
python main.py preprocess \
    -d /data/new_patients/ \
    -i 720 \
    -n TSPrime \
    --only-original
```

---

## `train-single-gpu`

Plans, preprocesses (nnU-Net internal), and trains a model on a single GPU. Optionally determines post-processing.

### Usage

```bash
python main.py train-single-gpu \
    --model-name TSPrime \
    --dataset-id 720 \
    [--model-fold 0] \
    [--gpu-id 0] \
    [--gpu-space 8] \
    [--determine-postprocessing] \
    [--train-continue]
```

### Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--model-name` | Yes | — | Cancer-site name (e.g., `TSPrime`). |
| `--dataset-id` | Yes | — | 3-digit dataset ID to train. |
| `--model-fold` | No | `0` | Cross-validation fold (`0`–`4`). |
| `--gpu-id` | No | `0` | CUDA device ID. |
| `--gpu-space` | No | `None` | GPU memory target in GB for nnU-Net planning. Leave unset to use nnU-Net defaults. |
| `--email-address` | No | `None` | Email for completion notification (not yet implemented). |
| `--determine-postprocessing` | No | `false` | After training, run evaluation and determine optimal post-processing (creates `postprocessing.pkl`). |
| `--train-continue` | No | `false` | Resume training from the last checkpoint. |

### Workflow

1. **Plan**: Runs `nnUNetv2_plan_and_preprocess` with dataset integrity verification
2. **Train**: Runs `nnUNetv2_train` with the configured trainer and fold
3. **Evaluate** (if `--determine-postprocessing`): Runs `nnUNetv2_evaluate_folder` on validation predictions
4. **Post-processing** (if `--determine-postprocessing`): Runs `nnUNetv2_determine_postprocessing` and copies the resulting `.pkl` file to the results directory

### Example

```bash
# Train prostate CTVn model, fold 1, on GPU 0
python main.py train-single-gpu \
    --model-name TSPrime \
    --dataset-id 722 \
    --model-fold 1 \
    --gpu-id 0 \
    --determine-postprocessing

# Resume interrupted training
python main.py train-single-gpu \
    --model-name TSPrime \
    --dataset-id 720 \
    --model-fold 0 \
    --train-continue
```

---

## `predict`

Runs inference on a batch of DICOM studies using trained models and outputs DICOM RT-Struct files.

### Usage

```bash
python main.py predict \
    --preds-dir /path/to/output \
    --root-dir /path/to/dicom/studies \
    --dataset-name TSPrime \
    [--only-original]
```

### Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--preds-dir` | `-p` | Yes | — | Output directory for predictions and RT-Struct files. Must exist. |
| `--root-dir` | `-r` | Yes | — | Directory containing DICOM study subdirectories to predict on. |
| `--dataset-name` | `-n` | Yes | — | Cancer-site model name. |
| `--only-original` | — | No | `false` | Skip RT-Struct parsing during preprocessing (use for new/unlabeled data). |

### Workflow

1. Discovers all subdirectories in `--root-dir`
2. For each sub-model in the cancer-site config (run in **parallel** via multiprocessing):
   - Converts DICOM → NIfTI (images only)
   - Runs `nnUNetv2_predict` with TTA disabled
   - Applies post-processing if configured (connected-component filtering)
3. Converts all NIfTI predictions back to a single DICOM RT-Struct per patient
4. Writes `AUTOSEGMENT.RT.dcm` to the output directory

### Output Structure

```
/path/to/output/TSPrime/
├── 720/modelpred/          # Raw NIfTI predictions (OAR model)
├── 721/modelpred/          # Raw NIfTI predictions (CTVp model)
├── 722/modelpred/          # Raw NIfTI predictions (CTVn model)
└── results/
    └── 2024-06-15.14-30/
        └── <series_instance_uid>/
            └── AUTOSEGMENT.RT.dcm
```

### Example

```bash
python main.py predict \
    -p /output/predictions \
    -r /data/new_patients \
    -n TSPrime \
    --only-original
```

---

## `start-pipeline`

Starts the continuous prediction pipeline for clinical deployment. Monitors a directory for incoming DICOM studies and automatically runs inference.

### Usage

```bash
python main.py start-pipeline
```

### Options

None. All configuration is read from `env.draw.yml`.

### Behavior

Spawns two long-lived processes:

1. **Directory Watcher** — Monitors `WATCH_DIR` using `watchdog`. When a new DICOM study directory appears:
   - Waits for file copy to complete (size-stability check, 20s intervals)
   - Reads DICOM `ProtocolName` tag to determine the cancer site
   - Maps protocol → model via YAML config (`protocol` field)
   - Deduplicates by `SeriesInstanceUID`
   - Enqueues a record in the database with status `INIT`

2. **Prediction Consumer** — Continuously cycles through all configured models:
   - Checks GPU memory (requires ≥5 GB free)
   - Dequeues batch of pending records (batch size = 1)
   - Runs prediction with retry (2 attempts, 30s delay)
   - Updates status: `INIT → STARTED → PREDICTED → SENT`

### Stopping

Send `SIGINT` (Ctrl+C) or `SIGTERM`. The watcher process handles `KeyboardInterrupt` gracefully.

### Configuration

All pipeline settings are in `env.draw.yml` and `draw/config.py`:

| Setting | Value | Source |
|---------|-------|--------|
| Watch directory | — | `env.draw.yml` → `WATCH_DIR` |
| GPU memory threshold | 5 GB | `config.py` → `GPU_FREE_GB` |
| Prediction cooldown | 30s | `config.py` → `PREDICTION_COOLDOWN_SECS` |
| GPU recheck interval | 10s | `config.py` → `GPU_RECHECK_TIME_SECONDS` |
| Parallel workers | 2 | `config.py` → `MULTIPROCESSING_NUM_POOL_WORKERS` |
| Batch size | 1 | `config.py` → `PRED_BATCH_SIZE` |

---

## `zip-model`

Exports a trained model as a ZIP archive for deployment to another machine.

### Usage

```bash
python main.py zip-model \
    --dataset-id 720 \
    --model-name TSPrime
```

### Options

| Option | Required | Description |
|--------|----------|-------------|
| `--dataset-id` | Yes | 3-digit dataset ID. |
| `--model-name` | Yes | Cancer-site model name. |

### Output

Creates `Dataset720_TSPrime.zip` in the current directory containing the full contents of:
```
data/nnUNet_results/Dataset720_TSPrime/
```

### Example

```bash
# Export prostate OAR model
python main.py zip-model --dataset-id 720 --model-name TSPrime

# Export prostate CTVn model
python main.py zip-model --dataset-id 722 --model-name TSPrime
```

To import on another machine, extract to `data/nnUNet_results/`.
