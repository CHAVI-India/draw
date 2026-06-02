# Installation

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (inference requires ~5 GB free VRAM minimum)
- `nvidia-smi` available on PATH (used for GPU memory checks)
- nnU-Net v2 installed and accessible via CLI commands (`nnUNetv2_predict`, `nnUNetv2_train`, etc.)

## Steps

### 1. Clone the Repository

```bash
git clone https://github.com/CHAVI-India/draw.git
cd draw
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `nnunetv2` (custom fork: [Dutta-SD/nnunet_draw](https://github.com/Dutta-SD/nnunet_draw))
- `torch` (with CUDA)
- `click` (CLI framework)
- `sqlalchemy` (database ORM)
- `watchdog` (filesystem monitoring)
- `nibabel` (NIfTI I/O)
- `pydicom` (DICOM I/O)
- `rt_utils` (RT-Struct generation)
- `dcmrtstruct2nii` (RT-Struct → NIfTI conversion)
- `SimpleITK` (medical image processing)
- `retry` (retry logic)

### 3. Configure Environment

Copy the environment template and fill in your values:

```bash
cp template.env.draw.yml env.draw.yml
```

Edit `env.draw.yml`:

```yaml
DB_URL: "sqlite:///draw.db"          # SQLAlchemy connection string
DB_NAME: "draw"                       # Database name
TABLE_NAME: "dicomlog"                # Table name for the prediction queue
WATCH_DIR: "/path/to/dicom/inbox"     # Directory to watch for incoming DICOM studies
MODEL_DEF_ROOT: "config_yaml"         # Path to cancer-site YAML configurations
```

#### Database URL Examples

| Database | URL Format |
|----------|-----------|
| SQLite (local) | `sqlite:///draw.db` |
| PostgreSQL | `postgresql://user:pass@host:5432/draw` |
| MySQL | `mysql+pymysql://user:pass@host:3306/draw` |

### 4. Set Up Data Directories

DRAW uses nnU-Net's standard directory structure. The defaults are:

```bash
mkdir -p data/nnUNet_raw
mkdir -p data/nnUNet_preprocessed
mkdir -p data/nnUNet_results
```

These paths are configured in `draw/accessor/nnunetv2.py`. For non-default locations, modify the `default_nnunet_adapter` instantiation.

### 5. Place Trained Models

Copy trained model checkpoints into the results directory:

```
data/nnUNet_results/
└── Dataset720_TSPrime/
    └── nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/
        ├── fold_0/
        │   └── checkpoint_best.pth
        ├── dataset.json
        └── plans.json
```

Models can be exported/imported as ZIP archives using the `zip-model` command.

### 6. Initialize the Database

On first run with `start-pipeline`, SQLAlchemy will auto-create the required tables. For manual initialization:

```python
from draw.dao.common import Base, DB_ENGINE
Base.metadata.create_all(DB_ENGINE)
```

## Verify Installation

```bash
python main.py --help
```

Expected output:

```
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  AutoSegmentation Pipeline based on NNUNet

Options:
  --help  Show this message and exit.

Commands:
  preprocess       Preprocess DICOM Data to nnUNet format
  predict          Generate Predictions from trained model
  start-pipeline   Starts Continuous Prediction Pipeline
  train-single-gpu Prepare and Train model on a single GPU
  zip-model        Convert model into ZIP archive
```

## Hardware Requirements

### Training

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM | 11+ GB VRAM (GTX 1080 Ti, RTX 3060+) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB per dataset | SSD recommended |

### Inference (Clinical Deployment)

| Component | Spec (Tata Medical Centre) |
|-----------|---------------------------|
| CPU | Intel i7-12700K |
| GPU | NVIDIA T1000 (8 GB) |
| RAM | 32 GB |
| Throughput | ~15 min/case, 300+ scans/day |
