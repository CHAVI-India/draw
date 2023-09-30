import os

TEMP_DIR_BASE = "temp"
OUTPUT_DIR = "output"
NNUNET_RAW_DATA_ENV_KEY = "nnUNet_raw"
CSV_FILE_PATH = "data/db/db.csv"
DB_NAME = "db.csv"
BASE_DIR = os.environ.get(NNUNET_RAW_DATA_ENV_KEY, None)
MODEL_CONFIG = "3d_fullres"
DEFAULT_FOLD = 0
