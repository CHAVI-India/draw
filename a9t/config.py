import os

from a9t.utils.mapping import get_model_maps
import logging.config

# Path Configs, change as per Environment
DB_CONFIG = {
    "URL": "sqlite:///data/db_a9t.sqlite",
    "DB_NAME": "a9t",
    "TABLE_NAME": "dicom_log",
}
MODEL_YAML_ROOT_DIR = os.path.normpath("config_yaml")
# DICOM_WATCH_DIR = os.path.normpath("D:\DICOM Database\DICOM IMPORT\dicom")
DICOM_WATCH_DIR = os.path.normpath("data/watch")

# Derived CONFIG
ALL_SEG_MAP, PROTOCOL_TO_MODEL = get_model_maps(MODEL_YAML_ROOT_DIR)
MODEL_CONFIG = {"KEYS": list(ALL_SEG_MAP.keys())}
PRED_BATCH_SIZE = 1
TEMP_DIR_BASE = "temp"
OUTPUT_DIR = "output"
NNUNET_RAW_DATA_ENV_KEY = "nnUNet_raw"
DB_NAME = "db.json"
DEFAULT_FOLD = 0
CSV_FILE_PATH = "data/db/db.csv"
NNUNET_RESULTS_KEY = "nnUNet_results"
RT_DEFAULT_FILE_NAME = "Pred_RT.dcm"
RTSTRUCT_STRING = "RTSTRUCT"
DCM_REGEX = "**/**.dcm"

# Logging
log_config = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "custom_format",
        },
    },
    "formatters": {
        "custom_format": {
            "format": "%(asctime)s [%(levelname)s]: %(message)s",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}

# Configure the logger using the dictionary configuration
logging.config.dictConfig(log_config)

# Create the logger named 'LOG'
LOG = logging.getLogger("LOG")
