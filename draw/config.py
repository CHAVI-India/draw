import dataclasses
import os

from draw.utils.logging import get_log
from draw.utils.mapping import get_model_maps, get_env_map

YML_ENV = get_env_map()

# Path Configs, change as per Environment
DB_CONFIG = {
    "URL": YML_ENV["DB_URL"],
    "DB_NAME": YML_ENV["DB_NAME"],
    "TABLE_NAME": YML_ENV["TABLE_NAME"],
}
MODEL_YAML_ROOT_DIR = os.path.normpath(YML_ENV["MODEL_DEF_ROOT"])
DICOM_WATCH_DIR = os.path.normpath(YML_ENV["WATCH_DIR"])

# Derived CONFIG
ALL_SEG_MAP, PROTOCOL_TO_MODEL = get_model_maps(MODEL_YAML_ROOT_DIR)
MODEL_CONFIG = {"KEYS": list(ALL_SEG_MAP.keys())}
PRED_BATCH_SIZE = 1
TEMP_DIR_BASE = "temp"
OUTPUT_DIR = "output"
NNUNET_RAW_DATA_ENV_KEY = "nnUNet_raw"
NNUNET_RESULTS_DATA_ENV_KEY = "nnUNet_results"
NNUNET_PREPROCESSED_ENV_KEY = "nnUNet_preprocessed"
RT_DEFAULT_FILE_NAME = "Pred_RT.dcm"
RTSTRUCT_STRING = "RTSTRUCT"
DCM_REGEX = "**/**.dcm"
LOG = get_log()
DEFAULT_MASK_NAME = "draw_default_mask.nii.gz"
SAMPLE_NUMBER_ZFILL = 3


# DCM Tag
@dataclasses.dataclass
class DicomKeyToTag:
    modality: tuple = (0x0008, 0x0060)


DEFAULT_DATASET_TAG = "seg"
SAMPLE_SEP_DELIM = "seg_"
DATASET_JSON_FILENAME = "dataset.json"
PLANS_JSON_FILENAME = "plans.json"
SUMMARY_JSON_FILENAME = "summary.json"
MODEL_FOLDS = ["0", "1", "2", "3", "4"]
