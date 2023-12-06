from .mapping import ALL_SEG_MAP

DB_CONFIG = {
    # "URL": "mysql+pymysql://root:root@localhost:4000/a9t",
    "URL": "sqlite:///data/db1",
    "DB_NAME": "a9t",
    "TABLE_NAME": "dicomlog",
}

MODEL_CONFIG = {"KEYS": list(ALL_SEG_MAP.keys())}
PRED_BATCH_SIZE=1
